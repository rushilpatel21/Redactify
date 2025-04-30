import sys
import os
import hashlib
import logging
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
CONFIG = {
    "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.68)),  # Increased base threshold
    "context_window": 6,  # Increased context window for better accuracy
    "max_workers": int(os.environ.get("MAX_WORKERS", 8)),  # Thread pool size
    "use_specialized_models": os.environ.get("USE_SPECIALIZED_MODELS", "True").lower() == "true",
    "enable_medical_pii": os.environ.get("ENABLE_MEDICAL_PII", "True").lower() == "true",
    "partial_mask_char": "*",  # Character used for partial masking
    "preserve_formatting": True  # Preserve formatting in redacted output
}

# Common terms that should not be redacted (expanded list)
BLOCKLIST = {
    "Submitted", "Customer", "Issue Description", "Order Number", "Account", 
    "Confirmation", "Attempts", "Reference", "Description", "Screenshots", 
    "Communication", "Number", "Information", "Details", "Subject", "Team",
    "Project", "Request", "Update", "From", "Hi", "Hello", "Dear", "Regards",
    "Best", "Thanks", "Thank you", "Report", "Board", "Contract", "Company",
    "Office", "Employee", "Manager", "Director", "VP", "CEO", "CTO", "CFO",
    "Approved by", "Case Priority", "High", "Medium", "Low", "Internal",
    "External", "Technical", "Model", "Device", "CONFIDENTIAL", "Support",
    "Ticket", "Date", "Phone", "Email", "Contact", "BILLING", "INFORMATION",
    "Expiration", "Security", "Code", "CVV", "DEVICE", "DETAILS", "NOTES",
    "Alternate", "HISTORY", "STATUS", "EMPLOYEE", "Priority"
}

# Add common name words that shouldn't be redacted on their own
COMMON_NAME_WORDS = {
    "Best", "Approved", "Location", "Contact", "Technical", "Internal",
    "University", "City", "State", "Country", "Street", "Avenue", "Street",
    "Customer", "Support", "Service", "Sales", "Marketing", "Priority", "Status"
}

BLOCKLIST.update(COMMON_NAME_WORDS)

# PII options with normalized categories
DEFAULT_PII_OPTIONS = {
    "PERSON": True,
    "ORGANIZATION": True,
    "LOCATION": True,
    "EMAIL_ADDRESS": True,
    "PHONE_NUMBER": True,
    "CREDIT_CARD": True,
    "SSN": True,
    "IP_ADDRESS": True,
    "URL": True,
    "DATE_TIME": True,
    "PASSWORD": True,
    "API_KEY": True,
    "DEPLOY_TOKEN": True,
    "AUTHENTICATION": True,
    "FINANCIAL": True,
    "CREDENTIAL": True,
    "ROLL_NUMBER": True,
    "DEVICE": True,
    "MEDICAL": True,
    "ID_NUMBER": True,
    "MAC_ADDRESS": True  # Added specific MAC address category
}

# Types to pseudonymize in full redaction mode (normalized)
PSEUDONYMIZE_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", 
    "API_KEY", "DEPLOY_TOKEN", "AUTHENTICATION", "MEDICAL"
}

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Initialize Models ---
def load_models():
    logger.info("Loading NLP models...")
    models = {}
    
    try:
        # 1. General NER model
        models["ner_general"] = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english", 
            aggregation_strategy="simple"
        )
        logger.info("Loaded general NER model")
        
        # 2. Specialized PII detection model
        if CONFIG["use_specialized_models"]:
            try:
                models["pii_specialized"] = pipeline(
                    "ner",
                    model="1-13-am/xlm-roberta-base-pii-finetuned",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded specialized PII detection model")
            except Exception as e:
                logger.warning(f"Could not load specialized PII model: {e}")
        
        # 3. Medical PII models if enabled
        if CONFIG["enable_medical_pii"]:
            try:
                models["medical_pii"] = pipeline(
                    "ner",
                    model="obi/deid_roberta_i2b2",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded medical PII model")
            except Exception as e:
                logger.warning(f"Could not load medical PII model: {e}")
                
            # Additional medical model
            try:
                models["medical_reports"] = pipeline(
                    "ner",
                    model="theekshana/deid-roberta-i2b2-NER-medical-reports",
                    aggregation_strategy="simple",
                    model_kwargs={"ignore_mismatched_sizes": True}
                )
                logger.info("Loaded medical reports NER model")
            except Exception as e:
                logger.warning(f"Could not load medical reports NER model: {e}")
                
        # 4. Technical NER model
        try:
            models["ner_tech"] = pipeline(
                "ner",
                model="Jean-Baptiste/roberta-large-ner-english",
                aggregation_strategy="simple"
            )
            logger.info("Loaded technical NER model")
        except Exception as e:
            logger.warning(f"Could not load technical NER model: {e}")
            # Fallback to using the general model
            models["ner_tech"] = models["ner_general"]
        
        # 5. Presidio analyzer for specialized PII detection
        models["presidio"] = AnalyzerEngine()
        logger.info("Loaded Presidio Analyzer")
        
        logger.info(f"Successfully loaded {len(models)} models")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        if not models:
            raise RuntimeError("Failed to load any models, cannot continue")
    
    return models

MODELS = load_models()

# --- Enhanced Regex Patterns ---
REGEX_PATTERNS = [
    # SSN patterns
    {"type": "SSN", "pattern": r"\b\d{3}-\d{2}-\d{4}\b", "context": ["ssn", "social security", "social"]},
    
    # IP address patterns
    {"type": "IP_ADDRESS", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "context": ["ip", "address", "server", "host"]},
    
    # MAC address patterns - new and improved
    {"type": "MAC_ADDRESS", "pattern": r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", "context": []},
    {"type": "MAC_ADDRESS", "pattern": r"\b([0-9A-Fa-f]{2}[.]){5}[0-9A-Fa-f]{2}\b", "context": ["mac", "address", "ethernet"]},
    
    # URL patterns
    {"type": "URL", "pattern": r"\bhttps?://[^\s]+\b", "context": []},
    {"type": "URL", "pattern": r"\b(?:www\.)[a-z0-9-]+(?:\.[a-z]{2,})+(?:/[^\s]*)?", "context": []},
    {"type": "URL", "pattern": r"\b[a-z0-9-]+\.[a-z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?", "context": ["http", "https", "web", "site", "portal", "access"]},
    
    # Date patterns - improved formatting detection
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "context": []},
    {"type": "DATE_TIME", "pattern": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", "context": []},
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}/\d{2}\b", "context": ["exp", "expiration", "valid", "until"]},  # MM/YY format
    {"type": "DATE_TIME", "pattern": r"\b\d{4}-\d{2}-\d{2}\b", "context": []},  # YYYY-MM-DD format
    {"type": "DATE_TIME", "pattern": r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b", "context": []},  # YYYY-MM-DD HH:MM:SS
    {"type": "DATE_TIME", "pattern": r"\b\d{2}/\d{2}/\d{4}\b", "context": []},  # MM/DD/YYYY format
    
    # Phone number patterns - improved
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{10}\b", "context": ["phone", "mobile", "cell", "tel", "telephone", "contact"]},
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "context": []},
    {"type": "PHONE_NUMBER", "pattern": r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b", "context": []},
    {"type": "PHONE_NUMBER", "pattern": r"\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "context": []},  # International format
    
    # Password patterns - improved precision
    {"type": "PASSWORD", "pattern": r"(?i)(?:password|passwd|pwd)(?::|=|\s+is\s+)\s*(\S+)", "context": []},
    {"type": "PASSWORD", "pattern": r"(?i)password(?:\s+was|\s+has\s+been)?\s+(?:reset|changed)(?:\s+to)?\s+(\S+)", "context": []},
    {"type": "PASSWORD", "pattern": r"(?=.*[A-Za-z])(?=.*\d)(?=.*[$#@!%^&*()_+])[A-Za-z\d$#@!%^&*()_+]{8,}", 
     "context": ["password", "pass", "pwd", "credential", "login", "auth", "secret", "temporary", "temp"]},
    
    # Credit card patterns - improved
    {"type": "CREDIT_CARD", "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b", "context": []},
    {"type": "CREDIT_CARD", "pattern": r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", "context": []},
    {"type": "CREDIT_CARD", "pattern": r"credit card:?\s*\**\d{4}", "context": []},  # Last 4 digits with prefix
    
    # CVV/CVC patterns - improved
    {"type": "FINANCIAL", "pattern": r"\bCVV:?\s*(\d{3,4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bCVC:?\s*(\d{3,4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bsecurity\s+code:?\s*(\d{3,4})\b", "context": []},
    
    # API key patterns - enhanced
    {"type": "API_KEY", "pattern": r"(?i)api[_-]?key(?::|=|\s+is\s+)\s*([A-Za-z0-9\-_\.]{8,})\b", "context": []},
    {"type": "API_KEY", "pattern": r"(?i)(?:api|app|access)[_-]?(?:key|token|secret|id)(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "API_KEY", "pattern": r"\b[A-Za-z0-9_\-]{20,40}\b", 
     "context": ["api", "key", "secret", "token", "auth", "access", "credentials"]},
    
    # JWT/Auth token patterns - improved
    {"type": "AUTHENTICATION", "pattern": r"ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*", "context": []},  # JWT format
    
    # Deploy token patterns
    {"type": "DEPLOY_TOKEN", "pattern": r"gh[pousr]_[A-Za-z0-9_]{16,}\b", "context": []},  # GitHub tokens
    {"type": "DEPLOY_TOKEN", "pattern": r"(?i)(?:deploy|access|auth|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Authentication patterns
    {"type": "AUTHENTICATION", "pattern": r"(?i)(?:bearer|basic|digest|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)auth(?:entication)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)credential(?:s)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"session\s+key:?\s*\S+", "context": []},  # Session keys
    
    # Financial information - enhanced with card details
    {"type": "FINANCIAL", "pattern": r"\brouting[:\s]+(\d{9})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\baccount\s+(?:number|#)?[:\s]+(\d+)\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\b(?:account|acct)(?:.+?)ending in (\d{4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"ending in \d{4}", "context": ["card", "account"]},
    {"type": "FINANCIAL", "pattern": r"card \(ending in \d{4}", "context": []},
    
    # Bank account numbers
    {"type": "FINANCIAL", "pattern": r"(?:bank|checking|savings)\s+account:?\s*(\d{8,})", "context": []},
    {"type": "FINANCIAL", "pattern": r"routing\s+number:?\s*(\d{8,})", "context": []},
    
    # Student roll number patterns
    {"type": "ROLL_NUMBER", "pattern": r"\b\d{2}[A-Za-z]{3}\d{3}\b", "context": ["student", "roll", "enrollment"]},
    {"type": "ROLL_NUMBER", "pattern": r"\b(?:roll|enrollment|student)(?:.+?)(?:number|no|#)?[:\s]+([A-Za-z0-9\-]{5,10})\b", "context": []},
    
    # Username patterns - improved precision
    {"type": "CREDENTIAL", "pattern": r"\busername[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\blogin[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\buser(?:name)?[:\s]+(\S+)\b", "context": []},
    
    # Device information - improved
    {"type": "DEVICE", "pattern": r"(?:iPhone|iPad|MacBook|Android|Windows|Device)\s+(?:\w+\s+)?\w+", 
     "context": ["device", "model", "using", "on"]},
    {"type": "DEVICE", "pattern": r"Serial\s+Number:?\s+([A-Z0-9]{5,})", "context": []},
    
    # Order/Account identifiers - improved specificity
    {"type": "ID_NUMBER", "pattern": r"(?:Order|Invoice)(?:\s+(?:Number|#|ID|No\.?)):\s*([A-Za-z0-9\-]+)", "context": []},
    {"type": "ID_NUMBER", "pattern": r"(?:Customer|Account)(?:\s+(?:ID|#|No\.?)):\s*([A-Za-z0-9\-]+)", 
     "context": ["customer", "account", "id", "number"]},
     
    # Medical record identifiers
    {"type": "MEDICAL", "pattern": r"\b(?:patient|medical|health|record)\s+(?:id|number|#):\s*([A-Za-z0-9\-]+)", "context": []},
    {"type": "MEDICAL", "pattern": r"\b(?:MRN|PHN)(?::|#|\s+number)?\s*:?\s*([A-Za-z0-9\-]+)", "context": []},
    {"type": "MEDICAL", "pattern": r"Medical Insurance ID:?\s*([A-Za-z0-9\-]+)", "context": []},
    {"type": "MEDICAL", "pattern": r"Provider ID:?\s*([A-Za-z0-9\-]+)", "context": []},
    
    # ID numbers and government identifiers
    {"type": "ID_NUMBER", "pattern": r"\b(?:passport|driver|license|id)\s+(?:number|#):\s*([A-Za-z0-9\-]+)", "context": []},
    {"type": "ID_NUMBER", "pattern": r"\b[A-Z]{1,2}[0-9]{6,9}\b", "context": ["passport", "government", "license", "identification"]},
    {"type": "ID_NUMBER", "pattern": r"Employee\s+ID:?\s*([A-Za-z0-9\-]+)", "context": []},
]

# --- Entity Normalization and Mapping ---
# Simplified and normalized entity type mapping
ENTITY_TYPE_MAPPING = {
    # Person entities
    "PERSON": "PERSON",
    "PER": "PERSON",
    "PEOPLE": "PERSON",
    "PERSONAL": "PERSON",
    "INDIVIDUAL": "PERSON",
    "NAME": "PERSON",
    "NAME_STUDENT": "PERSON",
    "PATIENT": "PERSON",
    "STAFF": "PERSON",
    "DOCTOR": "PERSON",
    
    # Organization entities
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "CORPORATION": "ORGANIZATION",
    "BUSINESS": "ORGANIZATION",
    "PATORG": "ORGANIZATION",
    "HOSP": "ORGANIZATION",
    
    # Location entities
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "LOCATION",
    "PLACE": "LOCATION",
    "STREET": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "ZIP": "LOCATION",
    "ZIPCODE": "LOCATION",
    "POSTAL_CODE": "LOCATION",
    
    # Email entities
    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "MAIL": "EMAIL_ADDRESS",
    
    # Phone entities
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "TEL": "PHONE_NUMBER",
    "TELEPHONE": "PHONE_NUMBER",
    "MOBILE": "PHONE_NUMBER",
    "CELL": "PHONE_NUMBER",
    
    # Payment card entities
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT": "CREDIT_CARD",
    "CC": "CREDIT_CARD",
    "PAYMENT_CARD": "CREDIT_CARD",
    "CARD_NUMBER": "CREDIT_CARD",
    "PAN": "CREDIT_CARD",
    
    # SSN entities
    "SSN": "SSN",
    "SOCIAL_SECURITY": "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    
    # IP address entities
    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    
    # MAC address entities
    "MAC": "MAC_ADDRESS",
    "MAC_ADDRESS": "MAC_ADDRESS",
    
    # URL entities
    "URL": "URL",
    "URI": "URL",
    "WEBSITE": "URL",
    "LINK": "URL",
    "WEB": "URL",
    
    # Date entities
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "DATE_TIME": "DATE_TIME",
    "DATETIME": "DATE_TIME",
    
    # Password entities
    "PASSWORD": "PASSWORD",
    "PWD": "PASSWORD",
    "PASSWD": "PASSWORD",
    "PASSCODE": "PASSWORD",
    
    # API key entities
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "KEY": "API_KEY",
    "SECRET_KEY": "API_KEY",
    
    # Token entities
    "TOKEN": "DEPLOY_TOKEN",
    "DEPLOY_TOKEN": "DEPLOY_TOKEN",
    "ACCESS_TOKEN": "DEPLOY_TOKEN",
    "SECRET_TOKEN": "DEPLOY_TOKEN",
    "OAUTH_TOKEN": "DEPLOY_TOKEN",
    
    # Authentication entities
    "AUTH": "AUTHENTICATION",
    "AUTHENTICATION": "AUTHENTICATION",
    "BEARER": "AUTHENTICATION",
    "SESSION": "AUTHENTICATION",
    
    # Credential entities
    "CREDENTIAL": "CREDENTIAL",
    "LOGIN": "CREDENTIAL",
    "USERNAME": "CREDENTIAL",
    "USER": "CREDENTIAL",
    
    # Financial entities
    "FINANCIAL": "FINANCIAL",
    "ACCOUNT": "FINANCIAL",
    "ROUTING": "FINANCIAL",
    "BANK": "FINANCIAL",
    "ACCOUNT_NUMBER": "FINANCIAL",
    "ROUTING_NUMBER": "FINANCIAL",
    "CVV": "FINANCIAL",
    "CVC": "FINANCIAL",
    
    # Student ID entities
    "ROLL_NUMBER": "ROLL_NUMBER",
    "ENROLLMENT": "ROLL_NUMBER",
    "STUDENT_ID": "ROLL_NUMBER",
    
    # Device entities
    "DEVICE": "DEVICE",
    
    # Product entities
    "PRODUCT": "PRODUCT",
    
    # ID number entities
    "ID_NUMBER": "ID_NUMBER",
    "DRIVER_LICENSE": "ID_NUMBER",
    "PASSPORT": "ID_NUMBER",
    "LICENSE_NUMBER": "ID_NUMBER",
    "ID": "ID_NUMBER",
    
    # Medical entities
    "MEDICAL": "MEDICAL",
    "PATIENT_ID": "MEDICAL",
    "HEALTH_ID": "MEDICAL",
    "MEDICAL_RECORD": "MEDICAL",
    "MRN": "MEDICAL",
    "PHN": "MEDICAL",
    "DIAGNOSIS": "MEDICAL",
    "CONDITION": "MEDICAL",
    "PROCEDURE": "MEDICAL",
    "HOSPITAL": "MEDICAL",
    "PROVIDER_NUMBER": "MEDICAL",
    
    # Skip miscellaneous entities - they're often false positives
    "MISC": None,
    "O": None,
}

# --- Masking Functions ---

def pseudonymize_value(value: str, entity_type: str) -> str:
    """Generate a consistent hash-based pseudonym for a given value."""
    h = hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    return f"[{entity_type}-{h}]"

def full_mask_token(token: str, entity_type: str) -> str:
    """For full redaction: return the hash‐based pseudonym with normalized entity type."""
    if entity_type is None:
        return '*' * len(token)
    return pseudonymize_value(token, entity_type.upper())

def partial_mask_token(token: str) -> str:
    """
    Improved partial masking function that preserves more meaningful parts
    of the token based on its length and type.
    """
    n = len(token)
    mask_char = CONFIG["partial_mask_char"]
    
    # Handle very short tokens
    if n <= 2:
        return mask_char * n
    
    # Handle short tokens (3-5 chars)
    elif n <= 5:
        return token[0] + mask_char * (n - 1)
    
    # Handle medium tokens (6-10 chars)
    elif n <= 10:
        return token[0:2] + mask_char * (n - 4) + token[-2:]
    
    # Handle longer tokens
    else:
        return token[0:2] + mask_char * (n - 5) + token[-3:]

def mask_email(email: str) -> str:
    """
    Improved email address masking that preserves some recognizability
    while ensuring privacy.
    """
    try:
        local, domain = email.split("@")
    except Exception as e:
        logger.error(f"Error splitting email '{email}': {e}")
        return partial_mask_token(email)
    
    # Mask local part
    if len(local) > 4:
        local_masked = local[0:2] + CONFIG["partial_mask_char"] * (len(local) - 4) + local[-2:]
    else:
        local_masked = local[0] + CONFIG["partial_mask_char"] * (len(local) - 1)
    
    # Mask domain part
    domain_parts = domain.split('.')
    
    if len(domain_parts) > 1:
        # Preserve the TLD (e.g., .com, .org)
        tld = domain_parts[-1]
        domain_name = '.'.join(domain_parts[:-1])
        
        if len(domain_name) > 5:
            domain_masked = domain_name[0:2] + CONFIG["partial_mask_char"] * (len(domain_name) - 2)
        else:
            domain_masked = CONFIG["partial_mask_char"] * len(domain_name)
            
        masked_domain = domain_masked + '.' + tld
    else:
        masked_domain = CONFIG["partial_mask_char"] * len(domain)
    
    return local_masked + "@" + masked_domain

def mask_url(url: str) -> str:
    """For full redaction: return a hash-based pseudonym for the URL."""
    return full_mask_token(url, "URL")

def partial_mask_url(url: str) -> str:
    """
    Improved URL masking that preserves structure but masks
    private information within domains and paths.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.error(f"Error parsing URL '{url}': {e}")
        return partial_mask_token(url)
        
    scheme, netloc, path, params, query, fragment = (
        parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
    )
    
    # Process netloc (handle port numbers)
    if ':' in netloc:
        domain, port = netloc.split(':', 1)
        port = ':' + port
    else:
        domain, port = netloc, ''
    
    # Process domain parts
    parts = domain.split('.')
    masked_parts = []
    
    for i, part in enumerate(parts):
        # Keep the TLD intact if it's the last part
        if i == len(parts) - 1 and len(parts) > 1:
            masked_parts.append(part)
        # Mask subdomain and domain parts
        elif len(part) > 3:
            masked_parts.append(part[0:2] + CONFIG["partial_mask_char"] * (len(part) - 2))
        else:
            masked_parts.append(CONFIG["partial_mask_char"] * len(part))
    
    masked_netloc = '.'.join(masked_parts) + port
    
    # Process path segments (more carefully)
    if path:
        path_segments = path.split('/')
        masked_segments = []
        
        for segment in path_segments:
            if not segment:  # Handle empty segments
                masked_segments.append(segment)
                continue
                
            # Keep common path elements like 'api', 'dashboard', etc.
            if segment.lower() in ['api', 'v1', 'v2', 'v3', 'dashboard', 'login', 'public', 'static']:
                masked_segments.append(segment)
            elif len(segment) >= 5:
                masked_segments.append(segment[0:2] + CONFIG["partial_mask_char"] * (len(segment) - 2))
            else:
                masked_segments.append(CONFIG["partial_mask_char"] * len(segment))
                
        masked_path = '/'.join(masked_segments)
    else:
        masked_path = path
    
    # Don't mask query parameters and fragments for simplicity
    return urlunparse((scheme, masked_netloc, masked_path, params, query, fragment))

def mask_phone(phone: str) -> str:
    """Specialized function for partial masking of phone numbers."""
    # Remove common formatting characters
    digits_only = re.sub(r'[^0-9+]', '', phone)
    
    if len(digits_only) <= 4:
        return CONFIG["partial_mask_char"] * len(phone)
    
    # Handle international prefix if present
    if digits_only.startswith('+'):
        prefix_end = digits_only.find('9')
        if prefix_end != -1 and prefix_end < 4:  # Valid country code
            prefix = digits_only[:prefix_end+1]
            main_number = digits_only[prefix_end+1:]
        else:
            prefix = '+'
            main_number = digits_only[1:]
    else:
        prefix = ''
        main_number = digits_only
    
    # Keep last 4 digits visible, mask the rest
    if len(main_number) > 4:
        masked_number = CONFIG["partial_mask_char"] * (len(main_number) - 4) + main_number[-4:]
    else:
        masked_number = CONFIG["partial_mask_char"] * len(main_number)
    
    # Reconstruct with original formatting
    masked_digits = prefix + masked_number
    
    # Reapply original format
    result = ''
    digit_index = 0
    for char in phone:
        if char.isdigit() or char == '+':
            result += masked_digits[digit_index]
            digit_index += 1
        else:
            result += char  # Preserve formatting characters
    
    return result

def smart_partial_mask(text: str, entity_type: str) -> str:
    """
    Apply the appropriate partial masking strategy based on entity type.
    This ensures consistent masking across different PII types.
    """
    if not text:
        return text
        
    # Apply specialized masking based on entity type
    if entity_type == "EMAIL_ADDRESS":
        return mask_email(text)
    elif entity_type == "URL":
        return partial_mask_url(text)
    elif entity_type == "PHONE_NUMBER":
        return mask_phone(text)
    elif entity_type == "CREDIT_CARD":
        # Only show last 4 digits for credit cards
        digits_only = re.sub(r'[^0-9]', '', text)
        if len(digits_only) >= 4:
            return CONFIG["partial_mask_char"] * (len(text) - 4) + text[-4:]
    elif entity_type == "SSN":
        # Format like ***-**-1234 (last 4 visible)
        if len(text) > 4:
            return CONFIG["partial_mask_char"] * (len(text) - 4) + text[-4:]
    elif entity_type in ["PASSWORD", "API_KEY", "AUTHENTICATION", "DEPLOY_TOKEN"]:
        # Extra security for sensitive credentials - show very little
        if len(text) > 8:
            return text[:2] + CONFIG["partial_mask_char"] * (len(text) - 2)
        else:
            return CONFIG["partial_mask_char"] * len(text)
    elif entity_type == "DATE_TIME":
        # Only mask year for dates if long enough
        if len(text) > 6 and re.search(r'\d{4}', text):
            date_parts = re.split(r'[-/\s:]', text)
            if len(date_parts) > 2:
                # Likely YYYY-MM-DD or similar
                date_parts[0] = CONFIG["partial_mask_char"] * len(date_parts[0])
                return re.sub(r'[-/\s:]', lambda m: m.group(0), '-'.join(date_parts))
    
    # Default to standard partial masking
    return partial_mask_token(text)

# --- Context-Aware Detection ---

def has_context(text, span_start, span_end, context_words):
    """
    Enhanced context detection to reduce false positives.
    Checks if context words appear near the entity or if there are contextual indicators.
    """
    if not context_words:
        return True  # No context check needed
    
    # Get the matched text
    matched_text = text[span_start:span_end]
    
    # Don't match blocklisted terms
    if matched_text in BLOCKLIST:
        return False
        
    # Special case for projects that might be flagged as people
    if matched_text.startswith("Project") and len(matched_text.split()) <= 2:
        return False
    
    # Extract a window of text before and after the entity
    context_size = CONFIG["context_window"]
    text_before = text[:span_start].split()[-context_size:] if span_start > 0 else []
    text_after = text[span_end:].split()[:context_size] if span_end < len(text) else []
    
    # Check for context words
    context_text = ' '.join(text_before + text_after).lower()
    
    # For potential passwords and other sensitive data, require stronger context
    high_risk_types = ["password", "key", "token", "secret", "credential", "api"]
    needs_strong_context = len(matched_text) > 10 and re.search(r'[A-Za-z0-9]{10,}', matched_text)
    
    for word in context_words:
        if word.lower() in context_text:
            # For high-risk patterns, require stronger context
            if needs_strong_context and not any(h_risk in context_text for h_risk in high_risk_types):
                continue
            return True
            
    # Check if the match is near a colon, equals sign, or other indicators
    nearby_text = text[max(0, span_start-20):min(len(text), span_end+10)]
    indicators = r'(?::|=|is\s+|was\s+reset\s+to\s+)'
    if re.search(indicators + r'\s*' + re.escape(matched_text), nearby_text, re.IGNORECASE):
        return True
        
    return False

# --- Entity Verification ---

def verify_entity(entity_type, text, confidence_score):
    """
    Enhanced verification to filter out false positives.
    Applies specialized rules for different entity types.
    """
    # Check if text is in blocklist
    if text.strip() in BLOCKLIST:
        return False
    
    # Skip None entity types
    if entity_type is None:
        return False
    
    # Check for common project names (which are often false positives)
    if entity_type == "PERSON" and text.startswith("Project"):
        return False
        
    # Check for generic team references
    if entity_type == "PERSON" and text.lower() in ["team", "hi team", "hello team"]:
        return False
    
    # Reject specific structural patterns that are likely false positives
    if (
        (text.isdigit() and len(text) < 4) or  # Short numeric strings
        text.startswith("#") or              # Section headers/references
        (len(text.split()) == 1 and len(text) < 4 and not re.search(r'\d', text)) or  # Very short single words
        text.lower() in ["from", "to", "hi", "hello", "subject", "best", "regards"]  # Common email parts
    ):
        return False
    
    # Reject standalone common first names in certain contexts
    if entity_type == "PERSON" and len(text.split()) == 1 and text in COMMON_NAME_WORDS:
        return False
        
    # Updated confidence thresholds for different entity types
    type_confidence_thresholds = {
        "PASSWORD": 0.75,      # Higher threshold for passwords
        "API_KEY": 0.75,       # Higher for API keys
        "CREDENTIAL": 0.75,    # Higher for credentials
        "FINANCIAL": 0.7,      # Higher for financial data
        "ORGANIZATION": 0.65,  # Normal for organizations
        "DEVICE": 0.75,        # Higher for devices (without identifiers)
        "MEDICAL": 0.7,        # Higher for medical data
        "PERSON": 0.68         # Slightly higher for person names
    }
    
    min_confidence = type_confidence_thresholds.get(entity_type, CONFIG["confidence_threshold"])
    if confidence_score < min_confidence:
        return False
        
    # Length-based checks for different entity types
    if entity_type == "PASSWORD":
        # Password must have clear indicators or be complex
        has_digits = bool(re.search(r'\d', text))
        has_letters = bool(re.search(r'[a-zA-Z]', text))
        has_special = bool(re.search(r'[^a-zA-Z0-9\s]', text))
        
        # If it doesn't look like a complex password, reject it
        if not ((has_digits and has_letters) or (has_special and (has_digits or has_letters))):
            return False
        
    elif entity_type == "API_KEY":
        # API keys should be long and have a mix of characters
        if len(text) < 12:
            return False
            
    # Entity-type specific rejections
    if entity_type == "CREDENTIAL" and text.lower() in ["user", "account", "login"]:
        return False
        
    if entity_type == "DEVICE" and text in ["iPhone", "MacBook", "Android", "Windows"]:
        # Only redact devices when paired with identifiers
        return bool(re.search(r'[a-zA-Z]+\s+[a-zA-Z0-9]+', text))
            
    return True

# --- Detection Functions ---

def normalize_entity(entity: dict) -> str:
    """
    Normalize entity types across different detectors.
    Returns a consistent entity type for all detectors.
    """
    if 'entity_group' in entity:
        raw_type = entity['entity_group'].upper()
    elif 'entity' in entity:
        raw_type = entity['entity'].upper()
        if raw_type.startswith("B-") or raw_type.startswith("I-"):
            raw_type = raw_type[2:]
    else:
        return None
        
    return ENTITY_TYPE_MAPPING.get(raw_type, raw_type)

def get_presidio_entities(text: str) -> list:
    """Detect entities using Microsoft Presidio."""
    try:
        results = MODELS["presidio"].analyze(text=text, language="en")
        presidio_entities = [{
            'entity_group': res.entity_type,
            'start': res.start,
            'end': res.end,
            'score': res.score,
            'detector': 'presidio'
        } for res in results]
        return presidio_entities
    except Exception as e:
        logger.error(f"Error in Presidio Analyzer: {e}")
        return []

def get_ner_entities(text: str, model_key: str) -> list:
    """Detect entities using a Hugging Face NER model."""
    try:
        results = MODELS[model_key](text)
        # Add detector information to results
        for item in results:
            item['detector'] = model_key
        return results
    except Exception as e:
        logger.error(f"Error in {model_key} pipeline: {e}")
        return []

def get_regex_entities(text: str) -> list:
    """
    Enhanced regex-based detection with context awareness
    and specialized pattern matching.
    """
    regex_entities = []
    
    # Special handling for MAC addresses
    mac_addresses = re.finditer(r'\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b', text)
    for match in mac_addresses:
        start, end = match.span()
        regex_entities.append({
            "entity_group": "MAC_ADDRESS",
            "start": start,
            "end": end,
            "score": 0.95,
            "detector": "regex"
        })
    
    # Process all other regex patterns
    for pattern_def in REGEX_PATTERNS:
        for match in re.finditer(pattern_def["pattern"], text, re.IGNORECASE):
            start, end = match.span()
            matched_text = text[start:end]
            
            # Skip very short matches unless they have context
            if len(matched_text) < 4 and not pattern_def.get("context"):
                continue
            
            # Skip matches of common words unless they have specific context
            if matched_text.lower() in ["customer", "submitted", "description", "reference"]:
                continue
                
            # Check for contextual clues
            if has_context(text, start, end, pattern_def.get("context", [])):
                # Use capture group if available, otherwise use the whole match
                if match.groups() and pattern_def.get("use_group", True):
                    group_start = match.start(1)
                    group_end = match.end(1)
                    regex_entities.append({
                        "entity_group": pattern_def["type"],
                        "start": group_start,
                        "end": group_end,
                        "score": 0.9,  # High confidence for regex with context
                        "detector": "regex"
                    })
                else:
                    regex_entities.append({
                        "entity_group": pattern_def["type"],
                        "start": start,
                        "end": end,
                        "score": 0.9,  # High confidence for regex with context
                        "detector": "regex"
                    })
    
    return regex_entities

def score_entity(entity: dict, text: str) -> float:
    """
    Enhanced scoring function for entities based on content and context.
    """
    score = entity.get('score', 0.6)
    entity_text = text[entity['start']:entity['end']]
    entity_type = normalize_entity(entity)
    
    # Skip banned terms
    if entity_text in BLOCKLIST or entity_type is None:
        return 0.0
    
    # Skip certain general terms that are often false positives
    if entity_text.lower() in ["project", "team", "update", "request", "from", "subject"] and entity_type == "PERSON":
        return 0.0
    
    # Boost confidence for entities detected by multiple systems
    if entity.get('detected_by', 1) > 1:
        score = min(1.0, score + 0.1 * entity['detected_by'])
    
    # Penalty for very short entities unless they're special types
    if len(entity_text) < 4 and entity_type not in ["SSN", "DATE_TIME", "FINANCIAL"]:
        score = max(0.0, score - 0.2)
        
    # Adjust scores based on entity type and content
    if entity_type == "PASSWORD" and re.search(r'[A-Za-z].*[0-9]|[0-9].*[A-Za-z]', entity_text):
        score = min(1.0, score + 0.15)  # Boost passwords with mixed chars
        
    if entity_type == "API_KEY" and len(entity_text) >= 20:
        score = min(1.0, score + 0.2)  # Boost long API keys
        
    if entity_type == "DEPLOY_TOKEN" and entity_text.startswith(('gh', 'gl')):
        score = min(1.0, score + 0.25)  # Boost GitHub/GitLab tokens
    
    # Medical data has higher confidence if detected by the medical model
    if entity_type == "MEDICAL" and entity.get('detector') == 'medical_pii':
        score = min(1.0, score + 0.2)
        
    # Higher confidence for entities with clear context indicators
    context_indicators = {
        "PASSWORD": ["password", "pwd", "pass"],
        "API_KEY": ["api", "key"],
        "DEPLOY_TOKEN": ["token", "deploy", "access"],
        "CREDENTIAL": ["login", "username", "user"],
        "MEDICAL": ["patient", "medical", "health", "doctor", "hospital"]
    }
    
    if entity_type in context_indicators:
        context = text[max(0, entity['start']-30):entity['start']].lower()
        for indicator in context_indicators[entity_type]:
            if indicator.lower() in context:
                score = min(1.0, score + 0.2)
                break
    
    # Penalty for common words that might be false positives
    common_words = ["submitted", "description", "reference", "customer", "account", "order", "number"]
    if entity_text.lower() in common_words:
        score = max(0.0, score - 0.3)
    
    return score

def merge_overlapping_entities(entities: list, text: str) -> list:
    """
    Enhanced entity merging that resolves overlapping entities intelligently.
    Handles nested entities and prioritizes specialized detections.
    """
    if not entities:
        return []
    
    # First, adjust confidence scores and normalize entity types
    for entity in entities:
        entity['score'] = score_entity(entity, text)
        entity['normalized_type'] = normalize_entity(entity)
    
    # Filter out low confidence entities and None types
    threshold = CONFIG["confidence_threshold"]
    entities = [e for e in entities if e.get('score', 0) >= threshold and e['normalized_type'] is not None]
    if not entities:
        return []
    
    # Sort by position (start index)
    entities.sort(key=lambda x: (x['start'], -x['end']))
    
    # Track overlapping entities and merge/select the best one
    result = []
    current_group = [entities[0]]
    
    for entity in entities[1:]:
        # Check for overlap with the current group
        overlaps = False
        for current in current_group:
            # Two entities overlap if one's start is before other's end
            if entity['start'] < current['end'] and current['start'] < entity['end']:
                overlaps = True
                break
        
        if overlaps:
            current_group.append(entity)
        else:
            # Process the current group to select best entity
            best_entity = select_best_entity(current_group, text)
            if best_entity:  # Only add if not None
                result.append(best_entity)
            current_group = [entity]
    
    # Don't forget the last group
    if current_group:
        best_entity = select_best_entity(current_group, text)
        if best_entity:  # Only add if not None
            result.append(best_entity)
    
    # Final verification pass
    final_entities = []
    for entity in result:
        entity_text = text[entity['start']:entity['end']]
        entity_type = entity['normalized_type']
        
        # Apply final verification and skip any disallowed entities
        if verify_entity(entity_type, entity_text, entity['score']):
            final_entities.append(entity)
    
    return final_entities

def select_best_entity(entities: list, text: str) -> dict:
    """
    Enhanced selection of the best entity from overlapping entities.
    Uses a comprehensive scoring system based on detector reliability,
    entity type priority, and confidence scores.
    """
    if len(entities) == 1:
        entity_text = text[entities[0]['start']:entities[0]['end']]
        entity_type = entities[0]['normalized_type']
        
        # Final verification step
        if not verify_entity(entity_type, entity_text, entities[0]['score']):
            return None
        return entities[0]
    
    # First, try to merge multiple detections of the same entity
    if all(e['normalized_type'] == entities[0]['normalized_type'] for e in entities):
        # Find the entity with widest span
        best = max(entities, key=lambda e: (e['end'] - e['start']))
        # Mark as detected by multiple systems
        best['detected_by'] = len(set(e['detector'] for e in entities))
        
        # Apply final verification
        entity_text = text[best['start']:best['end']]
        if not verify_entity(best['normalized_type'], entity_text, best['score']):
            return None
        return best
    
    # Otherwise, prioritize by score and other factors
    detector_priority = {
        'regex': 3, 
        'presidio': 2.8, 
        'pii_specialized': 2.5,
        'medical_pii': 2.2,
        'medical_reports': 2.1,
        'ner_tech': 1.8, 
        'ner_general': 1.5
    }
    
    # Entity type priority for overlapping entities
    type_priority = {
        "PASSWORD": 1.5,
        "API_KEY": 1.4,
        "CREDENTIAL": 1.3,
        "FINANCIAL": 1.3,
        "SSN": 1.5,
        "CREDIT_CARD": 1.4,
        "MEDICAL": 1.3,
        "IP_ADDRESS": 1.2,
        "MAC_ADDRESS": 1.2,
        "EMAIL_ADDRESS": 1.1,
        "PERSON": 1.0
    }
    
    best_score = -1
    best_entity = None
    
    for entity in entities:
        # Skip entities that fail verification
        entity_text = text[entity['start']:entity['end']]
        if not verify_entity(entity['normalized_type'], entity_text, entity['score']):
            continue
            
        # Calculate a composite score based on multiple factors
        base_score = entity['score']
        detector_boost = detector_priority.get(entity['detector'], 1)
        length_factor = min(1.0, (entity['end'] - entity['start']) / 20)  # Favor longer entities up to a point
        type_boost = type_priority.get(entity['normalized_type'], 1)
        
        # Boost entities that have clear context or are more likely to be sensitive
        composite_score = base_score * detector_boost * type_boost * (1 + 0.2 * length_factor)
        
        if composite_score > best_score:
            best_score = composite_score
            best_entity = entity
    
    return best_entity

# --- Ensemble Detection ---
def run_model_detectors(text: str) -> list:
    """
    Run all available entity detectors in parallel.
    Optimizes detection by using thread pools efficiently.
    """
    all_entities = []
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        
        # Add the core detectors
        futures.append(executor.submit(get_presidio_entities, text))
        futures.append(executor.submit(get_ner_entities, text, "ner_general"))
        futures.append(executor.submit(get_ner_entities, text, "ner_tech"))
        
        # Add the specialized models if available
        if "pii_specialized" in MODELS:
            futures.append(executor.submit(get_ner_entities, text, "pii_specialized"))
            
        if "medical_pii" in MODELS:
            futures.append(executor.submit(get_ner_entities, text, "medical_pii"))
            
        if "medical_reports" in MODELS:
            futures.append(executor.submit(get_ner_entities, text, "medical_reports"))
        
        # Add regex patterns (these are fast, but run them in parallel too)
        futures.append(executor.submit(get_regex_entities, text))
        
        # Collect results
        for future in as_completed(futures):
            try:
                entities = future.result()
                all_entities.extend(entities)
            except Exception as e:
                logger.error(f"Error in detector: {e}")
                
    return all_entities

# --- Anonymization Function ---
def anonymize_text(text: str, pii_options: dict = None, full_redaction: bool = True) -> str:
    """
    Enhanced anonymization function with better entity detection,
    smarter masking, and improved handling of edge cases.
    """
    if not text:
        return ""
        
    # Merge provided options with defaults
    options = DEFAULT_PII_OPTIONS.copy()
    if pii_options:
        options.update(pii_options)
    
    logger.info("Starting entity detection using multiple specialized models...")
    start_time = time.time()
    
    # Run entity detection
    all_entities = run_model_detectors(text)
    
    detection_time = time.time() - start_time
    logger.info(f"Detection completed in {detection_time:.2f} seconds, found {len(all_entities)} potential entities")
    
    # Merge overlapping entities
    merged_entities = merge_overlapping_entities(all_entities, text)
    logger.info(f"Merged into {len(merged_entities)} distinct entities")
    
    # Sort entities in reverse order (to avoid index shifting during replacement)
    merged_entities.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply redaction with consistency checking
    redaction_count = 0
    redacted_spans = {}  # Track already redacted spans
    
    for entity in merged_entities:
        start, end = entity['start'], entity['end']
        
        # Skip if already redacted
        if any(span_start <= start and end <= span_end for (span_start, span_end) in redacted_spans.keys()):
            continue
            
        original_token = text[start:end]
        entity_type = entity['normalized_type']
        
        # Skip redaction of common words and form field labels
        if original_token in BLOCKLIST:
            logger.debug(f"Skipping redaction of blocklisted term '{original_token}'")
            continue
        
        # Check if this entity type should be redacted based on user options
        if options.get(entity_type, True):
            try:
                if full_redaction:
                    masked = full_mask_token(original_token, entity_type)
                else:
                    # Use specialized partial masking based on entity type
                    masked = smart_partial_mask(original_token, entity_type)
            except Exception as e:
                logger.error(f"Error masking token '{original_token}' of type {entity_type}: {e}")
                masked = CONFIG["partial_mask_char"] * len(original_token)
                
            logger.debug(f"Masking '{original_token}' ({entity_type}) to '{masked}'")
            text = text[:start] + masked + text[end:]
            redacted_spans[(start, end)] = True
            redaction_count += 1
    
    total_time = time.time() - start_time
    logger.info(f"Anonymization completed in {total_time:.2f} seconds, redacted {redaction_count} entities")
    
    return text

# --- Flask API ---
app = Flask(__name__)
CORS(app, resources={r"/anonymize": {"origins": os.environ.get("FRONT_END_URL", "http://localhost:5173")}})

@app.route("/anonymize", methods=["POST"])
def anonymize():
    """
    Expect JSON with "text", optional "options", and optional "full_redaction".
    Returns anonymized text with consistent metadata.
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
        
    input_text = data["text"]
    pii_options = data.get("options")
    full_redaction = data.get("full_redaction", True)
    
    try:
        result = anonymize_text(input_text, pii_options, full_redaction)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        return jsonify({
            "anonymized_text": result,
            "timestamp": current_time,
            "user": "rushilpatel21"
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with system status information."""
    return jsonify({
        "status": "ok", 
        "models_loaded": list(MODELS.keys()),
        "version": "2.1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "user": "rushilpatel21",
        "config": {
            "workers": CONFIG["max_workers"],
            "threshold": CONFIG["confidence_threshold"]
        }
    })

@app.route("/entities", methods=["POST"])
def detect_entities():
    """
    Debugging endpoint that shows detected entities without masking.
    Only available in debug mode.
    """
    if not app.debug:
        return jsonify({"error": "This endpoint is only available in debug mode"}), 403
        
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
        
    input_text = data["text"]
    
    # Run detectors
    all_entities = run_model_detectors(input_text)
    
    # Process entities for the response
    entities = []
    for entity in all_entities:
        text_span = input_text[entity['start']:entity['end']]
        entity_type = normalize_entity(entity)
        if entity_type is None:
            continue  # Skip entities with no normalized type
            
        score = score_entity(entity, input_text)
        
        # Only include entities that pass verification
        if verify_entity(entity_type, text_span, score):
            entities.append({
                "text": text_span,
                "type": entity_type,
                "score": score,
                "detector": entity.get('detector', 'unknown'),
                "span": [entity['start'], entity['end']]
            })
            
    # Sort by confidence score
    entities.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify({
        "entities": entities,
        "total_count": len(entities),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "user": "rushilpatel21"
    })

@app.route("/config", methods=["GET"])
def get_config():
    """Return the current configuration for debugging purposes."""
    if not app.debug:
        return jsonify({"error": "This endpoint is only available in debug mode"}), 403
        
    return jsonify({
        "config": CONFIG,
        "models_loaded": list(MODELS.keys()),
        "entity_types": sorted(list(set(ENTITY_TYPE_MAPPING.values()))),
        "defaults": {
            "threshold": CONFIG["confidence_threshold"],
            "workers": CONFIG["max_workers"],
            "partial_mask_char": CONFIG["partial_mask_char"]
        },
        "timestamp": "2025-03-02 05:23:36",
        "user": "rushilpatel21"
    })

@app.route("/test", methods=["GET"])
def test_connection():
    """Simple test endpoint to verify connectivity."""
    return jsonify({
        "status": "ok",
        "message": "Redactify API is running and operational",
        "timestamp": "2025-03-02 05:23:36"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    host = "0.0.0.0" if not debug else "127.0.0.1"
    
    print(f"Redactify API v2.1.0 is ready and serving on port {port}")
    print(f"Models loaded: {list(MODELS.keys())}")
    print(f"Debug mode: {debug}")
    print(f"Worker threads: {CONFIG['max_workers']}")
    print(f"Current time: 2025-03-02 05:23:36")
    print(f"Current user: rushilpatel21")
    
    app.run(debug=debug, port=port, host=host)