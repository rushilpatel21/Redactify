import sys
import os
import hashlib
import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
CONFIG = {
    "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.65)),  # Increased threshold
    "context_window": 5,  # Words to check for context around potential PII
    "max_workers": 4,  # Thread pool size
}

# Common terms that are often false positives
BLOCKLIST = {
    "Submitted", "Customer", "Issue Description", "Order Number", 
    "Account", "confirmation", "attempts", "reference", "Description",
    "screenshots", "communication", "Number", "Information", "Details"
}

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
}

# Types to pseudonymize in full redaction mode
PSEUDONYMIZE_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", 
    "API_KEY", "DEPLOY_TOKEN", "AUTHENTICATION"
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
        # General NER model
        models["ner_general"] = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english", 
            aggregation_strategy="simple"
        )
        
        # Fallback in case the main model fails
        try:
            # Try to load a second model for technical content
            models["ner_tech"] = pipeline(
                "ner",
                model="Jean-Baptiste/roberta-large-ner-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.warning(f"Could not load technical NER model: {e}")
            # Fallback to using the general model
            models["ner_tech"] = models["ner_general"]
        
        # Presidio analyzer for specialized PII detection
        models["presidio"] = AnalyzerEngine()
        
        logger.info("All models loaded successfully")
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
    
    # URL patterns
    {"type": "URL", "pattern": r"\bhttps?://[^\s]+\b", "context": []},
    {"type": "URL", "pattern": r"\b(?:www\.)[a-z0-9-]+(?:\.[a-z]{2,})+(?:/[^\s]*)?", "context": []},
    
    # Date patterns - expanded and more precise
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "context": ["date", "on", "as of", "effective"]},
    {"type": "DATE_TIME", "pattern": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", "context": []},
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}/\d{2}\b", "context": ["exp", "expiration", "valid", "until"]},  # MM/YY format
    {"type": "DATE_TIME", "pattern": r"\b\d{4}-\d{2}-\d{2}\b", "context": []},  # YYYY-MM-DD format
    
    # Phone number patterns
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{10}\b", "context": ["phone", "mobile", "cell", "tel", "telephone", "contact"]},
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "context": []},
    {"type": "PHONE_NUMBER", "pattern": r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b", "context": []},
    
    # Password patterns - improved precision
    {"type": "PASSWORD", "pattern": r"(?i)(?:password|passwd|pwd)(?::|=|\s+is\s+)\s*(\S+)", "context": []},
    {"type": "PASSWORD", "pattern": r"(?i)password(?:\s+was|\s+has\s+been)?\s+(?:reset|changed)(?:\s+to)?\s+(\S+)", "context": []},
    # Only detect complex passwords with clear context
    {"type": "PASSWORD", "pattern": r"(?=.*[A-Za-z])(?=.*\d)(?=.*[$#@!%^&*()_+])[A-Za-z\d$#@!%^&*()_+]{8,}", 
     "context": ["password", "pass", "pwd", "credential", "login", "auth", "secret"]},
    
    # Credit card patterns
    {"type": "CREDIT_CARD", "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b", "context": ["credit", "card", "visa", "mastercard", "amex"]},
    {"type": "CREDIT_CARD", "pattern": r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", "context": []},
    
    # API key patterns - enhanced
    {"type": "API_KEY", "pattern": r"(?i)api[_-]?key(?::|=|\s+is\s+)\s*([A-Za-z0-9\-_\.]{8,})\b", "context": []},
    {"type": "API_KEY", "pattern": r"(?i)(?:api|app|access)[_-]?(?:key|token|secret|id)(?::|=|\s+is\s+)\s*\S+", "context": []},
    # Only match long alphanumeric strings if there's context
    {"type": "API_KEY", "pattern": r"\b[A-Za-z0-9_\-]{20,40}\b", 
     "context": ["api", "key", "secret", "token", "auth", "access", "credentials"]},
    
    # Deploy token patterns
    {"type": "DEPLOY_TOKEN", "pattern": r"gh[pousr]_[A-Za-z0-9_]{16,}\b", "context": []},  # GitHub tokens
    {"type": "DEPLOY_TOKEN", "pattern": r"(?i)(?:deploy|access|auth|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Authentication patterns
    {"type": "AUTHENTICATION", "pattern": r"(?i)(?:bearer|basic|digest|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)auth(?:entication)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)credential(?:s)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Financial information - enhanced with card details
    {"type": "FINANCIAL", "pattern": r"\brouting[:\s]+(\d{9})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\baccount\s+(?:number|#)?[:\s]+(\d+)\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\b(?:account|acct)(?:.+?)ending in (\d{4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"ending in \d{4}", "context": ["card", "account"]},
    {"type": "FINANCIAL", "pattern": r"card \(ending in \d{4}", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bCVV:?\s*(\d{3,4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bCVC:?\s*(\d{3,4})\b", "context": []},
    
    # Student roll number patterns
    {"type": "ROLL_NUMBER", "pattern": r"\b\d{2}[A-Za-z]{3}\d{3}\b", "context": ["student", "roll", "enrollment"]},
    {"type": "ROLL_NUMBER", "pattern": r"\b(?:roll|enrollment|student)(?:.+?)(?:number|no|#)?[:\s]+([A-Za-z0-9\-]{5,10})\b", "context": []},
    
    # Username patterns - improved precision
    {"type": "CREDENTIAL", "pattern": r"\busername[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\blogin[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\buser(?:name)?[:\s]+(\S+)\b", "context": []},
    
    # Device information
    {"type": "DEVICE", "pattern": r"(?:iPhone|iPad|MacBook|Android|Windows|Device)(?:\s+\w+){0,2}", 
     "context": ["IP", "device", "login", "from"]},
    
    # Order/Account identifiers - more specific to avoid false positives
    {"type": "CREDENTIAL", "pattern": r"(?:Order|Account|Invoice)(?:\s+(?:Number|#|ID|No\.?)):\s*([A-Za-z0-9\-]+)", 
     "context": ["order", "account", "#", "number"]}
]

# --- Entity Normalization and Mapping ---
ENTITY_TYPE_MAPPING = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "PEOPLE": "PERSON",
    "PERSONAL": "PERSON",
    
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "CORPORATION": "ORGANIZATION",
    
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "ADDRESS": "LOCATION",
    "PLACE": "LOCATION",
    
    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "TEL": "PHONE_NUMBER",
    "TELEPHONE": "PHONE_NUMBER",
    
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT": "CREDIT_CARD",
    "CC": "CREDIT_CARD",
    "PAYMENT_CARD": "CREDIT_CARD",
    
    "SSN": "SSN",
    "SOCIAL_SECURITY": "SSN",
    
    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    
    "URL": "URL",
    "URI": "URL",
    "WEBSITE": "URL",
    "LINK": "URL",
    
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "DATE_TIME": "DATE_TIME",
    "DATETIME": "DATE_TIME",
    
    "PASSWORD": "PASSWORD",
    "PWD": "PASSWORD",
    "PASSWD": "PASSWORD",
    
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "KEY": "API_KEY",
    
    "TOKEN": "DEPLOY_TOKEN",
    "DEPLOY_TOKEN": "DEPLOY_TOKEN",
    "ACCESS_TOKEN": "DEPLOY_TOKEN",
    "SECRET_TOKEN": "DEPLOY_TOKEN",
    
    "AUTH": "AUTHENTICATION",
    "AUTHENTICATION": "AUTHENTICATION",
    
    "CREDENTIAL": "CREDENTIAL",
    "LOGIN": "CREDENTIAL",
    
    "FINANCIAL": "FINANCIAL",
    "ACCOUNT": "FINANCIAL",
    "ROUTING": "FINANCIAL",
    "BANK": "FINANCIAL",
    
    "ROLL_NUMBER": "ROLL_NUMBER",
    "ENROLLMENT": "ROLL_NUMBER",
    "STUDENT_ID": "ROLL_NUMBER",
    
    "DEVICE": "DEVICE",
    "PRODUCT": "PRODUCT",
    
    # Skip miscellaneous entities - they're often false positives
    "MISC": None,
}

# --- Masking Functions ---

def pseudonymize_value(value: str, entity_type: str) -> str:
    """Generate a hash-based pseudonym (first 6 hex digits) for a given value."""
    h = hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    return f"[{entity_type}-{h}]"

def full_mask_token(token: str, entity_type: str) -> str:
    """For full redaction: return the hash‐based pseudonym."""
    if entity_type is None:
        return '*' * len(token)
    return pseudonymize_value(token, entity_type.upper())

def partial_mask_token(token: str) -> str:
    """
    Partially mask a token by preserving first 2 and last 3 characters.
    For shorter tokens, use appropriate masking strategy.
    """
    n = len(token)
    if n > 8:
        return token[:2] + '*' * (n - 5) + token[-3:]
    elif n > 5:
        return token[:1] + '*' * (n - 3) + token[-2:]
    else:
        return '*' * n

def mask_email(email: str) -> str:
    """
    Mask an email address by splitting it into local and domain parts.
    Local part: if >4 chars, preserve first 2 and last 2, mask the middle.
    Domain: mask everything before the last dot.
    """
    try:
        local, domain = email.split("@")
    except Exception as e:
        logger.error(f"Error splitting email '{email}': {e}")
        return '*' * len(email)
    
    if len(local) > 4:
        local_masked = local[:2] + '*' * (len(local) - 4) + local[-2:]
    else:
        local_masked = '*' * len(local)
    if '.' in domain:
        last_dot = domain.rfind('.')
        domain_name = domain[:last_dot]
        tld = domain[last_dot:]
        domain_masked = '*' * len(domain_name) + tld
    else:
        domain_masked = '*' * len(domain)
    return local_masked + "@" + domain_masked

def mask_url(url: str) -> str:
    """
    For full redaction: return a hash-based pseudonym for the URL.
    """
    return full_mask_token(url, "URL")

def partial_mask_url(url: str) -> str:
    """Partially mask a URL by processing its domain and path."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.error(f"Error parsing URL '{url}': {e}")
        return '*' * len(url)
    scheme, netloc, path, params, query, fragment = (
        parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
    )
    # Process netloc (handle port numbers)
    if ':' in netloc:
        domain, port = netloc.split(':', 1)
        port = ':' + port
    else:
        domain, port = netloc, ''
    parts = domain.split('.')
    masked_parts = []
    for part in parts:
        if len(part) >= 6:
            masked_parts.append(part[:2] + "*" * (len(part)-5) + part[-3:])
        else:
            masked_parts.append('*' * len(part))
    masked_netloc = '.'.join(masked_parts) + port
    # Process path segments
    path_segments = path.split('/')
    masked_segments = [partial_mask_token(seg) if seg and len(seg) >= 3 else seg for seg in path_segments]
    masked_path = '/'.join(masked_segments)
    return urlunparse((scheme, masked_netloc, masked_path, params, query, fragment))

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
            
    # Also check if the match is near a colon, equals sign, or other indicators
    nearby_text = text[max(0, span_start-20):min(len(text), span_end+10)]
    indicators = r'(?::|=|is\s+|was\s+reset\s+to\s+)'
    if re.search(indicators + r'\s*' + re.escape(matched_text), nearby_text, re.IGNORECASE):
        return True
        
    return False

# --- Entity Verification ---

def verify_entity(entity_type, text, confidence_score):
    """Additional verification to filter out false positives."""
    # Check if text is in blocklist
    if text.strip() in BLOCKLIST:
        return False
    
    # Skip None entity types
    if entity_type is None:
        return False
    
    # Reject specific structural patterns that are likely false positives
    if (
        text.isdigit() and len(text) < 4 or  # Short numeric strings
        text.startswith("#") or              # Section headers/references
        (len(text.split()) == 1 and len(text) < 5 and not re.search(r'\d', text))  # Short single words
    ):
        return False
    
    # Reject low-confidence entities of certain types
    type_confidence_thresholds = {
        "PASSWORD": 0.8,      # Higher threshold for passwords
        "API_KEY": 0.75,      # Higher for API keys
        "CREDENTIAL": 0.8,    # Higher for credentials
        "FINANCIAL": 0.7,     # Higher for financial data
        "ORGANIZATION": 0.7,  # Higher for organizations
        "DEVICE": 0.85,       # Very high for devices
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
    if entity_type == "CREDENTIAL" and text in ["user", "account", "login"]:
        return False
        
    if entity_type == "DEVICE" and text in ["iPhone", "MacBook", "Android", "Windows"]:
        # Only redact devices when paired with identifiers
        return bool(re.search(r'[a-zA-Z]+\s+[a-zA-Z0-9]+', text))
            
    return True

# --- Detection Functions ---

def normalize_entity(entity: dict) -> str:
    """Normalize entity types across different detectors."""
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
    """Detect entities using enhanced regex patterns with context awareness."""
    regex_entities = []
    
    for pattern_def in REGEX_PATTERNS:
        for match in re.finditer(pattern_def["pattern"], text):
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
    Adjust entity confidence score based on additional heuristics.
    This helps prioritize certain detections over others.
    """
    score = entity.get('score', 0.6)
    entity_text = text[entity['start']:entity['end']]
    entity_type = normalize_entity(entity)
    
    # Skip banned terms
    if entity_text in BLOCKLIST or entity_type is None:
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
    
    # Higher confidence for entities with clear context indicators
    context_indicators = {
        "PASSWORD": ["password", "pwd", "pass"],
        "API_KEY": ["api", "key"],
        "DEPLOY_TOKEN": ["token", "deploy", "access"],
        "CREDENTIAL": ["login", "username", "user"],
    }
    
    if entity_type in context_indicators:
        context = text[max(0, entity['start']-30):entity['start']]
        for indicator in context_indicators[entity_type]:
            if indicator.lower() in context.lower():
                score = min(1.0, score + 0.2)
                break
    
    # Penalty for common words that might be false positives
    common_words = ["submitted", "description", "reference", "customer", "account", "order", "number"]
    if entity_text.lower() in common_words:
        score = max(0.0, score - 0.3)
    
    return score

def merge_overlapping_entities(entities: list, text: str) -> list:
    """
    Merge overlapping entities, considering their confidence scores and source.
    This is more sophisticated than simple deduplication.
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
    
    return result

def select_best_entity(entities: list, text: str) -> dict:
    """
    From a group of overlapping entities, select the best one based on:
    1. Confidence score (adjusted)
    2. Detector priority
    3. Entity length (prefer longer entities)
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
    detector_priority = {'regex': 3, 'presidio': 2, 'ner_tech': 1.5, 'ner_general': 1}
    
    best_score = -1
    best_entity = None
    
    for entity in entities:
        # Skip entities that fail verification
        entity_text = text[entity['start']:entity['end']]
        if not verify_entity(entity['normalized_type'], entity_text, entity['score']):
            continue
            
        # Calculate a composite score based on confidence and other factors
        base_score = entity['score']
        detector_boost = detector_priority.get(entity['detector'], 1)
        length_factor = min(1.0, (entity['end'] - entity['start']) / 20)  # Favor longer entities up to a point
        
        # Prefer certain entity types for security (e.g., PASSWORD over general text)
        type_priority = {
            "PASSWORD": 1.5,
            "API_KEY": 1.4,
            "CREDENTIAL": 1.3,
            "FINANCIAL": 1.3,
            "SSN": 1.5,
            "CREDIT_CARD": 1.4
        }
        type_boost = type_priority.get(entity['normalized_type'], 1)
        
        composite_score = base_score * detector_boost * type_boost * (1 + 0.2 * length_factor)
        
        if composite_score > best_score:
            best_score = composite_score
            best_entity = entity
    
    return best_entity

# --- Anonymization Function ---
def anonymize_text(text: str, pii_options: dict = None, full_redaction: bool = True) -> str:
    """
    Enhanced anonymization function that combines multiple detection methods,
    uses context-aware entity merging, and applies appropriate masking.
    """
    if not text:
        return ""
        
    # Merge provided options with defaults
    options = DEFAULT_PII_OPTIONS.copy()
    if pii_options:
        options.update(pii_options)
    
    logger.info("Starting entity detection using multiple detectors...")
    
    # Run detectors in parallel
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_detector = {
            executor.submit(get_presidio_entities, text): "presidio",
            executor.submit(get_ner_entities, text, "ner_general"): "ner_general", 
            executor.submit(get_ner_entities, text, "ner_tech"): "ner_tech",
            executor.submit(get_regex_entities, text): "regex"
        }
        
        all_entities = []
        for future in as_completed(future_to_detector):
            detector = future_to_detector[future]
            try:
                entities = future.result()
                logger.info(f"{detector} found {len(entities)} entities")
                all_entities.extend(entities)
            except Exception as e:
                logger.error(f"Error in {detector}: {e}")
    
    logger.info(f"Total entities detected before merging: {len(all_entities)}")
    
    # Merge overlapping entities
    merged_entities = merge_overlapping_entities(all_entities, text)
    logger.info(f"Entities after merging and validation: {len(merged_entities)}")
    
    # Sort entities in reverse order (to avoid index shifting during replacement)
    merged_entities.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply redaction
    for entity in merged_entities:
        start, end = entity['start'], entity['end']
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
                    if entity_type == "EMAIL_ADDRESS":
                        masked = mask_email(original_token)
                    elif entity_type == "URL":
                        masked = partial_mask_url(original_token)
                    else:
                        masked = partial_mask_token(original_token)
            except Exception as e:
                logger.error(f"Error masking token '{original_token}' of type {entity_type}: {e}")
                masked = '*' * len(original_token)
                
            logger.debug(f"Masking token '{original_token}' ({entity_type}) to '{masked}'")
            text = text[:start] + masked + text[end:]
    
    return text

# --- Flask API ---
app = Flask(__name__)
CORS(app, resources={r"/anonymize": {"origins": os.environ.get("FRONT_END_URL", "http://localhost:5173")}})

@app.route("/anonymize", methods=["POST"])
def anonymize():
    """
    Expects JSON with "text", optional "options", and optional boolean "full_redaction".
    Returns the anonymized text.
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
        
    input_text = data["text"]
    pii_options = data.get("options")
    full_redaction = data.get("full_redaction", True)
    
    try:
        result = anonymize_text(input_text, pii_options, full_redaction)
        return jsonify({"anonymized_text": result})
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok", 
        "models_loaded": list(MODELS.keys()),
        "version": "1.1.0",
        "timestamp": re.sub(r'[^0-9-: ]', '', "2025-03-01 15:51:17")
    })

@app.route("/entities", methods=["POST"])
def detect_entities():
    """
    Debugging endpoint that shows what entities were detected without masking.
    Only available in debug mode.
    """
    if not app.debug:
        return jsonify({"error": "This endpoint is only available in debug mode"}), 403
        
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
        
    input_text = data["text"]
    
    # Run detectors
    entities = []
    
    try:
        presidio_entities = get_presidio_entities(input_text)
        ner_general_entities = get_ner_entities(input_text, "ner_general")
        ner_tech_entities = get_ner_entities(input_text, "ner_tech")
        regex_entities = get_regex_entities(input_text)
        
        all_entities = presidio_entities + ner_general_entities + ner_tech_entities + regex_entities
        
        # Process entities for the response
        entities = []
        for entity in all_entities:
            text_span = input_text[entity['start']:entity['end']]
            entity_type = normalize_entity(entity)
            score = score_entity(entity, input_text)
            
            entities.append({
                "text": text_span,
                "type": entity_type,
                "score": score,
                "detector": entity.get('detector', 'unknown')
            })
            
        # Sort by confidence score
        entities.sort(key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        return jsonify({"error": f"Error detecting entities: {str(e)}"}), 500
    
    return jsonify({"entities": entities})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    host = "0.0.0.0" if not debug else "127.0.0.1"
    
    print(f"Redactify API v1.1.0 is ready and serving on port {port}")
    app.run(debug=debug, port=port, host=host)