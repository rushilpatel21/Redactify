import sys
import os
import hashlib
import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
CONFIG = {
    "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5)),
    "context_window": 5,  # Words to check for context around potential PII
    "max_workers": 4,  # Thread pool size
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
    "ROLL_NUMBER": True
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
        
        # Technical/code specialized model
        models["ner_tech"] = pipeline(
            "ner",
            model="explosion/en_core_web_trf",
            aggregation_strategy="simple"
        )
        
        # Presidio analyzer with enhanced recognizers
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
    
    # Date patterns - expanded
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "context": ["date", "on", "as of", "effective"]},
    {"type": "DATE_TIME", "pattern": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", "context": []},
    {"type": "DATE_TIME", "pattern": r"\b\d{1,2}/\d{2}\b", "context": ["exp", "expiration", "valid", "until"]},  # MM/YY format
    {"type": "DATE_TIME", "pattern": r"\b\d{4}-\d{2}-\d{2}\b", "context": []},  # YYYY-MM-DD format
    
    # Phone number patterns
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{10}\b", "context": ["phone", "mobile", "cell", "tel", "telephone", "contact"]},
    {"type": "PHONE_NUMBER", "pattern": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "context": []},
    {"type": "PHONE_NUMBER", "pattern": r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b", "context": []},
    
    # Password patterns - enhanced
    {"type": "PASSWORD", "pattern": r"(?i)(?:password|passwd|pwd)(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "PASSWORD", "pattern": r"(?:[0-9a-zA-Z$#@!%^&*()_+]{8,})", "context": ["password", "pass", "pwd", "credential", "login", "auth"]},
    
    # Credit card patterns
    {"type": "CREDIT_CARD", "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b", "context": ["credit", "card", "visa", "mastercard", "amex"]},
    {"type": "CREDIT_CARD", "pattern": r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", "context": []},
    
    # API key patterns - enhanced
    {"type": "API_KEY", "pattern": r"(?i)api[_-]?key(?::|=|\s+is\s+)\s*([A-Za-z0-9\-_\.]{8,})\b", "context": []},
    {"type": "API_KEY", "pattern": r"\b[A-Za-z0-9_\-]{20,40}\b", "context": ["api", "key", "secret", "token", "auth"]},
    {"type": "API_KEY", "pattern": r"(?i)(?:api|app|access)[_-]?(?:key|token|secret|id)(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Deploy token patterns
    {"type": "DEPLOY_TOKEN", "pattern": r"gh[pousr]_[A-Za-z0-9_]{16,}\b", "context": []},  # GitHub tokens
    {"type": "DEPLOY_TOKEN", "pattern": r"(?i)(?:deploy|access|auth|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Authentication patterns
    {"type": "AUTHENTICATION", "pattern": r"(?i)(?:bearer|basic|digest|oauth)[_-]?token(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)auth(?:entication)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    {"type": "AUTHENTICATION", "pattern": r"(?i)credential(?:s)?(?::|=|\s+is\s+)\s*\S+", "context": []},
    
    # Financial information
    {"type": "FINANCIAL", "pattern": r"\brouting[:\s]+(\d{9})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\baccount\s+(?:number|#)?[:\s]+(\d+)\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\b(?:account|acct)(?:.+?)ending in (\d{4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bCVV:?\s*(\d{3,4})\b", "context": []},
    {"type": "FINANCIAL", "pattern": r"\bCVC:?\s*(\d{3,4})\b", "context": []},
    
    # Student roll number patterns
    {"type": "ROLL_NUMBER", "pattern": r"\b\d{2}[A-Za-z]{3}\d{3}\b", "context": ["student", "roll", "enrollment"]},
    {"type": "ROLL_NUMBER", "pattern": r"\b(?:roll|enrollment|student)(?:.+?)(?:number|no|#)?[:\s]+([A-Za-z0-9\-]{5,10})\b", "context": []},
    
    # Username patterns
    {"type": "CREDENTIAL", "pattern": r"\busername[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\blogin[:\s]+(\S+)\b", "context": []},
    {"type": "CREDENTIAL", "pattern": r"\buser[:\s]+(\S+)\b", "context": []}
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
    """Check if any context words appear near the detected entity."""
    if not context_words:
        return True  # No context check needed
    
    # Extract a window of text before and after the entity
    context_size = CONFIG["context_window"]
    text_before = text[:span_start].split()[-context_size:] if span_start > 0 else []
    text_after = text[span_end:].split()[:context_size] if span_end < len(text) else []
    
    # Check for context words
    context_text = ' '.join(text_before + text_after).lower()
    for word in context_words:
        if word.lower() in context_text:
            return True
            
    # Also check if the match is near a colon, equals sign, or other indicators
    nearby_text = text[max(0, span_start-10):min(len(text), span_end+10)]
    if re.search(r'[:=]', nearby_text):
        return True
        
    return False

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
                
            # Check for contextual clues
            if has_context(text, start, end, pattern_def.get("context", [])):
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
    
    # Boost confidence for entities detected by multiple systems
    if entity.get('detected_by', 1) > 1:
        score = min(1.0, score + 0.1 * entity['detected_by'])
        
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
    
    # Filter out low confidence entities
    threshold = CONFIG["confidence_threshold"]
    entities = [e for e in entities if e.get('score', 0) >= threshold]
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
            result.append(best_entity)
            current_group = [entity]
    
    # Don't forget the last group
    if current_group:
        best_entity = select_best_entity(current_group, text)
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
        return entities[0]
    
    # First, try to merge multiple detections of the same entity
    if all(e['normalized_type'] == entities[0]['normalized_type'] for e in entities):
        # Find the entity with widest span
        best = max(entities, key=lambda e: (e['end'] - e['start']))
        # Mark as detected by multiple systems
        best['detected_by'] = len(set(e['detector'] for e in entities))
        return best
    
    # Otherwise, prioritize by score and other factors
    detector_priority = {'regex': 3, 'presidio': 2, 'ner_tech': 1.5, 'ner_general': 1}
    
    best_score = -1
    best_entity = None
    
    for entity in entities:
        # Calculate a composite score based on confidence and other factors
        base_score = entity['score']
        detector_boost = detector_priority.get(entity['detector'], 1)
        length_factor = min(1.0, (entity['end'] - entity['start']) / 20)  # Favor longer entities up to a point
        
        composite_score = base_score * detector_boost * (1 + 0.2 * length_factor)
        
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
    logger.info(f"Entities after merging: {len(merged_entities)}")
    
    # Sort entities in reverse order (to avoid index shifting during replacement)
    merged_entities.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply redaction
    for entity in merged_entities:
        start, end = entity['start'], entity['end']
        original_token = text[start:end]
        entity_type = entity['normalized_type']
        
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
    return jsonify({"status": "ok", "models_loaded": list(MODELS.keys())})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Redactify API is ready and serving on port {port}")
    app.run(debug=True, port=port, host="0.0.0.0")