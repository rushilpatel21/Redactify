import sys
import os
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from flask_cors import CORS

# --- Configuration ---
# (These can be further externalized via environment variables or a separate config file)
CONFIG = {
    "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.6)),
    "placeholders": {
        "PHONE_NUMBER": "[PHONE]",
        "CREDIT_CARD": "[CREDIT_CARD]",
        "SSN": "[SSN]",
        "IP_ADDRESS": "[IP]",
        "URL": "[URL]",
        "DATE_TIME": "[DATE]",
        "PASSWORD": "[PASSWORD]",
    }
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
    "ROLL_NUMBER": True
}

# Types to pseudonymize in full redaction mode.
PSEUDONYMIZE_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS"}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cache the Models ---
HF_PIPELINE = pipeline(
    "ner", 
    model="dbmdz/bert-large-cased-finetuned-conll03-english", 
    aggregation_strategy="simple"
)
ANALYZER_ENGINE = AnalyzerEngine()

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
    Partially mask a token by preserving the first 2 and last 3 characters.
    If the token is too short, return a full mask.
    E.g. "9824667788" becomes "98*****788".
    """
    n = len(token)
    if n > 5:
        return token[:2] + '*' * (n - 5) + token[-3:]
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
    """
    Partially mask a URL by processing its domain and path.
    Domain: For each label, if len>=6 preserve first 2 and mask the 3rd character.
    Path: For each segment (if len>=3), apply partial_mask_token.
    """
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

# --- Entity Normalization ---

def normalize_entity(entity: dict) -> str:
    """
    Normalize the entity type from different detectors.
    Maps various labels to a consistent set.
    """
    raw = None
    if 'entity_group' in entity:
        raw = entity['entity_group']
    elif 'entity' in entity:
        raw = entity['entity']
        if raw.startswith("B-") or raw.startswith("I-"):
            raw = raw[2:]
    if raw:
        mapping = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "ORG": "ORGANIZATION",
            "EMAIL": "EMAIL_ADDRESS",
            "EMAIL_ADDRESS": "EMAIL_ADDRESS",
            "PHONE_NUMBER": "PHONE_NUMBER",
            "PHONE": "PHONE_NUMBER",
            "CREDIT_CARD": "CREDIT_CARD",
            "SSN": "SSN",
            "IP": "IP_ADDRESS",
            "IP_ADDRESS": "IP_ADDRESS",
            "URL": "URL",
            "DATE": "DATE_TIME",
            "TIME": "DATE_TIME",
            "PASSWORD": "PASSWORD",
            "API_KEY": "API_KEY",
            "ROLL_NUMBER": "ROLL_NUMBER"
        }
        return mapping.get(raw.upper(), raw.upper())
    return None

# --- Detection Functions ---
# Running detection methods concurrently to improve performance.
def get_presidio_entities(text: str) -> list:
    try:
        results = ANALYZER_ENGINE.analyze(text=text, language="en")
    except Exception as e:
        logger.error(f"Error in Presidio Analyzer: {e}")
        results = []
    presidio_entities = [{
        'entity_group': res.entity_type,
        'start': res.start,
        'end': res.end,
        'score': res.score
    } for res in results]
    return presidio_entities

def get_hf_entities(text: str) -> list:
    try:
        return HF_PIPELINE(text)
    except Exception as e:
        logger.error(f"Error in Hugging Face pipeline: {e}")
        return []

def get_regex_entities(text: str) -> list:
    """
    Additional regex-based detection.
    Added patterns for roll numbers (e.g., 22bce308) and API keys.
    """
    regex_patterns = [
        {"type": "SSN", "pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
        {"type": "IP_ADDRESS", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"},
        {"type": "URL", "pattern": r"\bhttps?://[^\s]+\b"},
        {"type": "DATE_TIME", "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"},
        {"type": "PHONE_NUMBER", "pattern": r"\b\d{10}\b"},
        {"type": "PHONE_NUMBER", "pattern": r"\b\d{3}-\d{3}-\d{4}\b"},
        {"type": "PHONE_NUMBER", "pattern": r"\b\d{5}-\d{5}\b"},
        {"type": "PASSWORD", "pattern": r"(?i)password:\s*\S+"},
        {"type": "CREDIT_CARD", "pattern": r"\b(?:\d{4}-){3}\d{4}\b"},
        {"type": "PASSWORD", "pattern": r"(?i)\bpassword\b(?:\s*(?:was|is|:))\s*\S+"},
        {"type": "API_KEY", "pattern": r"(?i)\bapi[_-]?key\s*[:=]\s*[A-Za-z0-9\-_]{8,}\b"},
        {"type": "ROLL_NUMBER", "pattern": r"\b\d{2}[A-Za-z]{3}\d{3}\b"}
    ]
    regex_entities = []
    for item in regex_patterns:
        for match in re.finditer(item["pattern"], text):
            start, end = match.span()
            regex_entities.append({
                "entity_group": item["type"],
                "start": start,
                "end": end,
                "score": 0.9
            })
    return regex_entities

def deduplicate_entities(entities: list) -> list:
    """
    Deduplicate overlapping entities based on their start and end positions,
    choosing the one with the highest confidence.
    """
    entities = [e for e in entities if e.get('score', 0) >= CONFIG["confidence_threshold"]]
    if not entities:
        return []
    entities_sorted = sorted(entities, key=lambda x: (x['start'], -x.get('score', 0)))
    deduped = []
    current = entities_sorted[0]
    for ent in entities_sorted[1:]:
        if ent['start'] < current['end']:
            if ent.get('score', 0) > current.get('score', 0):
                current = ent
        else:
            deduped.append(current)
            current = ent
    deduped.append(current)
    return deduped

# --- Anonymization Function ---
def anonymize_text(text: str, pii_options: dict = None, full_redaction: bool = True) -> str:
    """
    Detects PII using three detectors (Presidio, HF NER, Regex) in parallel,
    deduplicates them, and then masks each token according to options.
    
    - If full_redaction is True: all enabled entities are replaced with a hash-based pseudonym.
    - If False: EMAIL_ADDRESS and URL use custom partial masking; others are partially masked.
    """
    # Merge provided options with defaults.
    options = DEFAULT_PII_OPTIONS.copy()
    if pii_options:
        options.update(pii_options)
    
    logger.info("Starting entity detection using multiple detectors...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        presidio_future = executor.submit(get_presidio_entities, text)
        hf_future = executor.submit(get_hf_entities, text)
        regex_future = executor.submit(get_regex_entities, text)
        presidio_entities = presidio_future.result()
        hf_entities = hf_future.result()
        regex_entities = regex_future.result()
    
    all_entities = presidio_entities + hf_entities + regex_entities
    logger.info(f"Total entities detected before deduplication: {len(all_entities)}")
    all_entities = deduplicate_entities(all_entities)
    logger.info(f"Entities after deduplication and thresholding: {len(all_entities)}")
    
    # Process entities in reverse order to avoid index shifting.
    all_entities.sort(key=lambda x: x['start'], reverse=True)
    
    for entity in all_entities:
        start, end = entity['start'], entity['end']
        original_token = text[start:end]
        norm_type = normalize_entity(entity)
        if options.get(norm_type, True):
            try:
                if full_redaction:
                    masked = full_mask_token(original_token, norm_type)
                else:
                    if norm_type == "EMAIL_ADDRESS":
                        masked = mask_email(original_token)
                    elif norm_type == "URL":
                        masked = partial_mask_url(original_token)
                    else:
                        masked = partial_mask_token(original_token)
            except Exception as e:
                logger.error(f"Error masking token '{original_token}' of type {norm_type}: {e}")
                masked = '*' * len(original_token)
            logger.debug(f"Masking token '{original_token}' ({norm_type}) to '{masked}'")
            text = text[:start] + masked + text[end:]
    return text

# --- Flask API ---
app = Flask(__name__)
CORS(app, resources={r"/anonymize": {"origins": "http://localhost:5173"}})

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
    result = anonymize_text(input_text, pii_options, full_redaction)
    return jsonify({"anonymized_text": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Server is ready and serving on port {port}")
    app.run(debug=True, port=port)
