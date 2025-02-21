import sys
import hashlib
import logging
import re
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration for fixed replacements (not used in full redaction mode now)
CONFIG = {
    "confidence_threshold": 0.6,  # Minimum score to consider an entity valid
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

# Entity types that will be pseudonymized using hash in full redaction mode
PSEUDONYMIZE_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS"}

# Cache the HF pipeline and AnalyzerEngine so they're not reloaded repeatedly.
HF_PIPELINE = pipeline(
    "ner", 
    model="dbmdz/bert-large-cased-finetuned-conll03-english", 
    aggregation_strategy="simple"
)
ANALYZER_ENGINE = AnalyzerEngine()

def pseudonymize_value(value: str, entity_type: str) -> str:
    """
    Generate a pseudonym for a given value using an MD5 hash.
    Returns a string like [TYPE-320b8e] ensuring consistency.
    """
    h = hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    return f"[{entity_type}-{h}]"

def full_mask_token(token: str, entity_type: str) -> str:
    """
    For full redaction: always return the hash‐based pseudonym.
    """
    if entity_type is None:
        return '*' * len(token)
    return pseudonymize_value(token, entity_type.upper())

def partial_mask_token(token: str) -> str:
    """
    Partially mask a token by preserving the first 2 and last 3 characters.
    If the token is too short, mask it fully.
    Example: "9824667788" becomes "98*****788"
    """
    n = len(token)
    if n > 5:
        return token[:2] + '*' * (n - 5) + token[-3:]
    else:
        return '*' * n

def mask_email(email: str) -> str:
    """
    Mask an email address by splitting into local and domain parts.
    For the local part, if longer than 4 characters, preserve the first 2 and last 2 characters,
    masking the middle; otherwise fully mask.
    For the domain, mask everything before the final dot.
    
    Examples:
      "rushilpatel210@gmail.com" -> "ru**********10@*****.com"
    """
    try:
        local, domain = email.split("@")
    except ValueError:
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
    For full redaction: return a hash‐based pseudonym for the entire URL.
    (Uses full_mask_token with type "URL".)
    """
    return full_mask_token(url, "URL")

def partial_mask_url(url: str) -> str:
    """
    Partially mask a URL by processing both the domain (netloc) and the path.
    For the domain, split on '.' and for each label:
      - If length >= 6, preserve the first 2 characters and replace the 3rd with '*'
        then append the remainder (e.g. "linkedin" becomes "li***din").
      - Otherwise, fully mask that label.
    For the path, split on '/' and for each non-empty segment that is at least 3 characters,
    apply partial_mask_token.
    The scheme, query, and fragment remain unchanged.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    path = parsed.path
    params = parsed.params
    query = parsed.query
    fragment = parsed.fragment

    # Process netloc (handle potential port numbers)
    if ':' in netloc:
        domain, port = netloc.split(':', 1)
        port = ':' + port
    else:
        domain = netloc
        port = ''
    parts = domain.split('.')
    masked_parts = []
    for part in parts:
        if len(part) >= 6:
            # print(part + " " + part[:2] + "*" * (len(part)-5) + part[-3:]) 
            # print("\n")
            masked_parts.append(part[:2] + "*" * (len(part)-5) + part[-3:])
        else:
            masked_parts.append('*' * len(part))
    masked_netloc = '.'.join(masked_parts) + port

    # Process path: for each segment, if length >= 3, apply partial_mask_token; else leave as is.
    path_segments = path.split('/')
    masked_segments = []
    for seg in path_segments:
        if seg and len(seg) >= 3:
            masked_segments.append(partial_mask_token(seg))
        else:
            masked_segments.append(seg)
    masked_path = '/'.join(masked_segments)
    
    return urlunparse((scheme, masked_netloc, masked_path, params, query, fragment))

def normalize_entity(entity: dict) -> str:
    """
    Normalize the entity type from different detectors.
    Maps Spacy/HF labels (with B-/I- prefixes) and regex detectors to a consistent set.
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
            "PASSWORD": "PASSWORD"
        }
        return mapping.get(raw.upper(), raw.upper())
    return None

def get_presidio_entities(text: str) -> list:
    """
    Use Presidio Analyzer to detect PII.
    Returns a list of dictionaries with keys: 'entity_group', 'start', 'end', and 'score'.
    """
    results = ANALYZER_ENGINE.analyze(text=text, language="en")
    presidio_entities = []
    for res in results:
        presidio_entities.append({
            'entity_group': res.entity_type,
            'start': res.start,
            'end': res.end,
            'score': res.score
        })
    return presidio_entities

def get_hf_entities(text: str) -> list:
    """
    Use Hugging Face NER pipeline to detect entities.
    Returns a list of dictionaries with keys: 'entity', 'start', 'end', and 'score'.
    """
    return HF_PIPELINE(text)

def get_regex_entities(text: str) -> list:
    """
    Use regular expressions to detect additional PII patterns.
    Covers SSN, IP addresses, URLs, dates, and phone numbers.
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
        {"type": "PASSWORD", "pattern": r"(?i)\bpassword\b(?:\s*(?:was|is|:))\s*\S+"}
    ]
    regex_entities = []
    for item in regex_patterns:
        for match in re.finditer(item["pattern"], text):
            start, end = match.span()
            regex_entities.append({
                "entity_group": item["type"],
                "start": start,
                "end": end,
                "score": 0.9  # High confidence for regex matches
            })
    return regex_entities

def deduplicate_entities(entities: list) -> list:
    """
    Deduplicate overlapping entities by choosing the one with the highest score,
    filtering out entities below the confidence threshold.
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

def anonymize_text(text: str, pii_options: dict = None, full_redaction: bool = True) -> str:
    """
    Detect sensitive information using Presidio, Hugging Face, and regex methods;
    deduplicate overlapping entities; and mask each detected token based on user-specified options.
    
    Only entities with types enabled in pii_options (True) will be anonymized.
      - If full_redaction is True, all enabled entities are replaced with a hash‐based pseudonym.
      - If full_redaction is False, EMAIL_ADDRESS and URL entities use custom masking functions,
        while other entities are partially masked.
    If pii_options is None, all PII types are anonymized.
    """
    default_options = {
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
        "PASSWORD": True
    }
    if pii_options is None:
        pii_options = default_options
    else:
        for key, value in default_options.items():
            if key not in pii_options:
                pii_options[key] = value

    logger.info("Starting entity detection using multiple detectors...")
    presidio_entities = get_presidio_entities(text)
    hf_entities = get_hf_entities(text)
    regex_entities = get_regex_entities(text)
    
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
        if pii_options.get(norm_type, True):
            if full_redaction:
                # Use hash-based pseudonym for all types.
                masked = full_mask_token(original_token, norm_type)
            else:
                if norm_type == "EMAIL_ADDRESS":
                    masked = mask_email(original_token)
                elif norm_type == "URL":
                    # For partial redaction of URLs, mask both the domain and the path.
                    masked = partial_mask_url(original_token)
                else:
                    masked = partial_mask_token(original_token)
            logger.debug(f"Masking token '{original_token}' ({norm_type}) to '{masked}'")
            text = text[:start] + masked + text[end:]
    return text

# Create a Flask app instance.
app = Flask(__name__)

@app.route("/anonymize", methods=["POST"])
def anonymize():
    """
    Expects a JSON payload with a "text" field, an optional "options" field, and
    an optional boolean "full_redaction" field.
    Returns the anonymized text as a JSON response.
    """
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data["text"]
    pii_options = data.get("options", None)
    full_redaction = data.get("full_redaction", True)
    anonymized_text = anonymize_text(input_text, pii_options, full_redaction)
    return jsonify({"anonymized_text": anonymized_text})

if __name__ == "__main__":
    port = 8000
    print(f"Server is ready and serving on port {port}")
    app.run(debug=True, port=port)
