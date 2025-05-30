import sys
import os
import hashlib
import logging
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import requests
from dotenv import load_dotenv
from typing import List, Dict

# --- Load Environment Variables ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedactifyMCPServer")

# --- Initialize FastAPI App ---
app = FastAPI(title="Redactify MCP Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONT_END_URL", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Function to Load JSON Config ---
def load_json_config(filename, default_value=None):
    """Loads configuration from a JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded configuration from {filename}")
            return data
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filepath}. Using default value: {default_value}")
        return default_value
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}. Using default value: {default_value}")
        return default_value
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}", exc_info=True)
        return default_value

# --- Load Configurations from JSON and Environment ---
CONFIG_STATIC = load_json_config('config_static.json', {})

CONFIG = {
    **CONFIG_STATIC,
    "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5)),
    "max_workers": int(os.environ.get("MAX_WORKERS", 8)),
    "enable_medical_pii": os.environ.get("ENABLE_MEDICAL_PII", "True").lower() == "true",
    "enable_technical_ner": os.environ.get("ENABLE_TECHNICAL_NER", "True").lower() == "true",
    "enable_pii_specialized": os.environ.get("ENABLE_PII_SPECIALIZED", "True").lower() == "true",
    "enable_legal_ner": os.environ.get("ENABLE_LEGAL_NER", "True").lower() == "true",
    "enable_financial_ner": os.environ.get("ENABLE_FINANCIAL_NER", "True").lower() == "true",
    "mcp_classifier_url": os.environ.get("MCP_CLASSIFIER_URL", "http://localhost:8001"),
    "a2a_general_url": os.environ.get("A2A_GENERAL_URL", "http://localhost:8002"),
    "a2a_medical_url": os.environ.get("A2A_MEDICAL_URL", "http://localhost:8003"),
    "a2a_technical_url": os.environ.get("A2A_TECHNICAL_URL", "http://localhost:8004"),
    "a2a_pii_specialized_url": os.environ.get("A2A_PII_SPECIALIZED_URL", "http://localhost:8005"),
    "a2a_legal_url": os.environ.get("A2A_LEGAL_URL", "http://localhost:8006"),
    "a2a_financial_url": os.environ.get("A2A_FINANCIAL_URL", "http://localhost:8007"),
    "classifier_timeout": int(os.environ.get("CLASSIFIER_TIMEOUT", 10)), # Increased timeout for LLM
    "ner_agent_timeout": 180,
    "parallel_detector_timeout": 200,
    "context_window": 40,
    "entity_confidence_threshold": 0.1,
    "enable_context_detection": True,
    "enable_fallback_name_detector": True,
}

BLOCKLIST_LIST = load_json_config('blocklist.json', [])
COMMON_NAME_WORDS_LIST = load_json_config('common_name_words.json', [])
DEFAULT_PII_OPTIONS = load_json_config('default_pii_options.json', {})
PSEUDONYMIZE_TYPES_LIST = load_json_config('pseudonymize_types.json', [])
ENTITY_TYPE_MAPPING = load_json_config('entity_type_mapping.json', {})

BLOCKLIST = set(BLOCKLIST_LIST)
COMMON_NAME_WORDS = set(COMMON_NAME_WORDS_LIST)
PSEUDONYMIZE_TYPES = set(PSEUDONYMIZE_TYPES_LIST)
BLOCKLIST.update(COMMON_NAME_WORDS)

# --- Load Regex Patterns from JSON ---
def load_regex_patterns(filepath=os.path.join(os.path.dirname(__file__), 'regex_patterns.json')):
    """Loads and compiles regex patterns from a JSON file."""
    patterns = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        logger.info(f"Successfully loaded {len(patterns_data)} regex pattern definitions from {filepath}")
        for p_def in patterns_data:
            pattern_str = p_def.get('pattern')
            pattern_type = p_def.get('type', 'N/A')
            if not pattern_str:
                logger.warning(f"Skipping pattern type '{pattern_type}' due to missing 'pattern' field.")
                continue
            try:
                p_def['compiled_pattern'] = re.compile(pattern_str, re.IGNORECASE)
                patterns.append(p_def)
            except re.error as e:
                logger.error(f"Failed to compile regex for type '{pattern_type}' with pattern '{pattern_str}': {e}")
            except Exception as comp_e:
                logger.error(f"Unexpected error compiling regex pattern '{pattern_type}': {comp_e}")
        logger.info(f"Successfully compiled {len(patterns)} regex patterns.")
        return patterns
    except FileNotFoundError:
        logger.error(f"Regex patterns file not found: {filepath}. Regex detection will be disabled.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos}). Regex detection may fail.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading regex patterns from {filepath}: {e}", exc_info=True)
        return []

REGEX_PATTERNS = load_regex_patterns()
if not REGEX_PATTERNS:
    logger.warning("REGEX_PATTERNS list is empty or loading failed. Regex detection will not function correctly.")

# --- Initialize Core Services (Presidio) ---
def load_core_services():
    logger.info("Loading core services (Presidio)...")
    services = {}
    try:
        services["presidio"] = AnalyzerEngine()
        logger.info("Loaded Presidio Analyzer successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading Presidio Analyzer: {e}", exc_info=True)
        raise RuntimeError("Failed to load core Presidio service. Cannot start.")
    return services

CORE_SERVICES = load_core_services()

# --- Agent/Service Communication ---
def get_text_classification(text: str) -> List[str]:
    """Calls the MCP Text Classifier service using JSON-RPC."""
    url = CONFIG["mcp_classifier_url"] + "/mcp"  # Changed from /predict to /mcp
    default_classification = ["general"]
    
    try:
        # JSON-RPC 2.0 format
        payload = {
            "jsonrpc": "2.0",
            "method": "predict",
            "params": {
                "inputs": text
            },
            "id": 1  # Request identifier
        }
        
        response = requests.post(url, json=payload, timeout=CONFIG["classifier_timeout"])
        response.raise_for_status()
        
        # Extract classifications from JSON-RPC result
        result = response.json()
        if "error" in result:
            logger.error(f"JSON-RPC error from classifier: {result['error']}")
            return default_classification
            
        classifications = result.get("result", {}).get("classifications", default_classification)
        return classifications
    except Exception as e:
        logger.error(f"Error calling classifier: {e}", exc_info=True)
        return default_classification

def invoke_a2a_agent(agent_url: str, text: str) -> list:
    """Calls a specific A2A NER Agent using MCP JSON-RPC."""
    url = agent_url + "/mcp"
    try:
        # JSON-RPC 2.0 format
        payload = {
            "jsonrpc": "2.0",
            "method": "predict",
            "params": {
                "inputs": text
            },
            "id": 1
        }
        
        response = requests.post(url, json=payload, timeout=CONFIG["ner_agent_timeout"])
        response.raise_for_status()
        
        # Extract entities from JSON-RPC result format
        result = response.json()
        if "error" in result:
            logger.error(f"JSON-RPC error from agent {agent_url}: {result['error']}")
            return []
            
        entities = result.get("result", {}).get("entities", [])
        # Log what entities were found
        logger.debug(f"Agent {agent_url} returned {len(entities)} entities: {[e.get('entity_group', 'unknown') for e in entities]}")
        return entities
    except Exception as e:
        logger.error(f"Error calling agent {agent_url}: {e}", exc_info=True)
        return []

# --- Internal Detection Functions (Presidio, Regex) ---
def get_presidio_entities(text: str) -> list:
    """Detect entities using internal Presidio Analyzer."""
    try:
        analyzer = CORE_SERVICES.get("presidio")
        if not analyzer:
            logger.error("Presidio Analyzer not loaded.")
            return []
        results = analyzer.analyze(text=text, language="en")
        presidio_entities = [{
            'entity_group': res.entity_type,
            'start': res.start,
            'end': res.end,
            'score': res.score,
            'detector': 'presidio_internal'
        } for res in results]
        logger.debug(f"Presidio found {len(presidio_entities)} entities.")
        return presidio_entities
    except Exception as e:
        logger.error(f"Error running internal Presidio Analyzer: {e}", exc_info=True)
        return []

def has_context(text, span_start, span_end, context_words):
    """Checks if context words appear near the entity or if there are contextual indicators."""
    if not context_words: return True
    matched_text = text[span_start:span_end]
    if matched_text in BLOCKLIST: return False
    if matched_text.startswith("Project") and len(matched_text.split()) <= 2: return False
    context_size = CONFIG.get("context_window", 40)
    text_before = text[:span_start].split()[-context_size:] if span_start > 0 else []
    text_after = text[span_end:].split()[:context_size] if span_end < len(text) else []
    context_text = ' '.join(text_before + text_after).lower()
    high_risk_types = ["password", "key", "token", "secret", "credential", "api"]
    needs_strong_context = len(matched_text) > 10 and re.search(r'[A-Za-z0-9]{10,}', matched_text)
    for word in context_words:
        if word.lower() in context_text:
            if needs_strong_context and not any(h_risk in context_text for h_risk in high_risk_types): continue
            return True
    nearby_text = text[max(0, span_start-20):min(len(text), span_end+10)]
    indicators = r'(?::|=|is\s+|was\s+reset\s+to\s+)'
    if re.search(indicators + r'\s*' + re.escape(matched_text), nearby_text, re.IGNORECASE): return True
    return False

def get_regex_entities(text: str) -> list:
    """Enhanced regex-based detection with context awareness using loaded patterns."""
    regex_entities = []
    if not REGEX_PATTERNS:
        return []

    for pattern_def in REGEX_PATTERNS:
        compiled_pattern = pattern_def.get('compiled_pattern')
        if not compiled_pattern:
            continue

        try:
            for match in compiled_pattern.finditer(text):
                start, end = match.span()
                matched_text = text[start:end]

                if len(matched_text) < 3 and not pattern_def.get("context"): continue
                if not matched_text.strip(): continue

                if has_context(text, start, end, pattern_def.get("context", [])):
                    use_group = pattern_def.get("use_group", True)
                    group_start, group_end = start, end

                    if compiled_pattern.groups > 0 and use_group:
                        try:
                            gs, ge = match.span(1)
                            if gs >= 0 and ge >= gs:
                                group_start, group_end = gs, ge
                            else:
                                logger.debug(f"Invalid group span (span(1)) for regex {pattern_def['type']} on '{matched_text}'. Using full match.")
                        except IndexError:
                            logger.debug(f"Group 1 did not participate in match for regex {pattern_def['type']} on '{matched_text}'. Using full match.")

                    regex_entities.append({
                        "entity_group": pattern_def["type"],
                        "start": group_start,
                        "end": group_end,
                        "score": 0.9,
                        "detector": "regex_internal"
                    })
        except Exception as e:
            logger.error(f"Error processing regex pattern {pattern_def.get('type', 'N/A')}: {e}", exc_info=False)

    return regex_entities

def add_contextual_entities(text: str) -> list:
    """Pre-process text to identify ambiguous entities based on context."""
    contextual_entities = []
    
    # Common ambiguous company names (lowercase to handle case-insensitive matching)
    ambiguous_companies = {
        "apple": "ORGANIZATION",
        "amazon": "ORGANIZATION",
        "google": "ORGANIZATION",
        "meta": "ORGANIZATION",
        "microsoft": "ORGANIZATION",
        "oracle": "ORGANIZATION", 
        "shell": "ORGANIZATION",
        "twitter": "ORGANIZATION",
        "uber": "ORGANIZATION"
    }
    
    # Context indicators for different entity types
    context_indicators = {
        "ORGANIZATION": [
            r'\b(work|working|job|career|company|corporation|inc|firm)\b',
            r'\b(tech|technology|product|products|device|phone|computer)\b',
            r'\b(stock|share|market|investor|investment)\b',
            r'\b(ceo|founder|employee|staff|team)\b'
        ]
    }
    
    # Process text for ambiguous companies
    for company_name, entity_type in ambiguous_companies.items():
        # Case insensitive search
        pattern = rf'\b{re.escape(company_name)}\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            
            # Check if company context exists nearby
            context_window = 40  # Check 40 chars before and after
            context_text = text[max(0, start-context_window):min(len(text), end+context_window)].lower()
            
            # Check for context indicators
            is_context_match = False
            for indicator in context_indicators.get(entity_type, []):
                if re.search(indicator, context_text, re.IGNORECASE):
                    is_context_match = True
                    break
                    
            if is_context_match:
                contextual_entities.append({
                    'entity_group': entity_type,
                    'start': start,
                    'end': end,
                    'score': 0.88,  # High confidence due to context match
                    'detector': 'context_entity_detector'
                })
    
    # Add more specialized entity detection here in the future
    
    return contextual_entities

# --- Core Logic (Modified run_detectors) ---
def run_detectors(text: str) -> tuple[list, list, list]:
    """
    Runs internal detectors and calls relevant A2A agents based on classification list.
    Returns a tuple: (all_entities, classifications, invoked_agent_urls)
    """
    all_entities = []
    start_time = time.time()

    # Add context-aware entity detection
    contextual_entities = add_contextual_entities(text)
    if contextual_entities:
        all_entities.extend(contextual_entities)
        logger.info(f"Context entity detector found {len(contextual_entities)} entities")

    classifications = get_text_classification(text)
    logger.info(f"Text classified as: {classifications}. Determining detectors to run...")

    agent_calls = []
    agent_calls.append((invoke_a2a_agent, CONFIG["a2a_general_url"]))

    if CONFIG["enable_pii_specialized"]:
        agent_calls.append((invoke_a2a_agent, CONFIG["a2a_pii_specialized_url"]))

    if "medical" in classifications and CONFIG["enable_medical_pii"]:
        agent_calls.append((invoke_a2a_agent, CONFIG["a2a_medical_url"]))
    if "technical" in classifications and CONFIG["enable_technical_ner"]:
        agent_calls.append((invoke_a2a_agent, CONFIG["a2a_technical_url"]))
    if "legal" in classifications and CONFIG.get("enable_legal_ner", False):
        agent_calls.append((invoke_a2a_agent, CONFIG["a2a_legal_url"]))
    if "financial" in classifications and CONFIG.get("enable_financial_ner", False):
        agent_calls.append((invoke_a2a_agent, CONFIG["a2a_financial_url"]))

    unique_agent_calls = list({call[1]: call for call in agent_calls}.values())
    invoked_agent_urls = [url for func, url in unique_agent_calls]

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []

        futures.append(executor.submit(get_presidio_entities, text))
        futures.append(executor.submit(get_regex_entities, text))
        futures.append(executor.submit(add_contextual_entities, text))

        for func, url in unique_agent_calls:
            logger.debug(f"Submitting call to agent: {url}")
            futures.append(executor.submit(func, url, text))

        results = [future.result() for future in as_completed(futures)]

        # Get regular agents' results
        for result in results:
            if isinstance(result, list):
                all_entities.extend(result)

        # Add fallback name detection, but with improved filtering
        # Check if presidio already found person entities first
        presidio_found_persons = any(
            e.get('entity_group', '').upper() == 'PERSON' and e.get('detector') == 'presidio_internal'
            for e in all_entities
        )
        
        # Only run fallback detection if presidio didn't find persons and it's enabled
        if not presidio_found_persons and CONFIG.get("enable_fallback_name_detector", True):
            # Extended list of common words that shouldn't be detected as names
            common_non_names = [
                'the', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 'my', 'your', 'his', 'her',
                'our', 'their', 'its', 'if', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'as', 'of', 'from',
                'about', 'ssn', 'id', 'cc', 'cv', 'cvv', 'pin', 'no', 'yes', 'ok', 'new', 'old', 'first', 'last'
            ]
            
            # Find potential names using capital letter pattern, but with better filtering
            name_patterns = [r'\b([A-Z][a-z]{2,})\b']  # Minimum 3 chars, first letter capital, rest lowercase
            for pattern in name_patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1)
                    
                    # Skip if it's a common non-name word
                    if name.lower() in common_non_names:
                        continue
                        
                    # Skip if it appears at sentence beginning (likely not a name)
                    pre_context = text[max(0, match.start(1)-20):match.start(1)].strip()
                    if pre_context == "" or pre_context.endswith(('.', '!', '?', '\n', '\r')):
                        # This might be start of sentence, check if it's a common sentence starter
                        if name in ["The", "This", "That", "These", "Those", "My", "Your", "Our", "Their", "It"]:
                            continue
                    
                    # Skip if it's in the common name words list (already loaded globally)
                    if name.lower() in COMMON_NAME_WORDS:
                        continue
                        
                    # Add with a lower confidence score to avoid overriding better matches
                    all_entities.append({
                        'entity_group': 'PERSON',
                        'start': match.start(1),
                        'end': match.end(1),
                        'score': 0.65,  # Lower score so it doesn't override better matches
                        'word': name,
                        'detector': 'fallback_name_detector'
                    })

        # Detection for names with titles is more reliable, so keep it
        potential_names = re.findall(r'(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.) ([A-Z][a-zA-Z\-]{2,})', text)
        for name in potential_names:
            if name.lower() in COMMON_NAME_WORDS:
                continue
            all_entities.append({
                'entity_group': 'PERSON',
                'start': text.find(name),
                'end': text.find(name) + len(name),
                'score': 0.92,  # High confidence for names with titles
                'detector': 'title_name_detector'
            })

    duration = time.time() - start_time
    logger.info(f"All detection tasks completed in {duration:.2f}s. Total potential entities: {len(all_entities)}")
    return all_entities, classifications, invoked_agent_urls

# --- Additional Detector Function for Internal-Only Detection ---
def run_internal_detectors(text: str) -> list:
    """
    Runs only the internal detectors (Presidio, Regex, and Context-aware).
    Does not call any external A2A agents.
    Returns a list of detected entities.
    """
    all_entities = []
    start_time = time.time()

    # Get entities from internal detectors
    presidio_entities = get_presidio_entities(text)
    regex_entities = get_regex_entities(text)
    contextual_entities = add_contextual_entities(text)
    
    # Combine all internal results
    all_entities.extend(presidio_entities)
    all_entities.extend(regex_entities)
    all_entities.extend(contextual_entities)
    
    # Add fallback name detection (same as in run_detectors)
    name_patterns = [r'\b([A-Z][a-z]{2,})\b']
    for pattern in name_patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1)
            # Skip common non-name words
            if name.lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 'my', 'your', 'his', 'her',
                                'our', 'their', 'its', 'if', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'as', 'of', 'from',
                                'about', 'ssn', 'id', 'cc', 'cv', 'cvv', 'pin', 'no', 'yes', 'ok', 'new', 'old', 'first', 'last']:
                continue
            all_entities.append({
                'entity_group': 'PERSON',
                'start': match.start(1),
                'end': match.end(1),
                'score': 0.65,
                'word': name,
                'detector': 'fallback_name_detector'
            })

    # Detection for names with titles
    potential_names = re.findall(r'(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.) ([A-Z][a-zA-Z\-]{2,})', text)
    for name in potential_names:
        if name.lower() in COMMON_NAME_WORDS:
            continue
        all_entities.append({
            'entity_group': 'PERSON',
            'start': text.find(name),
            'end': text.find(name) + len(name),
            'score': 0.92,  # High confidence for names with titles
            'detector': 'title_name_detector'
        })

    duration = time.time() - start_time
    logger.info(f"Internal detection completed in {duration:.2f}s. Found {len(all_entities)} potential entities.")
    return all_entities

# --- Masking, Merging, Verification Logic ---
def pseudonymize_value(value: str, entity_type: str) -> str:
    h = hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    return f"[{entity_type}-{h}]"

def full_mask_token(token: str, entity_type: str) -> str:
    if entity_type is None: return '*' * len(token)
    return pseudonymize_value(token, entity_type.upper())

def partial_mask_token(token: str) -> str:
    n = len(token); mask_char = CONFIG.get("partial_mask_char", "*")
    if n <= 2: return mask_char * n
    elif n <= 5: return token[0] + mask_char * (n - 1)
    elif n <= 10: return token[0:2] + mask_char * (n - 4) + token[-2:]
    else: return token[0:2] + mask_char * (n - 5) + token[-3:]

def mask_email(email: str) -> str:
    mask_char = CONFIG.get("partial_mask_char", "*")
    try: local, domain = email.split("@")
    except Exception: return partial_mask_token(email)
    if len(local) > 4: local_masked = local[0:2] + mask_char * (len(local) - 4) + local[-2:]
    else: local_masked = local[0] + mask_char * (len(local) - 1)
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        tld = domain_parts[-1]; domain_name = '.'.join(domain_parts[:-1])
        if len(domain_name) > 5: domain_masked = domain_name[0:2] + mask_char * (len(domain_name) - 2)
        else: domain_masked = mask_char * len(domain_name)
        masked_domain = domain_masked + '.' + tld
    else: masked_domain = mask_char * len(domain)
    return local_masked + "@" + masked_domain

def mask_url(url: str) -> str: return full_mask_token(url, "URL")

def partial_mask_url(url: str) -> str:
    mask_char = CONFIG.get("partial_mask_char", "*")
    try: parsed = urlparse(url)
    except Exception: return partial_mask_token(url)
    scheme, netloc, path, params, query, fragment = parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
    if ':' in netloc: domain, port = netloc.split(':', 1); port = ':' + port
    else: domain, port = netloc, ''
    parts = domain.split('.'); masked_parts = []
    for i, part in enumerate(parts):
        if i == len(parts) - 1 and len(parts) > 1: masked_parts.append(part)
        elif len(part) > 3: masked_parts.append(part[0:2] + mask_char * (len(part) - 2))
        else: masked_parts.append(mask_char * len(part))
    masked_netloc = '.'.join(masked_parts) + port
    if path:
        path_segments = path.split('/'); masked_segments = []
        for segment in path_segments:
            if not segment: masked_segments.append(segment); continue
            if segment.lower() in ['api', 'v1', 'v2', 'v3', 'dashboard', 'login', 'public', 'static']: masked_segments.append(segment)
            elif len(segment) >= 5: masked_segments.append(segment[0:2] + mask_char * (len(segment) - 2))
            else: masked_segments.append(mask_char * len(segment))
        masked_path = '/'.join(masked_segments)
    else: masked_path = path
    return urlunparse((scheme, masked_netloc, masked_path, params, query, fragment))

def mask_phone(phone: str) -> str:
    mask_char = CONFIG.get("partial_mask_char", "*")
    digits_only = re.sub(r'[^0-9+]', '', phone)
    if len(digits_only) <= 4: return mask_char * len(phone)
    if digits_only.startswith('+'):
        prefix_end = digits_only.find('9');
        if prefix_end != -1 and prefix_end < 4: prefix = digits_only[:prefix_end+1]; main_number = digits_only[prefix_end+1:]
        else: prefix = '+'; main_number = digits_only[1:]
    else: prefix = ''; main_number = digits_only
    if len(main_number) > 4: masked_number = mask_char * (len(main_number) - 4) + main_number[-4:]
    else: masked_number = mask_char * len(main_number)
    masked_digits = prefix + masked_number; result = ''; digit_index = 0
    for char in phone:
        if char.isdigit() or char == '+':
            if digit_index < len(masked_digits): result += masked_digits[digit_index]; digit_index += 1
            else: result += mask_char
        else: result += char
    return result

def smart_partial_mask(text: str, entity_type: str) -> str:
    mask_char = CONFIG.get("partial_mask_char", "*")
    if not text: return text
    if entity_type == "EMAIL_ADDRESS": return mask_email(text)
    elif entity_type == "URL": return partial_mask_url(text)
    elif entity_type == "PHONE_NUMBER": return mask_phone(text)
    elif entity_type == "CREDIT_CARD":
        digits_only = re.sub(r'[^0-9]', '', text)
        if len(digits_only) >= 4: return mask_char * (len(text) - 4) + text[-4:]
    elif entity_type == "SSN":
        if len(text) > 4: return mask_char * (len(text) - 4) + text[-4:]
    elif entity_type in ["PASSWORD", "API_KEY", "AUTHENTICATION", "DEPLOY_TOKEN"]:
        if len(text) > 8: return text[:2] + mask_char * (len(text) - 2)
        else: return mask_char * len(text)
    elif entity_type == "DATE_TIME":
        if len(text) > 6 and re.search(r'\d{4}', text):
            date_parts = re.split(r'[-/\s:]', text)
            if len(date_parts) > 2 and len(date_parts[0]) == 4:
                date_parts[0] = mask_char * 4
                original_separators = re.findall(r'[-/\s:]', text)
                reconstructed = date_parts[0]
                for i, part in enumerate(date_parts[1:]):
                    if i < len(original_separators):
                        reconstructed += original_separators[i] + part
                    else:
                        reconstructed += "-" + part
                return reconstructed
            elif len(date_parts) > 2 and len(date_parts[-1]) == 4:
                 date_parts[-1] = mask_char * 4
                 original_separators = re.findall(r'[-/\s:]', text)
                 reconstructed = date_parts[0]
                 for i, part in enumerate(date_parts[1:]):
                     if i < len(original_separators):
                         reconstructed += original_separators[i] + part
                     else:
                         reconstructed += "-" + part
                 return reconstructed

    return partial_mask_token(text)

def verify_entity(entity_type, text, confidence_score):
    if text.strip() in BLOCKLIST: return False
    if entity_type is None: return False
    if entity_type == "PERSON" and text.startswith("Project"): return False
    if entity_type == "PERSON" and text.lower() in ["team", "hi team", "hello team"]: return False
    if ((text.isdigit() and len(text) < 4) or text.startswith("#") or
        (len(text.split()) == 1 and len(text) < 4 and not re.search(r'\d', text)) or
        text.lower() in ["from", "to", "hi", "hello", "subject", "best", "regards"]): return False
    if entity_type == "PERSON" and len(text.split()) == 1 and text in COMMON_NAME_WORDS: return False
    type_confidence_thresholds = {
        "PASSWORD": 0.75, "API_KEY": 0.75, "CREDENTIAL": 0.75, "FINANCIAL": 0.7,
        "ORGANIZATION": 0.60,  # Lowered from 0.65 to catch more organizations
        "DEVICE": 0.75, "MEDICAL": 0.7, "PERSON": 0.68
    }
    min_confidence = type_confidence_thresholds.get(entity_type, CONFIG.get("confidence_threshold", 0.68))
    if confidence_score < min_confidence: return False
    if entity_type == "PASSWORD":
        has_digits = bool(re.search(r'\d', text)); has_letters = bool(re.search(r'[a-zA-Z]', text)); has_special = bool(re.search(r'[^a-zA-Z0-9\s]', text))
        if not ((has_digits and has_letters) or (has_special and (has_digits or has_letters))): return False
    elif entity_type == "API_KEY":
        if len(text) < 12: return False
    if entity_type == "CREDENTIAL" and text.lower() in ["user", "account", "login"]: return False
    if entity_type == "DEVICE" and text in ["iPhone", "MacBook", "Android", "Windows"]: return bool(re.search(r'[a-zA-Z]+\s+[a-zA-Z0-9]+', text))
    return True

def normalize_entity(entity: dict) -> str:
    if 'entity_group' in entity: raw_type = entity['entity_group'].upper()
    elif 'entity' in entity:
        raw_type = entity['entity'].upper()
        if raw_type.startswith("B-") or raw_type.startswith("I-"): raw_type = raw_type[2:]
    else: return None
    return ENTITY_TYPE_MAPPING.get(raw_type, raw_type)

def score_entity(entity: dict, text: str) -> float:
    score = entity.get('score', 0.6); entity_text = text[entity['start']:entity['end']]; entity_type = normalize_entity(entity)
    if entity_text in BLOCKLIST or entity_type is None: return 0.0
    if entity_text.lower() in ["project", "team", "update", "request", "from", "subject"] and entity_type == "PERSON": return 0.0
    if entity.get('detected_by', 1) > 1: score = min(1.0, score + 0.1 * entity['detected_by'])
    if len(entity_text) < 4 and entity_type not in ["SSN", "DATE_TIME", "FINANCIAL"]: score = max(0.0, score - 0.2)
    if entity_type == "PASSWORD" and re.search(r'[A-Za-z].*[0-9]|[0-9].*[A-Za-z]', entity_text): score = min(1.0, score + 0.15)
    if entity_type == "API_KEY" and len(entity_text) >= 20: score = min(1.0, score + 0.2)
    if entity_type == "DEPLOY_TOKEN" and entity_text.startswith(('gh', 'gl')): score = min(1.0, score + 0.25)
    if entity_type == "MEDICAL" and entity.get('detector', '').startswith('a2a_ner_medical'): score = min(1.0, score + 0.2)
    context_indicators = {
        "PASSWORD": ["password", "pwd", "pass"], "API_KEY": ["api", "key"], "DEPLOY_TOKEN": ["token", "deploy", "access"],
        "CREDENTIAL": ["login", "username", "user"], "MEDICAL": ["patient", "medical", "health", "doctor", "hospital"]
    }
    if entity_type in context_indicators:
        context = text[max(0, entity['start']-30):entity['start']].lower()
        for indicator in context_indicators[entity_type]:
            if indicator.lower() in context: score = min(1.0, score + 0.2); break
    common_words = ["submitted", "description", "reference", "customer", "account", "order", "number"]
    if entity_text.lower() in common_words: score = max(0.0, score - 0.3)
    return score

def select_best_entity(entities: list, text: str) -> dict:
    if not entities: return None
    if len(entities) == 1:
        entity_text = text[entities[0]['start']:entities[0]['end']]; entity_type = entities[0]['normalized_type']
        if not verify_entity(entity_type, entity_text, entities[0]['score']): return None
        return entities[0]
    first_type = entities[0]['normalized_type']
    if all(e['normalized_type'] == first_type for e in entities):
        best = max(entities, key=lambda e: (e['end'] - e['start'], e['score']))
        best['detected_by'] = len(set(e['detector'] for e in entities))
        entity_text = text[best['start']:best['end']]
        if not verify_entity(best['normalized_type'], entity_text, best['score']): return None
        return best
    detector_priority = {
        'context_entity_detector': 3.2,  # New highest priority detector
        'regex_internal': 3, 'presidio_internal': 2.8, 'a2a_ner_pii_specialized': 2.5,
        'a2a_ner_medical': 2.2,  # Updated from model1
        'a2a_ner_technical': 1.8, 'a2a_ner_general': 1.5
    }
    type_priority = {
        "PASSWORD": 1.5, "API_KEY": 1.4, "CREDENTIAL": 1.3, "FINANCIAL": 1.3, "SSN": 1.5,
        "CREDIT_CARD": 1.4, "MEDICAL": 1.3, "IP_ADDRESS": 1.2, "MAC_ADDRESS": 1.2,
        "EMAIL_ADDRESS": 1.1, "PERSON": 1.0
    }
    best_score = -1; best_entity = None
    for entity in entities:
        entity_text = text[entity['start']:entity['end']]
        if not verify_entity(entity['normalized_type'], entity_text, entity['score']): continue
        base_score = entity['score']
        detector_boost = detector_priority.get(entity['detector'], 1)
        length_factor = min(1.0, (entity['end'] - entity['start']) / 20)
        type_boost = type_priority.get(entity['normalized_type'], 1)
        composite_score = base_score * detector_boost * type_boost * (1 + 0.2 * length_factor)
        if composite_score > best_score: best_score = composite_score; best_entity = entity
    return best_entity

def merge_overlapping_entities(entities: List[Dict], text: str) -> List[Dict]:
    """Merge overlapping entity spans from different detectors."""
    if not entities:
        return []
    
    # Sort by start position, then by longest span, then by highest score
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -len(text[x['start']:x['end']]), -x.get('score', 0)))
    
    # Debug entity information
    for i, entity in enumerate(sorted_entities):
        logger.debug(f"Entity {i}: {entity.get('entity_group', 'UNKNOWN')} ({entity['start']}:{entity['end']}) "
                    f"'{text[entity['start']:entity['end']]}' "
                    f"score: {entity.get('score', 0):.4f}, detector: {entity.get('detector', 'unknown')}")
    
    # Remove entities below confidence threshold
    confidence_threshold = CONFIG.get("entity_confidence_threshold", 0.1)
    filtered_entities = []
    for entity in sorted_entities:
        if entity.get('score', 1.0) >= confidence_threshold:
            filtered_entities.append(entity)
        else:
            logger.debug(f"Filtered out low-confidence entity: {entity.get('entity_group', 'UNKNOWN')} "
                        f"'{text[entity['start']:entity['end']]}' (score: {entity.get('score', 0):.4f})")
    
    # Process filtered entities to merge overlapping spans
    result = []
    i = 0
    while i < len(filtered_entities):
        current = filtered_entities[i]
        current_start, current_end = current['start'], current['end']
        
        # Find all entities that overlap with current one
        overlapping = []
        for j in range(len(filtered_entities)):
            if i != j:
                other = filtered_entities[j]
                if (current_start <= other['end'] and other['start'] <= current_end):
                    overlapping.append(other)
        
        # If there are overlapping entities, select the best one
        if overlapping:
            # Add normalized types to current and overlapping entities
            current['normalized_type'] = normalize_entity(current)
            for o in overlapping:
                o['normalized_type'] = normalize_entity(o)
            
            # Include current in the selection pool
            candidates = [current] + overlapping
            best = select_best_entity(candidates, text)
            
            if best:
                result.append(best)
                
            # Skip all the entities that overlapped with this one
            skip_indices = set([filtered_entities.index(o) for o in overlapping])
            i = max(skip_indices) + 1 if skip_indices else i + 1
        else:
            # No overlap, just add normalized type and append
            current['normalized_type'] = normalize_entity(current)
            result.append(current)
            i += 1
    
    logger.info(f"Original entities: {len(entities)}, After filtering: {len(filtered_entities)}, After merging: {len(result)}")
    return result

# --- Anonymization Function ---
def anonymize_text(text: str, pii_options: dict = None, full_redaction: bool = True) -> tuple[str, list, list]:
    """
    Anonymization function using the new distributed detection approach.
    Returns a tuple: (redacted_text, classifications, invoked_agent_urls)
    """
    if not text: return "", [], []
    options = DEFAULT_PII_OPTIONS.copy()
    if pii_options: options.update(pii_options)

    logger.info("Starting distributed entity detection...")
    start_time = time.time()

    all_entities, classifications, invoked_agent_urls = run_detectors(text)

    merged_entities = merge_overlapping_entities(all_entities, text)
    logger.info(f"Merged into {len(merged_entities)} distinct entities for redaction.")

    merged_entities.sort(key=lambda x: x['start'], reverse=True)

    redaction_count = 0
    redacted_spans = {}
    redacted_text = text

    for entity in merged_entities:
        start, end = entity['start'], entity['end']

        current_length = len(redacted_text)
        if start >= current_length or end > current_length or start < 0:
             logger.warning(f"Skipping entity with invalid span after modifications: {entity} (text length: {current_length})")
             continue

        if any(span_start <= start and end <= span_end for (span_start, span_end) in redacted_spans.keys()):
            continue

        original_token = redacted_text[start:end]
        entity_type = entity['normalized_type']

        if original_token in BLOCKLIST:
            logger.debug(f"Skipping redaction of blocklisted term '{original_token}'")
            continue

        if options.get(entity_type, True):
            try:
                if full_redaction: masked = full_mask_token(original_token, entity_type)
                else: masked = smart_partial_mask(original_token, entity_type)
                logger.debug(f"Masking '{original_token}' ({entity_type}) to '{masked}'")

                redacted_text = redacted_text[:start] + masked + redacted_text[end:]

                redacted_spans[(start, end)] = True
                redaction_count += 1

            except Exception as e:
                logger.error(f"Error masking token '{original_token}' type {entity_type}: {e}", exc_info=True)

    total_time = time.time() - start_time
    logger.info(f"Anonymization completed in {total_time:.2f} seconds, redacted {redaction_count} entities.")
    return redacted_text, classifications, invoked_agent_urls

# --- FastAPI Endpoints ---
@app.post("/anonymize")
async def anonymize(request: Request):
    """Main anonymization endpoint with detailed request logging."""
    try:
        payload = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not payload or "text" not in payload:
        logger.error("Missing 'text' field")
        raise HTTPException(status_code=400, detail="No text provided")

    input_text = payload["text"]
    pii_options = payload.get("options")
    full_redaction = payload.get("full_redaction", True)

    logger.info(f"--- /anonymize Request Received ---")
    try:
        log_payload = payload.copy()
        if len(log_payload.get("text", "")) > 500:
             log_payload["text"] = log_payload["text"][:500] + "... (truncated)"
        logger.info(f"Payload: {json.dumps(log_payload, indent=2)}")
    except Exception as log_e:
        logger.error(f"Error formatting payload for logging: {log_e}")
        logger.info(f"Raw Payload (approx): {payload}")

    try:
        result_text, classifications, invoked_agents = anonymize_text(input_text, pii_options, full_redaction)

        logger.info(f"MCP Classifications: {classifications}")
        logger.info(f"Invoked A2A Agents: {invoked_agents}")
        logger.info(f"--- /anonymize Request End ---")

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        return JSONResponse({
            "anonymized_text": result_text,
            "timestamp": current_time,
            "user": "rushilpatel21"
        })
    except Exception as e:
        logger.error(f"Error processing /anonymize request: {e}", exc_info=True)
        logger.info(f"--- /anonymize Request End (Error) ---")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint showing dispatcher status and agent URLs."""
    return JSONResponse({
        "status": "ok",
        "service": "Redactify MCP Server",
        "core_services": list(CORE_SERVICES.keys()),
        "version": "2.1.1-A2A-MCP-MultiClass",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "config_urls": {
            "classifier": CONFIG.get("mcp_classifier_url"),
            "general_ner": CONFIG.get("a2a_general_url"),
            "medical_ner": CONFIG.get("a2a_medical_url"),
            "technical_ner": CONFIG.get("a2a_technical_url"),
            "pii_specialized": CONFIG.get("a2a_pii_specialized_url"),
            "legal_ner": CONFIG.get("a2a_legal_url"),
            "financial_ner": CONFIG.get("a2a_financial_url"),
        }
    })

@app.post("/entities")
async def detect_entities_debug(request: Request):
    """Debugging endpoint to show detected entities without masking."""
    if not app.debug:
        raise HTTPException(status_code=403, detail="Debug mode only")
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not data or "text" not in data:
        raise HTTPException(status_code=400, detail="No text provided")
    input_text = data["text"]
    try:
        all_entities = run_detectors(input_text)[0]
        merged_entities = merge_overlapping_entities(all_entities, input_text)
        response_entities = [{
            "text": input_text[e['start']:e['end']],
            "type": e['normalized_type'],
            "score": e['score'],
            "detector": e.get('detector', 'unknown'),
            "span": [e['start'], e['end']]
        } for e in merged_entities]
        response_entities.sort(key=lambda x: x['score'], reverse=True)
        return JSONResponse({
            "entities": response_entities,
            "total_count": len(response_entities),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "user": "rushilpatel21"
        })
    except Exception as e:
        logger.error(f"Error in /entities endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error detecting entities: {str(e)}")

@app.post("/internal_entities")
async def internal_entities_endpoint(request: Request):
    """
    Endpoint that uses only internal detection methods (Presidio, regex).
    Does not call any external MCP agents.
    """
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not data or "text" not in data:
        raise HTTPException(status_code=400, detail="No text provided")
    
    input_text = data["text"]
    
    try:
        # Get entities using only internal methods
        all_entities = run_internal_detectors(input_text)
        
        # Apply the same merging logic as the main system
        merged_entities = merge_overlapping_entities(all_entities, input_text)
        
        # Format response
        response_entities = [{
            "text": input_text[e['start']:e['end']],
            "type": e['normalized_type'],
            "score": e['score'],
            "detector": e.get('detector', 'unknown'),
            "span": [e['start'], e['end']]
        } for e in merged_entities]
        
        # Group by detector for clarity
        detector_counts = {}
        for e in response_entities:
            detector = e['detector']
            if detector not in detector_counts:
                detector_counts[detector] = 0
            detector_counts[detector] += 1
        
        response_entities.sort(key=lambda x: x['score'], reverse=True)
        return JSONResponse({
            "entities": response_entities,
            "total_count": len(response_entities),
            "detector_counts": detector_counts,
            "detectors_used": list(detector_counts.keys()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        })
    except Exception as e:
        logger.error(f"Error in /internal_entities endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error detecting entities: {str(e)}")

@app.get("/config")
async def get_config():
    """Return the current configuration for debugging purposes."""
    if not app.debug:
        raise HTTPException(status_code=403, detail="Debug mode only")
    safe_config = {k: v for k, v in CONFIG.items() if "key" not in k.lower() and "secret" not in k.lower() and "token" not in k.lower()}
    return JSONResponse({
        "config": safe_config,
        "core_services_loaded": list(CORE_SERVICES.keys()),
        "entity_types": sorted(list(set(filter(None, ENTITY_TYPE_MAPPING.values())))),
        "defaults": {
            "threshold": CONFIG.get("confidence_threshold"),
            "workers": CONFIG.get("max_workers"),
            "partial_mask_char": CONFIG.get("partial_mask_char")
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "user": "rushilpatel21"
    })

@app.get("/test")
async def test_connection():
    """Simple test endpoint to verify connectivity."""
    return JSONResponse({
        "status": "ok",
        "message": "Redactify MCP Server is running",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    })

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    print(f"--- Redactify MCP Server v2.1.1 (A2A/MCP MultiClass Arch) ---")
    print(f"Serving on http://0.0.0.0:{port}")
    print(f"Debug mode: {debug}")
    print(f"Worker threads: {CONFIG.get('max_workers')}")
    print(f"Core Services loaded: {list(CORE_SERVICES.keys())}")
    print(f"--- Agent/Service URLs ---")
    print(f"Classifier (MCP): {CONFIG.get('mcp_classifier_url')}")
    print(f"General NER (A2A): {CONFIG.get('a2a_general_url')}")
    print(f"Medical NER (A2A): {CONFIG.get('a2a_medical_url')}")
    print(f"Technical NER (A2A): {CONFIG.get('a2a_technical_url')}")
    print(f"PII Specialized (A2A): {CONFIG.get('a2a_pii_specialized_url')}")
    print(f"Legal NER (A2A): {CONFIG.get('a2a_legal_url')}")
    print(f"Financial NER (A2A): {CONFIG.get('a2a_financial_url')}")
    print(f"--------------------------")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=debug)