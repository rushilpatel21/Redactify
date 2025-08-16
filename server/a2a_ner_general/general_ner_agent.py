import os
import logging
import time
import uuid
import json
import re
import hashlib
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from transformers import pipeline
import numpy as np
from dotenv import load_dotenv

# --- Enhanced Logging Setup ---
load_dotenv()
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A2AGeneralNER")
logger.info(f"Logging initialized at {log_level} level")

# --- Model Configuration ---
MODEL_NAME = os.environ.get("A2A_GENERAL_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english")
AGENT_ID = "a2a_ner_general"

# --- Load model at module initialization ---
logger.info(f"[{AGENT_ID}] Starting model load: {MODEL_NAME}")
ner_pipeline = None
try:
    logger.info(f"[{AGENT_ID}] Step 1: Setting up model pipeline")
    start_time = time.time()
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    load_time = time.time() - start_time
    logger.info(f"[{AGENT_ID}] Step 2: Model loaded successfully in {load_time:.2f}s")
except Exception as e:
    logger.error(f"[{AGENT_ID}] CRITICAL ERROR: Failed to load model {MODEL_NAME}: {e}", exc_info=True)
    logger.error(f"[{AGENT_ID}] Model loading stack trace", stack_info=True)

# --- MCP Server Setup ---
logger.info(f"[{AGENT_ID}] Initializing FastMCP server")
mcp = FastMCP(
    name="GeneralNERAgent", 
    version="1.0.0",
    description="General-purpose Named Entity Recognition agent"
)
logger.info(f"[{AGENT_ID}] FastMCP initialized")

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Detect standard named entities in text using BERT-based NER.
    
    Specializes in:
    - PERSON: Names of individuals
    - ORG: Organizations, companies, institutions  
    - LOC: Locations, places, geographical entities
    - MISC: Miscellaneous entities
    
    Best used for: General documents, news articles, business communications
    Confidence range: 0.0-1.0 (typically 0.8+ for high-quality detections)
    """
    import asyncio
    import concurrent.futures
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: predict function called")
    
    text = inputs or ""
    if not text:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text provided, returning empty result")
        return {"entities": []}
    
    if ner_pipeline is None:
        logger.error(f"[{AGENT_ID}][{request_id}] NER pipeline not loaded. Cannot perform entity detection.")
        return {"entities": []}

    logger.info(f"[{AGENT_ID}][{request_id}] Processing text of length {len(text)}")
    logger.debug(f"[{AGENT_ID}][{request_id}] Text snippet (first 100 chars): {text[:100]}")
    
    def run_ner_pipeline(text_input):
        """Run NER pipeline in a separate thread to avoid blocking the event loop"""
        try:
            return ner_pipeline(text_input)
        except Exception as e:
            logger.error(f"[{AGENT_ID}][{request_id}] Error in NER pipeline: {e}")
            return []
    
    try:
        # Step 1: Entity detection - run in thread pool to avoid blocking
        logger.info(f"[{AGENT_ID}][{request_id}] Step 1: Starting entity detection in thread pool")
        start_time = time.time()
        
        # Use thread pool executor to run the blocking NER pipeline
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            raw_results = await loop.run_in_executor(executor, run_ner_pipeline, text)
        
        detection_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Step 2: Detection completed in {detection_time:.2f}s")
        
        # Step 2: Process results
        logger.info(f"[{AGENT_ID}][{request_id}] Step 3: Processing {len(raw_results)} entities")
        processed_results = []
        
        for i, item in enumerate(raw_results):
            logger.debug(f"[{AGENT_ID}][{request_id}] Processing entity {i+1}/{len(raw_results)}")
            
            # Type conversion and validation
            if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                item['score'] = float(item['score'])
                logger.debug(f"[{AGENT_ID}][{request_id}] Entity {i+1} score: {item['score']:.4f}")
            
            if 'start' in item: 
                item['start'] = int(item['start'])
            if 'end' in item:
                item['end'] = int(item['end'])
                
            # Logging the entity type and text
            if 'entity_group' in item and 'start' in item and 'end' in item:
                entity_text = text[item['start']:item['end']] if 0 <= item['start'] < len(text) and item['end'] <= len(text) else "INVALID_SPAN"
                logger.debug(f"[{AGENT_ID}][{request_id}] Entity {i+1}: {item.get('entity_group', 'UNKNOWN')} '{entity_text}' ({item['start']}:{item['end']})")
            
            # Add detector information
            item['detector'] = AGENT_ID
            processed_results.append(item)
        
        total_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Step 4: Processing completed, returning {len(processed_results)} entities in {total_time:.2f}s")
        
        # Group entities by type for better logging
        entity_types = {}
        for item in processed_results:
            entity_type = item.get('entity_group', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        logger.info(f"[{AGENT_ID}][{request_id}] Entity types found: {entity_types}")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function completed successfully")
        
        return {"entities": processed_results}
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error during NER detection: {e}", exc_info=True)
        logger.error(f"[{AGENT_ID}][{request_id}] Stack trace", stack_info=True)
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function failed")
        return {"entities": []}

@mcp.tool()
async def pseudonymize_entities(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Replace detected entities with consistent pseudonyms for anonymization.
    
    Creates hash-based pseudonyms that are:
    - Consistent: Same entity always gets same pseudonym
    - Secure: Original values cannot be recovered
    - Formatted: Maintains entity type context
    
    Parameters:
    - inputs: Original text
    - parameters: {
        "entities": List of detected entities with positions,
        "strategy": "hash" | "sequential" | "random",
        "preserve_format": boolean
      }
    
    Returns: Anonymized text with pseudonymized entities
    """
    import hashlib
    import re
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: pseudonymize_entities function called")
    
    text = inputs or ""
    params = parameters or {}
    entities = params.get("entities", [])
    strategy = params.get("strategy", "hash")
    preserve_format = params.get("preserve_format", True)
    
    if not text or not entities:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text or entities, returning original")
        return {"anonymized_text": text, "entities_processed": 0}
    
    try:
        logger.info(f"[{AGENT_ID}][{request_id}] Pseudonymizing {len(entities)} entities using {strategy} strategy")
        
        # Filter out sentiment entities and invalid entities BEFORE processing
        valid_entities = []
        sentiment_types = {'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'positive', 'negative', 'neutral'}
        
        for entity in entities:
            entity_type = entity.get('entity_group', '').upper()
            start = entity.get('start')
            end = entity.get('end')
            
            # Skip sentiment entities
            if entity_type in sentiment_types:
                continue
                
            # Skip invalid positions
            if start is None or end is None or start >= end:
                continue
                
            # Skip out-of-bounds entities
            if not (0 <= start < len(text) and start < end <= len(text)):
                continue
                
            valid_entities.append(entity)
        
        logger.info(f"[{AGENT_ID}][{request_id}] Pseudonymizing {len(valid_entities)} entities (filtered from {len(entities)}) using {strategy} strategy")
        
        if not valid_entities:
            logger.info(f"[{AGENT_ID}][{request_id}] No valid entities to pseudonymize after filtering")
            return {"anonymized_text": text, "entities_processed": 0}
        
        # Limit entities to prevent overwhelming the system
        MAX_ENTITIES = 50  # Reasonable limit to prevent corruption
        if len(valid_entities) > MAX_ENTITIES:
            logger.warning(f"[{AGENT_ID}][{request_id}] Too many entities ({len(valid_entities)}), limiting to {MAX_ENTITIES} highest confidence ones")
            # Sort by confidence and take top entities
            valid_entities = sorted(valid_entities, key=lambda x: x.get('score', 0), reverse=True)[:MAX_ENTITIES]
        
        # Sort entities by start position (forward order, process left to right)
        sorted_entities = sorted(valid_entities, key=lambda x: x.get('start', 0))
        
        # ROBUST overlap detection - create a comprehensive conflict-free entity list
        final_entities = []
        position_map = set()  # Track all occupied positions
        
        for entity in sorted_entities:
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            
            # Check if any position in this entity range is already occupied
            entity_positions = set(range(start, end))
            if entity_positions & position_map:  # If there's any intersection
                logger.warning(f"[{AGENT_ID}][{request_id}] Skipping overlapping entity at {start}-{end} (conflicts with existing positions)")
                continue
            
            # This entity doesn't conflict, add it and mark positions as occupied
            final_entities.append(entity)
            position_map.update(entity_positions)
        
        logger.info(f"[{AGENT_ID}][{request_id}] Final entity list: {len(final_entities)} entities after overlap removal")
        
        if not final_entities:
            logger.info(f"[{AGENT_ID}][{request_id}] No entities remaining after overlap detection")
            return {"anonymized_text": text, "entities_processed": 0}
        
        # SAFE text replacement using string segments approach
        # Sort by position DESC to process from end to start (preserves earlier positions)
        replacement_entities = sorted(final_entities, key=lambda x: x.get('start', 0), reverse=True)
        
        anonymized_text = text
        entities_processed = 0
        pseudonym_map = {}
        
        for entity in replacement_entities:
            start = entity.get('start')
            end = entity.get('end')
            entity_type = entity.get('entity_group', 'UNKNOWN').upper()
            
            # Final bounds check with original text
            if not (0 <= start < len(text) and start < end <= len(text)):
                logger.warning(f"[{AGENT_ID}][{request_id}] Entity bounds {start}-{end} invalid for text length {len(text)}, skipping")
                continue
            
            # Extract original text (from original, not modified text)
            actual_text = text[start:end]
            
            # Skip if empty or whitespace only
            if not actual_text.strip():
                continue
            
            # Generate pseudonym
            if actual_text in pseudonym_map:
                pseudonym = pseudonym_map[actual_text]
            else:
                # Map entity types to more readable names
                readable_type = _get_readable_entity_type(entity_type)
                
                if strategy == "hash":
                    # Create hash-based pseudonym with shorter, cleaner format
                    hash_obj = hashlib.md5(actual_text.encode()).hexdigest()[:4]
                    pseudonym = f"[{readable_type}_{hash_obj.upper()}]"
                elif strategy == "sequential":
                    # Sequential numbering per entity type
                    type_count = len([p for p in pseudonym_map.values() if readable_type in p]) + 1
                    pseudonym = f"[{readable_type}_{type_count:02d}]"
                else:  # random
                    import random
                    rand_id = random.randint(1000, 9999)
                    pseudonym = f"[{readable_type}_{rand_id}]"
                
                pseudonym_map[actual_text] = pseudonym
            
            # Safe replacement: rebuild string with segments
            anonymized_text = anonymized_text[:start] + pseudonym + anonymized_text[end:]
            entities_processed += 1
            
            logger.debug(f"[{AGENT_ID}][{request_id}] Replaced '{actual_text}' with '{pseudonym}' at position {start}-{end}")
        
        logger.info(f"[{AGENT_ID}][{request_id}] Successfully pseudonymized {entities_processed} entities")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: pseudonymize_entities function completed")
        
        # Clean up any formatting issues
        if entities_processed > 0:
            # Remove extra spaces around punctuation
            anonymized_text = re.sub(r'\s+', ' ', anonymized_text)
            anonymized_text = re.sub(r'\s+([,.!?;:])', r'\1', anonymized_text)
        
        return {
            "anonymized_text": anonymized_text,
            "entities_processed": entities_processed,
            "pseudonym_map": pseudonym_map,
            "strategy_used": strategy,
            "tool_used": "pseudonymize_entities"
        }
        
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error in pseudonymization: {e}", exc_info=True)
        return {
            "anonymized_text": text,
            "entities_processed": 0,
            "error": str(e),
            "tool_used": "pseudonymize_entities"
        }

@mcp.tool()
async def mask_entities(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Partially mask detected entities while preserving format and context.
    
    Masking strategies:
    - Names: Show first letter + asterisks (J*** S***)
    - Numbers: Show first/last digits (***-**-1234)
    - Emails: Show domain (@company.com)
    - Phones: Show area code (555-***-****)
    
    Parameters:
    - inputs: Original text
    - parameters: {
        "entities": List of detected entities,
        "mask_char": Character to use for masking (default: "*"),
        "preserve_length": boolean,
        "show_partial": boolean
      }
    
    Returns: Text with entities partially masked
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: mask_entities function called")
    
    text = inputs or ""
    params = parameters or {}
    entities = params.get("entities", [])
    mask_char = params.get("mask_char", "*")
    preserve_length = params.get("preserve_length", True)
    show_partial = params.get("show_partial", True)
    
    if not text or not entities:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text or entities, returning original")
        return {"anonymized_text": text, "entities_processed": 0}
    
    logger.info(f"[{AGENT_ID}][{request_id}] Masking {len(entities)} entities")
    
    try:
        # Sort entities by start position (reverse order)
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
        
        anonymized_text = text
        entities_processed = 0
        
        for entity in sorted_entities:
            start = entity.get('start')
            end = entity.get('end')
            entity_type = entity.get('entity_group', 'UNKNOWN').upper()
            
            if start is None or end is None or start >= end:
                continue
            
            if not (0 <= start < len(text) and start < end <= len(text)):
                continue
            
            original_text = text[start:end]
            masked_text = original_text
            
            # Apply entity-specific masking rules
            if entity_type in ['PERSON', 'PER']:
                # Names: Show first letter of each word
                if show_partial and len(original_text) > 2:
                    words = original_text.split()
                    masked_words = []
                    for word in words:
                        if len(word) > 1:
                            masked_word = word[0] + mask_char * (len(word) - 1)
                        else:
                            masked_word = mask_char
                        masked_words.append(masked_word)
                    masked_text = " ".join(masked_words)
                else:
                    masked_text = mask_char * len(original_text)
            
            elif entity_type in ['EMAIL', 'EMAIL_ADDRESS']:
                # Emails: Show domain part
                if show_partial and '@' in original_text:
                    local, domain = original_text.split('@', 1)
                    masked_local = mask_char * len(local)
                    masked_text = f"{masked_local}@{domain}"
                else:
                    masked_text = mask_char * len(original_text)
            
            elif entity_type in ['PHONE', 'PHONE_NUMBER']:
                # Phone: Show area code or last 4 digits
                if show_partial and len(original_text) >= 7:
                    if original_text.startswith('+') or '-' in original_text:
                        # Keep format, mask middle
                        masked_text = re.sub(r'\d', mask_char, original_text[:-4]) + original_text[-4:]
                    else:
                        masked_text = original_text[:3] + mask_char * (len(original_text) - 7) + original_text[-4:]
                else:
                    masked_text = mask_char * len(original_text)
            
            elif entity_type in ['SSN', 'SOCIAL_SECURITY']:
                # SSN: Show last 4 digits
                if show_partial and len(original_text) >= 4:
                    masked_text = mask_char * (len(original_text) - 4) + original_text[-4:]
                else:
                    masked_text = mask_char * len(original_text)
            
            else:
                # Default: Mask everything or show first character
                if show_partial and len(original_text) > 2:
                    masked_text = original_text[0] + mask_char * (len(original_text) - 1)
                else:
                    masked_text = mask_char * len(original_text)
            
            # Replace in text
            anonymized_text = anonymized_text[:start] + masked_text + anonymized_text[end:]
            entities_processed += 1
            
            logger.debug(f"[{AGENT_ID}][{request_id}] Masked '{original_text}' -> '{masked_text}'")
        
        logger.info(f"[{AGENT_ID}][{request_id}] Successfully masked {entities_processed} entities")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: mask_entities function completed")
        
        return {
            "anonymized_text": anonymized_text,
            "entities_processed": entities_processed,
            "mask_character": mask_char,
            "tool_used": "mask_entities"
        }
        
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error in masking: {e}", exc_info=True)
        return {
            "anonymized_text": text,
            "entities_processed": 0,
            "error": str(e),
            "tool_used": "mask_entities"
        }

@mcp.tool()
async def redact_entities(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Completely remove or redact detected entities from text.
    
    Redaction strategies:
    - Remove: Complete removal with space cleanup
    - Replace: Replace with [REDACTED] or custom placeholder
    - Blackout: Replace with █ characters
    
    Parameters:
    - inputs: Original text
    - parameters: {
        "entities": List of detected entities,
        "redaction_style": "remove" | "replace" | "blackout",
        "placeholder": Custom placeholder text,
        "clean_whitespace": boolean
      }
    
    Returns: Text with entities completely redacted
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: redact_entities function called")
    
    text = inputs or ""
    params = parameters or {}
    entities = params.get("entities", [])
    redaction_style = params.get("redaction_style", "replace")
    placeholder = params.get("placeholder", "[REDACTED]")
    clean_whitespace = params.get("clean_whitespace", True)
    
    if not text or not entities:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text or entities, returning original")
        return {"anonymized_text": text, "entities_processed": 0}
    
    # Filter out sentiment entities and invalid entities
    valid_entities = []
    sentiment_types = {'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'positive', 'negative', 'neutral'}
    
    for entity in entities:
        entity_type = entity.get('entity_group', '').upper()
        start = entity.get('start')
        end = entity.get('end')
        
        # Skip sentiment entities
        if entity_type in sentiment_types:
            continue
            
        # Skip invalid positions
        if start is None or end is None or start >= end:
            continue
            
        # Skip out-of-bounds entities
        if not (0 <= start < len(text) and start < end <= len(text)):
            continue
            
        valid_entities.append(entity)
    
    logger.info(f"[{AGENT_ID}][{request_id}] Redacting {len(valid_entities)} entities (filtered from {len(entities)}) using {redaction_style} style")
    
    if not valid_entities:
        logger.info(f"[{AGENT_ID}][{request_id}] No valid entities to redact after filtering")
        return {"anonymized_text": text, "entities_processed": 0}
    
    try:
        # Sort entities by start position (reverse order to maintain indices)
        sorted_entities = sorted(valid_entities, key=lambda x: x.get('start', 0), reverse=True)
        
        anonymized_text = text
        entities_processed = 0
        
        for entity in sorted_entities:
            start = entity.get('start')
            end = entity.get('end')
            entity_type = _get_readable_entity_type(entity.get('entity_group', 'UNKNOWN'))
            
            # Double-check bounds after previous replacements
            if start >= len(anonymized_text) or end > len(anonymized_text):
                logger.warning(f"[{AGENT_ID}][{request_id}] Entity bounds out of range after previous redactions, skipping")
                continue
            
            original_text = text[start:end]  # Use original text for logging
            
            # Apply redaction based on style
            if redaction_style == "remove":
                replacement = ""
            elif redaction_style == "blackout":
                replacement = "█" * len(original_text)
            elif redaction_style == "replace":
                if placeholder == "[REDACTED]":
                    replacement = f"[{entity_type}_REDACTED]"
                else:
                    replacement = placeholder
            else:
                replacement = "[REDACTED]"
            
            # Replace in text (using current anonymized_text bounds)
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            entities_processed += 1
            
            logger.debug(f"[{AGENT_ID}][{request_id}] Redacted '{original_text}' -> '{replacement}'")
        
        # Clean up whitespace if requested
        if clean_whitespace and redaction_style == "remove":
            # Remove extra spaces, but preserve sentence structure
            anonymized_text = re.sub(r'\s+', ' ', anonymized_text)
            anonymized_text = re.sub(r'\s+([,.!?;:])', r'\1', anonymized_text)
            anonymized_text = anonymized_text.strip()
        
        logger.info(f"[{AGENT_ID}][{request_id}] Successfully redacted {entities_processed} entities")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: redact_entities function completed")
        
        return {
            "anonymized_text": anonymized_text,
            "entities_processed": entities_processed,
            "redaction_style": redaction_style,
            "tool_used": "redact_entities"
        }
        
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error in redaction: {e}", exc_info=True)
        return {
            "anonymized_text": text,
            "entities_processed": 0,
            "error": str(e),
            "tool_used": "redact_entities"
        }

@mcp.tool()
async def merge_overlapping_entities(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Intelligently merge overlapping entity detections from multiple models.
    
    Handles cases where:
    - Multiple models detect the same entity with different boundaries
    - Entities overlap partially (e.g., "John Smith" vs "Smith")
    - Different confidence scores for same entity
    
    Parameters:
    - inputs: Original text (for validation)
    - parameters: {
        "entities": List of entities to merge,
        "merge_strategy": "highest_confidence" | "longest_span" | "most_specific",
        "overlap_threshold": 0.5 (minimum overlap ratio to consider merging)
      }
    
    Returns: Merged list of entities without overlaps
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: merge_overlapping_entities function called")
    
    text = inputs or ""
    params = parameters or {}
    entities = params.get("entities", [])
    merge_strategy = params.get("merge_strategy", "highest_confidence")
    overlap_threshold = params.get("overlap_threshold", 0.5)
    
    if not entities:
        logger.warning(f"[{AGENT_ID}][{request_id}] No entities provided, returning empty list")
        return {"entities": [], "merges_performed": 0}
    
    # Filter out sentiment entities and invalid entities before merging
    valid_entities = []
    sentiment_types = {'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'positive', 'negative', 'neutral'}
    
    for entity in entities:
        entity_type = entity.get('entity_group', '').upper()
        start = entity.get('start')
        end = entity.get('end')
        
        # Skip sentiment entities
        if entity_type in sentiment_types:
            continue
            
        # Skip invalid positions
        if start is None or end is None or start >= end:
            continue
            
        # Skip out-of-bounds entities
        if text and not (0 <= start < len(text) and start < end <= len(text)):
            continue
            
        valid_entities.append(entity)
    
    logger.info(f"[{AGENT_ID}][{request_id}] Merging {len(valid_entities)} entities (filtered from {len(entities)}) using {merge_strategy} strategy")
    
    if not valid_entities:
        logger.info(f"[{AGENT_ID}][{request_id}] No valid entities to merge after filtering")
        return {"entities": [], "merges_performed": 0}
    
    try:
        # Sort entities by start position
        sorted_entities = sorted(valid_entities, key=lambda x: x.get('start', 0))
        merged_entities = []
        merges_performed = 0
        
        i = 0
        while i < len(sorted_entities):
            current_entity = sorted_entities[i]
            current_start = current_entity.get('start', 0)
            current_end = current_entity.get('end', 0)
            
            # Find all overlapping entities
            overlapping = [current_entity]
            j = i + 1
            
            while j < len(sorted_entities):
                next_entity = sorted_entities[j]
                next_start = next_entity.get('start', 0)
                next_end = next_entity.get('end', 0)
                
                # Check if entities overlap
                overlap_start = max(current_start, next_start)
                overlap_end = min(current_end, next_end)
                
                if overlap_start < overlap_end:
                    # Calculate overlap ratio
                    overlap_length = overlap_end - overlap_start
                    current_length = current_end - current_start
                    next_length = next_end - next_start
                    
                    overlap_ratio = overlap_length / min(current_length, next_length)
                    
                    if overlap_ratio >= overlap_threshold:
                        overlapping.append(next_entity)
                        # Update current bounds to include this entity
                        current_end = max(current_end, next_end)
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Merge overlapping entities
            if len(overlapping) > 1:
                merged_entity = _merge_entity_group(overlapping, merge_strategy, text)
                merged_entities.append(merged_entity)
                merges_performed += len(overlapping) - 1
                logger.debug(f"[{AGENT_ID}][{request_id}] Merged {len(overlapping)} overlapping entities")
            else:
                merged_entities.append(current_entity)
            
            # Move to next non-overlapping entity
            i += len(overlapping)
        
        logger.info(f"[{AGENT_ID}][{request_id}] Completed merging: {len(valid_entities)} -> {len(merged_entities)} entities")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: merge_overlapping_entities function completed")
        
        return {
            "entities": merged_entities,
            "original_count": len(entities),
            "merged_count": len(merged_entities),
            "merges_performed": merges_performed,
            "merge_strategy": merge_strategy,
            "tool_used": "merge_overlapping_entities"
        }
        
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error in entity merging: {e}", exc_info=True)
        return {
            "entities": valid_entities,  # Return filtered entities on error
            "original_count": len(entities),
            "merged_count": len(valid_entities),
            "merges_performed": 0,
            "error": str(e),
            "tool_used": "merge_overlapping_entities"
        }

def _get_readable_entity_type(entity_type: str) -> str:
    """Convert entity type to more readable format"""
    type_mapping = {
        'PER': 'PERSON',
        'PERSON': 'PERSON',
        'ORG': 'ORG',
        'ORGANIZATION': 'ORG', 
        'LOC': 'LOCATION',
        'LOCATION': 'LOCATION',
        'MISC': 'MISC',
        'EMAIL': 'EMAIL',
        'EMAIL_ADDRESS': 'EMAIL',
        'PHONE': 'PHONE',
        'PHONE_NUMBER': 'PHONE',
        'SSN': 'SSN',
        'CREDIT_CARD': 'CARD',
        'ID_NUM': 'ID',
        'NAME_STUDENT': 'NAME',
        'PATIENT': 'PATIENT',
        'STAFF': 'STAFF',
        'HOSP': 'HOSPITAL',
        'PATORG': 'ORG',
        # Medical entities
        'DOCTOR': 'DOCTOR',
        'AGE': 'AGE',
        'DATE': 'DATE',
        # Financial entities  
        'FINANCIAL': 'FINANCIAL',
        # Legal entities
        'LABEL_0': 'ENTITY',
        'LABEL_1': 'ENTITY',
        # Skip sentiment types - these should be filtered out before this function
        'POSITIVE': 'WORD',
        'NEGATIVE': 'WORD', 
        'NEUTRAL': 'WORD'
    }
    
    cleaned_type = type_mapping.get(entity_type.upper(), entity_type.upper())
    
    # Additional cleanup for any remaining sentiment or weird types
    if cleaned_type in ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'WORD']:
        return 'ENTITY'
    
    return cleaned_type

def _merge_entity_group(entities: List[Dict], strategy: str, text: str) -> Dict:
    """Helper method to merge a group of overlapping entities"""
    if not entities:
        return {}
    
    if len(entities) == 1:
        return entities[0]
    
    # Determine merged boundaries
    min_start = min(e.get('start', 0) for e in entities)
    max_end = max(e.get('end', 0) for e in entities)
    
    # Select best entity based on strategy
    if strategy == "highest_confidence":
        best_entity = max(entities, key=lambda x: x.get('score', 0))
    elif strategy == "longest_span":
        best_entity = max(entities, key=lambda x: x.get('end', 0) - x.get('start', 0))
    elif strategy == "most_specific":
        # Prefer more specific entity types (medical > general, etc.)
        type_priority = {
            'PERSON': 5, 'PER': 5,
            'ORGANIZATION': 4, 'ORG': 4,
            'LOCATION': 3, 'LOC': 3,
            'MISC': 2,
            'O': 1
        }
        best_entity = max(entities, key=lambda x: type_priority.get(x.get('entity_group', ''), 0))
    else:
        best_entity = entities[0]
    
    # Create merged entity
    merged = best_entity.copy()
    merged['start'] = min_start
    merged['end'] = max_end
    
    # Update word field with actual text
    if text and 0 <= min_start < len(text) and min_start < max_end <= len(text):
        merged['word'] = text[min_start:max_end]
    
    # Combine detector information
    detectors = set()
    for entity in entities:
        if 'detector' in entity:
            detectors.add(entity['detector'])
    merged['detector'] = ','.join(sorted(detectors))
    
    # Average confidence scores
    scores = [e.get('score', 0) for e in entities if 'score' in e]
    if scores:
        merged['score'] = sum(scores) / len(scores)
    
    merged['merged_from'] = len(entities)
    
    return merged

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the NER agent service."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: health_check function called")
    
    status = {
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME,
        "available_tools": [
            "predict", "pseudonymize_entities", "mask_entities", 
            "redact_entities", "merge_overlapping_entities", "health_check"
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }
    
    logger.info(f"[{AGENT_ID}][{request_id}] Health check result: {status}")
    logger.info(f"[{AGENT_ID}][{request_id}] EXIT: health_check function completed")
    return status

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    
    # Service configuration
    port = int(os.environ.get("A2A_GENERAL_PORT", 3001))
    logger.info(f"[{AGENT_ID}] Starting server on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="General NER MCP Server")
    
    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        """MCP JSON-RPC endpoint that processes requests and forwards to appropriate tools"""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{AGENT_ID}][{request_id}] Received MCP request")
        
        try:
            # Parse request JSON
            logger.info(f"[{AGENT_ID}][{request_id}] Parsing request JSON")
            data = await request.json()
            logger.debug(f"[{AGENT_ID}][{request_id}] Request data: {json.dumps(data, indent=2)}")
            
            # Check if this is a JSON-RPC request
            if "jsonrpc" in data and "method" in data:
                method = data["method"]
                params = data.get("params", {})
                json_rpc_id = data.get("id")
                
                logger.info(f"[{AGENT_ID}][{request_id}] Processing JSON-RPC method: {method}")
                
                # Call the appropriate tool
                if method == "predict" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling predict with text length: {len(params['inputs'])}")
                    start_time = time.time()
                    result = await predict(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] predict completed in {duration:.2f}s, found {len(result.get('entities', []))} entities")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    logger.info(f"[{AGENT_ID}][{request_id}] Returning response")
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "pseudonymize_entities" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling pseudonymize_entities")
                    start_time = time.time()
                    result = await pseudonymize_entities(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] pseudonymize_entities completed in {duration:.2f}s")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "mask_entities" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling mask_entities")
                    start_time = time.time()
                    result = await mask_entities(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] mask_entities completed in {duration:.2f}s")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "redact_entities" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling redact_entities")
                    start_time = time.time()
                    result = await redact_entities(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] redact_entities completed in {duration:.2f}s")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "merge_overlapping_entities" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling merge_overlapping_entities")
                    start_time = time.time()
                    result = await merge_overlapping_entities(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] merge_overlapping_entities completed in {duration:.2f}s")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "health_check":
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling health_check")
                    result = await health_check()
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    logger.info(f"[{AGENT_ID}][{request_id}] Returning health check response")
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                else:
                    # Method not found
                    logger.warning(f"[{AGENT_ID}][{request_id}] Method not found: {method}")
                    return Response(
                        content=json.dumps({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method {method} not found"
                            },
                            "id": json_rpc_id
                        }),
                        media_type="application/json"
                    )
            else:
                # Not a JSON-RPC request
                logger.warning(f"[{AGENT_ID}][{request_id}] Invalid JSON-RPC request")
                return Response(
                    content=json.dumps({
                        "error": "Invalid JSON-RPC request"
                    }),
                    status_code=400,
                    media_type="application/json"
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"[{AGENT_ID}][{request_id}] JSON decode error: {e}")
            return Response(
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    },
                    "id": None
                }),
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"[{AGENT_ID}][{request_id}] Error processing MCP request: {e}", exc_info=True)
            return Response(
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": None
                }),
                media_type="application/json"
            )
    
    # Add /health endpoint for basic monitoring
    @app.get("/health")
    async def health():
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{AGENT_ID}][{request_id}] Received health check request")
        status = await health_check()
        logger.info(f"[{AGENT_ID}][{request_id}] Returning health status: {status['status']}")
        return status
    
    # Start the server
    logger.info(f"[{AGENT_ID}] Server initialization complete")
    logger.info(f"[{AGENT_ID}] Starting uvicorn server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)