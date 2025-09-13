"""
Detection Engine for Redactify MCP Server

This module contains the core PII detection logic, refactored from the original
server.py to work as an internal service rather than a FastAPI server.
"""

import os
import logging
import re
import json
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from presidio_analyzer import AnalyzerEngine
if TYPE_CHECKING:
    from mcp_client import MCPClientManager

logger = logging.getLogger("DetectionEngine")

class DetectionEngine:
    """
    Core PII detection engine that combines multiple detection methods:
    - Presidio Analyzer (built-in PII detection)
    - Regex patterns (custom patterns)
    - Contextual detection (domain-aware)
    - ML-based NER models (via ModelManager)
    """
    
    def __init__(self):
        self.mcp_client_manager: Optional['MCPClientManager'] = None
        self.presidio_analyzer = None
        self.regex_patterns = []
        self.config = {}
        self.blocklist = set()
        self.common_name_words = set()
        self.pseudonymize_types = set()
        self.entity_type_mapping = {}
        
        # Initialize components
        self._load_configuration()
        self._load_presidio()
        self._load_regex_patterns()
        
        logger.info("DetectionEngine initialized successfully")
    
    def set_mcp_client_manager(self, mcp_client_manager: 'MCPClientManager'):
        """Set the MCP client manager for model communication"""
        self.mcp_client_manager = mcp_client_manager
        logger.info("MCP client manager configured for DetectionEngine")
    
    def _load_configuration(self):
        """Load configuration from JSON files and environment variables"""
        base_dir = os.path.dirname(__file__)
        
        # Load static configuration
        config_static = self._load_json_config('config_static.json', {})
        
        # Merge with environment variables
        self.config = {
            **config_static,
            "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5)),
            "max_workers": int(os.environ.get("MAX_WORKERS", 8)),
            "enable_medical_pii": os.environ.get("ENABLE_MEDICAL_PII", "True").lower() == "true",
            "enable_technical_ner": os.environ.get("ENABLE_TECHNICAL_NER", "True").lower() == "true",
            "enable_pii_specialized": os.environ.get("ENABLE_PII_SPECIALIZED", "True").lower() == "true",
            "enable_legal_ner": os.environ.get("ENABLE_LEGAL_NER", "True").lower() == "true",
            "enable_financial_ner": os.environ.get("ENABLE_FINANCIAL_NER", "True").lower() == "true",
            "context_window": 40,
            "entity_confidence_threshold": 0.3,  # Increased from 0.1 to reduce noise
            "enable_context_detection": True,
            "enable_fallback_name_detector": True,
            # Model-specific confidence thresholds
            "legal_model_threshold": 0.8,  # Higher threshold for noisy legal model
            "financial_model_threshold": 0.7,  # Higher threshold for financial model
        }
        
        # Load additional configuration files
        blocklist_data = self._load_json_config('blocklist.json', [])
        common_names_data = self._load_json_config('common_name_words.json', [])
        pseudonymize_data = self._load_json_config('pseudonymize_types.json', [])
        self.entity_type_mapping = self._load_json_config('entity_type_mapping.json', {})
        
        self.blocklist = set(blocklist_data)
        self.common_name_words = set(common_names_data)
        self.pseudonymize_types = set(pseudonymize_data)
        
        # Combine blocklist with common names
        self.blocklist.update(self.common_name_words)
        
        logger.info(f"Configuration loaded: {len(self.blocklist)} blocked terms, "
                   f"{len(self.common_name_words)} common names")
    
    def _load_json_config(self, filename: str, default_value: Any) -> Any:
        """Load configuration from a JSON file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded configuration from {filename}")
                return data
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {filepath}. Using default.")
            return default_value
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath}: {e}. Using default.")
            return default_value
        except Exception as e:
            logger.error(f"Unexpected error loading {filepath}: {e}")
            return default_value
    
    def _load_presidio(self):
        """Initialize Presidio Analyzer"""
        try:
            self.presidio_analyzer = AnalyzerEngine()
            logger.info("Presidio Analyzer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Presidio Analyzer: {e}", exc_info=True)
            raise RuntimeError("Failed to load core Presidio service")
    
    def _load_regex_patterns(self):
        """Load and compile regex patterns from JSON file"""
        filepath = os.path.join(os.path.dirname(__file__), 'regex_patterns.json')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            self.regex_patterns = []
            for p_def in patterns_data:
                pattern_str = p_def.get('pattern')
                pattern_type = p_def.get('type', 'N/A')
                
                if not pattern_str:
                    logger.warning(f"Skipping pattern type '{pattern_type}' - missing pattern")
                    continue
                
                try:
                    p_def['compiled_pattern'] = re.compile(pattern_str, re.IGNORECASE)
                    self.regex_patterns.append(p_def)
                except re.error as e:
                    logger.error(f"Failed to compile regex for '{pattern_type}': {e}")
            
            logger.info(f"Loaded {len(self.regex_patterns)} regex patterns")
            
        except FileNotFoundError:
            logger.error(f"Regex patterns file not found: {filepath}")
            self.regex_patterns = []
        except Exception as e:
            logger.error(f"Error loading regex patterns: {e}")
            self.regex_patterns = []
    
    async def detect_entities(self, text: str, domains: Optional[List[str]] = None) -> Tuple[List[Dict], List[str]]:
        """
        Main entity detection method that combines all detection approaches.
        
        Args:
            text: Text to analyze
            domains: Optional list of domains to focus on (e.g., ['medical', 'technical'])
            
        Returns:
            Tuple of (entities, domains_used)
        """
        results = await self.detect_entities_batch([text], domains)
        return results[0]
    
    async def detect_entities_batch(self, texts: List[str], domains: Optional[List[str]] = None) -> List[Tuple[List[Dict], List[str]]]:
        """
        Batch entity detection for multiple texts with improved concurrency.
        
        Args:
            texts: List of texts to analyze
            domains: Optional list of domains to focus on (e.g., ['medical', 'technical'])
            
        Returns:
            List of tuples (entities, domains_used) for each text
        """
        if not texts:
            return []
        
        # For single text, use the optimized single detection
        if len(texts) == 1:
            result = await self._detect_entities_single(texts[0], domains)
            return [result]
        
        # For multiple texts, process them concurrently
        start_time = time.time()
        
        # Create tasks for each text
        detection_tasks = [
            self._detect_entities_single(text, domains) 
            for text in texts
        ]
        
        # Run all detection tasks concurrently
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing text {i}: {result}")
                processed_results.append(([], ["general"]))  # Fallback
            else:
                processed_results.append(result)
        
        duration = time.time() - start_time
        logger.info(f"Batch detection completed for {len(texts)} texts in {duration:.2f}s")
        
        return processed_results
    
    async def _detect_entities_single(self, text: str, domains: Optional[List[str]] = None) -> Tuple[List[Dict], List[str]]:
        if not text:
            return [], []
        
        start_time = time.time()
        all_entities = []
        
        # Step 1: Classify text to determine which models to use
        if domains is None:
            domains = await self._classify_text(text)
        
        logger.info(f"Processing text with domains: {domains}")
        
        # Step 2: Run all detectors concurrently
        detection_tasks = []
        
        # Internal detectors (run in thread pool)
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            internal_futures = [
                executor.submit(self._get_presidio_entities, text),
                executor.submit(self._get_regex_entities, text),
                executor.submit(self._get_contextual_entities, text)
            ]
            
            # ML model detectors (async tasks)
            model_tasks = []
            
            # Always use general model
            model_tasks.append(self._get_model_entities(text, "general"))
            
            # Add specialized models based on domains and configuration
            if "medical" in domains and self.config["enable_medical_pii"]:
                model_tasks.append(self._get_model_entities(text, "medical"))
            if "technical" in domains and self.config["enable_technical_ner"]:
                model_tasks.append(self._get_model_entities(text, "technical"))
            if "legal" in domains and self.config["enable_legal_ner"]:
                model_tasks.append(self._get_model_entities(text, "legal"))
            if "financial" in domains and self.config["enable_financial_ner"]:
                model_tasks.append(self._get_model_entities(text, "financial"))
            if self.config["enable_pii_specialized"]:
                model_tasks.append(self._get_model_entities(text, "pii_specialized"))
            
            # Run all model tasks concurrently
            if model_tasks:
                model_results = await asyncio.gather(*model_tasks, return_exceptions=True)
                
                # Process model results
                for result in model_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in model detector: {result}")
                    elif result:
                        all_entities.extend(result)
            
            # Wait for internal detectors
            for future in as_completed(internal_futures):
                try:
                    entities = future.result()
                    if entities:
                        all_entities.extend(entities)
                except Exception as e:
                    logger.error(f"Error in internal detector: {e}")
        
        # Step 3: Add fallback name detection if needed
        if self.config.get("enable_fallback_name_detector", True):
            fallback_entities = self._get_fallback_name_entities(text, all_entities)
            all_entities.extend(fallback_entities)
        
        # Step 4: Post-process entities
        processed_entities = self._post_process_entities(all_entities, text)
        
        duration = time.time() - start_time
        logger.info(f"Detection completed in {duration:.2f}s. Found {len(processed_entities)} entities")
        
        return processed_entities, domains
    
    async def _classify_text(self, text: str) -> List[str]:
        """Classify text to determine relevant domains using MCP classifier"""
        try:
            # For now, use simple heuristic classification
            
            domains = ["general"]
            
            # Simple keyword-based classification
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['medical', 'patient', 'doctor', 'hospital', 'diagnosis']):
                domains.append("medical")
            
            # Disabled domains for testing:
            # if any(word in text_lower for word in ['agreement', 'contract', 'legal', 'court', 'law']):
            #     domains.append("legal")
            # 
            # if any(word in text_lower for word in ['financial', 'bank', 'credit', 'loan', 'investment']):
            #     domains.append("financial")
            # 
            # if any(word in text_lower for word in ['technical', 'software', 'code', 'api', 'system']):
            #     domains.append("technical")
            
            logger.debug(f"Classified text domains: {domains}")
            return domains
            
            # Use the text classifier logic adapted for Gemini
            max_length = 4000
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            categories = ["medical", "general"]  # Only enabled categories
            
            prompt = f"""
            You are a document classifier that helps route text to specialized PII detection models.
            
            Analyze the following text and determine which categories it belongs to from: {', '.join(categories)}.
            Multiple categories can apply if the text contains mixed content.
            Always include "general" as a fallback category if no specific category applies.
            
            Return ONLY a JSON array of category names, nothing else.
            
            Text to classify:
            {truncated_text}
            """
            
            # Generate response using Gemini
            response = gemini_model.generate_content(prompt)
            content = response.text.strip()
            
            # Parse JSON response
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                
                classifications = json.loads(content)
                if not isinstance(classifications, list):
                    classifications = ["general"]
                
                # Validate classifications
                valid_classifications = [c.lower() for c in classifications if c.lower() in categories]
                
                if not valid_classifications:
                    valid_classifications = ["general"]
                
                logger.info(f"Gemini classified text as: {valid_classifications}")
                return sorted(list(set(valid_classifications)))
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Gemini response: {content}")
                return ["general"]
                
        except Exception as e:
            logger.error(f"Error in Gemini text classification: {e}")
            return ["general"]
    
    async def _get_model_entities(self, text: str, model_name: str) -> List[Dict]:
        """Get entities from a specific MCP model server"""
        try:
            if not self.mcp_client_manager:
                logger.warning("MCP client manager not available")
                return []
            
            # Get MCP client for the model
            try:
                client = self.mcp_client_manager.get_client(model_name)
            except Exception as e:
                logger.warning(f"MCP client {model_name} not available: {e}")
                return []
            
            # Make prediction request to MCP server
            result = await client.predict(text)
            
            # Extract entities from MCP response
            raw_entities = result.get('entities', [])
            
            # Apply model-specific filtering
            filtered_entities = []
            for entity in raw_entities:
                # Apply model-specific confidence thresholds
                min_confidence = self._get_model_confidence_threshold(model_name)
                if entity.get('score', 0) < min_confidence:
                    continue
                
                # Filter out generic labels for specific models
                entity_type = entity.get('entity_group', '').upper()
                if model_name in ['legal', 'financial'] and self._is_generic_label(entity_type):
                    continue
                
                # Add detector information
                entity['detector'] = f'mcp_{model_name}'
                
                filtered_entities.append(entity)
            
            logger.debug(f"MCP model {model_name} found {len(raw_entities)} raw entities, {len(filtered_entities)} after filtering")
            return filtered_entities
            
        except Exception as e:
            logger.error(f"Error getting entities from MCP model {model_name}: {e}")
            return []
    
    def _get_model_confidence_threshold(self, model_name: str) -> float:
        """Get confidence threshold for specific model"""
        model_thresholds = {
            'legal': self.config.get('legal_model_threshold', 0.8),
            'financial': self.config.get('financial_model_threshold', 0.7),
            'general': 0.5,
            'medical': 0.6,
            'technical': 0.6,
            'pii_specialized': 0.5
        }
        return model_thresholds.get(model_name, self.config.get('entity_confidence_threshold', 0.3))
    
    def _is_generic_label(self, entity_type: str) -> bool:
        """Check if entity type is a generic/meaningless label"""
        generic_labels = {
            'LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4',
            'LABEL_5', 'LABEL_6', 'LABEL_7', 'LABEL_8', 'LABEL_9',
            'B-MISC', 'I-MISC', 'O', 'MISC', 'UNKNOWN', 'OTHER',
            'NEGATIVE', 'POSITIVE', 'NEUTRAL'  # Sentiment labels
        }
        return entity_type.upper() in generic_labels
    
    def _get_presidio_entities(self, text: str) -> List[Dict]:
        """Get entities using Presidio Analyzer"""
        try:
            if not self.presidio_analyzer:
                return []
            
            results = self.presidio_analyzer.analyze(text=text, language="en")
            entities = []
            
            for result in results:
                entities.append({
                    'entity_group': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'score': result.score,
                    'detector': 'presidio_internal'
                })
            
            logger.debug(f"Presidio found {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error in Presidio detection: {e}")
            return []
    
    def _get_regex_entities(self, text: str) -> List[Dict]:
        """Get entities using regex patterns"""
        entities = []
        
        for pattern_def in self.regex_patterns:
            compiled_pattern = pattern_def.get('compiled_pattern')
            if not compiled_pattern:
                continue
            
            try:
                for match in compiled_pattern.finditer(text):
                    start, end = match.span()
                    matched_text = text[start:end]
                    
                    # Basic filtering
                    if len(matched_text) < 3 and not pattern_def.get("context"):
                        continue
                    if not matched_text.strip():
                        continue
                    
                    # Context checking
                    if self._has_context(text, start, end, pattern_def.get("context", [])):
                        entities.append({
                            "entity_group": pattern_def["type"],
                            "start": start,
                            "end": end,
                            "score": 0.9,
                            "detector": "regex_internal"
                        })
                        
            except Exception as e:
                logger.error(f"Error processing regex pattern {pattern_def.get('type')}: {e}")
        
        logger.debug(f"Regex patterns found {len(entities)} entities")
        return entities
    
    def _get_contextual_entities(self, text: str) -> List[Dict]:
        """Get entities using contextual detection"""
        entities = []
        
        # Ambiguous company names with context
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
        
        context_indicators = {
            "ORGANIZATION": [
                r'\b(work|working|job|career|company|corporation|inc|firm)\b',
                r'\b(tech|technology|product|products|device|phone|computer)\b',
                r'\b(stock|share|market|investor|investment)\b',
                r'\b(ceo|founder|employee|staff|team)\b'
            ]
        }
        
        for company_name, entity_type in ambiguous_companies.items():
            pattern = rf'\b{re.escape(company_name)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # Check context
                context_window = self.config.get("context_window", 40)
                context_text = text[max(0, start-context_window):min(len(text), end+context_window)].lower()
                
                is_context_match = False
                for indicator in context_indicators.get(entity_type, []):
                    if re.search(indicator, context_text, re.IGNORECASE):
                        is_context_match = True
                        break
                
                if is_context_match:
                    entities.append({
                        'entity_group': entity_type,
                        'start': start,
                        'end': end,
                        'score': 0.88,
                        'detector': 'context_entity_detector'
                    })
        
        logger.debug(f"Contextual detection found {len(entities)} entities")
        return entities
    
    def _get_fallback_name_entities(self, text: str, existing_entities: List[Dict]) -> List[Dict]:
        """Fallback name detection when other methods don't find person entities"""
        # Check if we already have person entities
        has_person_entities = any(
            e.get('entity_group', '').upper() == 'PERSON' 
            for e in existing_entities
        )
        
        if has_person_entities:
            return []
        
        entities = []
        
        # Simple name pattern with better filtering
        name_patterns = [r'\b([A-Z][a-z]{2,})\b']
        common_non_names = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 
            'my', 'your', 'his', 'her', 'our', 'their', 'its', 'if', 'in', 
            'on', 'at', 'to', 'for', 'with', 'by', 'as', 'of', 'from', 
            'about', 'ssn', 'id', 'cc', 'cv', 'cvv', 'pin', 'no', 'yes', 
            'ok', 'new', 'old', 'first', 'last'
        }
        
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                
                # Skip common non-names
                if name.lower() in common_non_names:
                    continue
                
                # Skip if in blocklist
                if name.lower() in self.common_name_words:
                    continue
                
                # Skip sentence starters
                pre_context = text[max(0, match.start(1)-20):match.start(1)].strip()
                if pre_context == "" or pre_context.endswith(('.', '!', '?', '\n', '\r')):
                    if name in ["The", "This", "That", "These", "Those", "My", "Your", "Our", "Their", "It"]:
                        continue
                
                entities.append({
                    'entity_group': 'PERSON',
                    'start': match.start(1),
                    'end': match.end(1),
                    'score': 0.65,
                    'word': name,
                    'detector': 'fallback_name_detector'
                })
        
        # Names with titles (higher confidence)
        title_pattern = r'(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.) ([A-Z][a-zA-Z\-]{2,})'
        for match in re.finditer(title_pattern, text):
            name = match.group(1)
            if name.lower() not in self.common_name_words:
                entities.append({
                    'entity_group': 'PERSON',
                    'start': match.start(1),
                    'end': match.end(1),
                    'score': 0.92,
                    'detector': 'title_name_detector'
                })
        
        logger.debug(f"Fallback name detection found {len(entities)} entities")
        return entities
    
    def _has_context(self, text: str, span_start: int, span_end: int, context_words: List[str]) -> bool:
        """Check if context words appear near the entity"""
        if not context_words:
            return True
        
        matched_text = text[span_start:span_end]
        
        # Check blocklist
        if matched_text in self.blocklist:
            return False
        
        # Check for project names (common false positive)
        if matched_text.startswith("Project") and len(matched_text.split()) <= 2:
            return False
        
        # Get context
        context_size = self.config.get("context_window", 40)
        text_before = text[:span_start].split()[-context_size:] if span_start > 0 else []
        text_after = text[span_end:].split()[:context_size] if span_end < len(text) else []
        context_text = ' '.join(text_before + text_after).lower()
        
        # Check for context words
        for word in context_words:
            if word.lower() in context_text:
                return True
        
        # Check for nearby indicators
        nearby_text = text[max(0, span_start-20):min(len(text), span_end+10)]
        indicators = r'(?::|=|is\s+|was\s+reset\s+to\s+)'
        if re.search(indicators + r'\s*' + re.escape(matched_text), nearby_text, re.IGNORECASE):
            return True
        
        return False
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """Post-process entities: merge overlapping, filter by confidence, etc."""
        if not entities:
            return []
        
        # Filter by confidence threshold and false positives
        confidence_threshold = self.config.get("entity_confidence_threshold", 0.1)
        filtered_entities = []
        
        for e in entities:
            # Basic confidence filter
            if e.get('score', 0) < confidence_threshold:
                continue
            
            # Get entity text
            start, end = e.get('start', 0), e.get('end', 0)
            entity_text = text[start:end] if start < len(text) and end <= len(text) else ""
            entity_type = e.get('entity_group', '').upper()
            
            # Filter out false positives
            if self._is_false_positive(entity_text, entity_type):
                continue
            
            # Filter out very short entities (likely noise)
            if end - start < 2:  # Less than 2 characters
                continue
            
            # Filter out single character entities
            if len(entity_text.strip()) <= 1:
                continue
            
            # Add entity text to the entity dict for easier processing
            e['entity_text'] = entity_text
            filtered_entities.append(e)
        
        # Sort by start position
        filtered_entities.sort(key=lambda x: x.get('start', 0))
        
        # Remove exact duplicates and merge overlapping entities
        deduplicated_entities = self._deduplicate_entities(filtered_entities, text)
        
        logger.debug(f"Post-processing: {len(entities)} -> {len(deduplicated_entities)} entities")
        return deduplicated_entities
    
    def _deduplicate_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """Remove duplicate entities and merge overlapping ones"""
        if not entities:
            return []
        
        # Group entities by their text content and type
        entity_groups = {}
        for entity in entities:
            entity_text = entity.get('entity_text', '')
            entity_type = entity.get('entity_group', '').upper()
            key = (entity_text.lower(), entity_type)
            
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # For each group, keep only the highest confidence entity
        unique_entities = []
        for (text_key, type_key), group in entity_groups.items():
            if not group:
                continue
            
            # Sort by confidence (highest first)
            group.sort(key=lambda x: x.get('score', 0), reverse=True)
            best_entity = group[0]
            
            # If there are multiple instances of the same entity, keep only the first occurrence
            # (this prevents the same entity from being anonymized multiple times)
            unique_entities.append(best_entity)
        
        # Sort by start position
        unique_entities.sort(key=lambda x: x.get('start', 0))
        
        # Now handle overlapping entities (different entities that overlap in position)
        merged_entities = []
        for entity in unique_entities:
            if not merged_entities:
                merged_entities.append(entity)
                continue
            
            last_entity = merged_entities[-1]
            
            # Check for overlap
            if (entity.get('start', 0) < last_entity.get('end', 0) and 
                entity.get('end', 0) > last_entity.get('start', 0)):
                
                # Choose entity with higher confidence or better type
                if (entity.get('score', 0) > last_entity.get('score', 0) or 
                    self._is_better_entity_type(entity.get('entity_group', ''), last_entity.get('entity_group', ''))):
                    merged_entities[-1] = entity
            else:
                merged_entities.append(entity)
        
        return merged_entities
    
    def _is_false_positive(self, entity_text: str, entity_type: str) -> bool:
        """Check if entity is likely a false positive"""
        entity_text = entity_text.strip().lower()
        entity_type = entity_type.upper()
        
        # Common false positives for all entity types
        common_false_positives = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'a', 'an', 'this', 'that', 'these', 'those',
            '.', ',', ':', ';', '(', ')', '[', ']', '{', '}', '-', '_'
        }
        
        # Organization-specific false positives
        org_false_positives = {
            'this', 'that', 'these', 'those', 'agreement', 'contract', 'document',
            'between', 'among', 'within', 'under', 'over', 'above', 'below',
            'made', 'signed', 'executed', 'entered', 'dated', 'effective',
            'party', 'parties', 'section', 'clause', 'paragraph', 'article',
            'whereas', 'therefore', 'hereby', 'herein', 'hereof', 'hereunder',
            'including', 'excluding', 'subject', 'pursuant', 'accordance',
            'respect', 'regard', 'connection', 'relation', 'reference'
        }
        
        # Person-specific false positives
        person_false_positives = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam', 'miss'
        }
        
        # Location-specific false positives
        location_false_positives = {
            'here', 'there', 'where', 'everywhere', 'somewhere', 'nowhere',
            'above', 'below', 'under', 'over', 'inside', 'outside'
        }
        
        # Check common false positives first
        if entity_text in common_false_positives:
            return True
        
        # Check entity-type specific false positives
        if entity_type in ['ORGANIZATION', 'ORG'] and entity_text in org_false_positives:
            return True
        
        if entity_type in ['PERSON', 'PER'] and entity_text in person_false_positives:
            return True
        
        if entity_type in ['LOCATION', 'LOC', 'GPE'] and entity_text in location_false_positives:
            return True
        
        # Check for single character or very short meaningless text
        if len(entity_text) <= 2 and entity_text.isalpha():
            return True
        
        # Check for common English words that shouldn't be entities
        common_words = {
            'agreement', 'contract', 'document', 'letter', 'email', 'message',
            'text', 'content', 'information', 'data', 'details', 'description',
            'summary', 'report', 'analysis', 'review', 'study', 'research'
        }
        
        if entity_text in common_words:
            return True
        
        return False
    
    def _is_better_entity_type(self, type1: str, type2: str) -> bool:
        """Determine if type1 is better than type2"""
        # Preference order: specific types > generic types
        specific_types = {
            'PERSON', 'ORGANIZATION', 'LOCATION', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
            'SSN', 'CREDIT_CARD', 'IP_ADDRESS', 'URL', 'DATE_TIME'
        }
        
        type1_specific = type1.upper() in specific_types
        type2_specific = type2.upper() in specific_types
        
        # Prefer specific types over generic ones
        if type1_specific and not type2_specific:
            return True
        elif type2_specific and not type1_specific:
            return False
        
        # If both are specific or both are generic, no preference
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        return {
            "presidio_loaded": self.presidio_analyzer is not None,
            "regex_patterns": len(self.regex_patterns),
            "blocklist_size": len(self.blocklist),
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')},
            "model_stats": self.model_manager.get_model_stats()
        }

# Global detection engine instance
_detection_engine: Optional[DetectionEngine] = None

def get_detection_engine() -> DetectionEngine:
    """Get the global detection engine instance"""
    global _detection_engine
    if _detection_engine is None:
        _detection_engine = DetectionEngine()
    return _detection_engine