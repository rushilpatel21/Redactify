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
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, Tuple
from presidio_analyzer import AnalyzerEngine
from model_manager import get_model_manager

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
        self.model_manager = get_model_manager()
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
            "entity_confidence_threshold": 0.1,
            "enable_context_detection": True,
            "enable_fallback_name_detector": True,
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
        if not text:
            return [], []
        
        start_time = time.time()
        all_entities = []
        
        # Step 1: Classify text to determine which models to use
        if domains is None:
            domains = await self._classify_text(text)
        
        logger.info(f"Processing text with domains: {domains}")
        
        # Step 2: Run internal detectors in parallel
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = []
            
            # Internal detectors
            futures.append(executor.submit(self._get_presidio_entities, text))
            futures.append(executor.submit(self._get_regex_entities, text))
            futures.append(executor.submit(self._get_contextual_entities, text))
            
            # ML model detectors based on domains
            model_futures = []
            
            # Always use general model
            model_futures.append(self._get_model_entities(text, "general"))
            
            # Add specialized models based on domains and configuration
            if "medical" in domains and self.config["enable_medical_pii"]:
                model_futures.append(self._get_model_entities(text, "medical"))
            if "technical" in domains and self.config["enable_technical_ner"]:
                model_futures.append(self._get_model_entities(text, "technical"))
            if "legal" in domains and self.config["enable_legal_ner"]:
                model_futures.append(self._get_model_entities(text, "legal"))
            if "financial" in domains and self.config["enable_financial_ner"]:
                model_futures.append(self._get_model_entities(text, "financial"))
            if self.config["enable_pii_specialized"]:
                model_futures.append(self._get_model_entities(text, "pii_specialized"))
            
            # Wait for internal detectors
            for future in as_completed(futures):
                try:
                    entities = future.result()
                    if entities:
                        all_entities.extend(entities)
                except Exception as e:
                    logger.error(f"Error in internal detector: {e}")
            
            # Wait for model detectors
            for model_future in model_futures:
                try:
                    entities = await model_future
                    if entities:
                        all_entities.extend(entities)
                except Exception as e:
                    logger.error(f"Error in model detector: {e}")
        
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
        """Classify text to determine relevant domains using Gemini"""
        try:
            gemini_model = self.model_manager.get_gemini_model()
            if not gemini_model:
                logger.info("Gemini model not available, using default classification")
                return ["general"]
            
            # Use the text classifier logic adapted for Gemini
            max_length = 4000
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            categories = ["medical", "technical", "legal", "financial", "general"]
            
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
        """Get entities from a specific ML model"""
        try:
            model_wrapper = await self.model_manager.get_model(model_name)
            if not model_wrapper:
                logger.warning(f"Model {model_name} not available")
                return []
            
            # Run NER prediction using the wrapper's predict method
            entities = model_wrapper.predict(text)
            
            logger.debug(f"Model {model_name} found {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error getting entities from model {model_name}: {e}")
            return []
    
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
        
        # Filter by confidence threshold
        confidence_threshold = self.config.get("entity_confidence_threshold", 0.1)
        filtered_entities = [
            e for e in entities 
            if e.get('score', 0) >= confidence_threshold
        ]
        
        # Sort by start position
        filtered_entities.sort(key=lambda x: x.get('start', 0))
        
        # Merge overlapping entities (simplified version)
        merged_entities = []
        for entity in filtered_entities:
            if not merged_entities:
                merged_entities.append(entity)
                continue
            
            last_entity = merged_entities[-1]
            
            # Check for overlap
            if (entity.get('start', 0) < last_entity.get('end', 0) and 
                entity.get('end', 0) > last_entity.get('start', 0)):
                
                # Choose entity with higher confidence
                if entity.get('score', 0) > last_entity.get('score', 0):
                    merged_entities[-1] = entity
            else:
                merged_entities.append(entity)
        
        logger.debug(f"Post-processing: {len(entities)} -> {len(merged_entities)} entities")
        return merged_entities
    
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