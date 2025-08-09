"""
Technical NER Model Wrapper

Specialized for technical document entity recognition including:
- API keys and tokens
- System identifiers
- Technical specifications
- Software and hardware names
- Network information
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("TechnicalNER")

class TechnicalNERModel:
    """
    Technical Named Entity Recognition model wrapper.
    
    Specialized for detecting technical entities such as:
    - API keys and authentication tokens
    - System identifiers and UUIDs
    - IP addresses and network information
    - Software and hardware names
    - Technical specifications
    - Configuration parameters
    - Database identifiers
    """
    
    def __init__(self, model_name: str = None):
        # Use technical-focused model or fall back to general model
        self.model_name = model_name or "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        # Technical patterns for enhanced detection
        self._technical_patterns = self._compile_technical_patterns()
        
        logger.info(f"TechnicalNERModel initialized with model: {self.model_name}")
    
    def _compile_technical_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for technical entity detection"""
        patterns = {
            'api_key': re.compile(r'\b[A-Za-z0-9]{20,}\b'),
            'uuid': re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'mac_address': re.compile(r'\b[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}\b', re.IGNORECASE),
            'version': re.compile(r'\bv?\d+\.\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?\b'),
            'port': re.compile(r'\b(?:port\s+)?([1-9][0-9]{0,4})\b', re.IGNORECASE),
            'hash': re.compile(r'\b[a-f0-9]{32,}\b', re.IGNORECASE),
            'token': re.compile(r'\b(?:token|key|secret)[\s:=]+([A-Za-z0-9+/]{20,}={0,2})\b', re.IGNORECASE)
        }
        return patterns
    
    def load(self) -> bool:
        """Load the technical NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading technical NER model: {self.model_name}")
            start_time = time.time()
            
            self.pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple",
                device=-1  # Use CPU for now
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Technical NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load technical NER model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict technical named entities in the given text.
        
        Args:
            text: Input text to analyze for technical entities
            
        Returns:
            List of detected technical entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - technical model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing technical text of length {len(text)}")
            start_time = time.time()
            
            # Run standard NER prediction
            raw_results = self.pipeline(text)
            
            # Process standard NER results
            processed_results = []
            for item in raw_results:
                # Handle numpy types
                if isinstance(item.get('score'), (np.floating, np.integer)):
                    score = float(item['score'])
                else:
                    score = float(item.get('score', 0))
                
                # Ensure integer positions
                start_pos = int(item.get('start', 0))
                end_pos = int(item.get('end', 0))
                
                # Validate span
                if start_pos >= 0 and end_pos <= len(text) and start_pos < end_pos:
                    entity_text = text[start_pos:end_pos]
                    entity_group = item.get('entity_group', 'UNKNOWN')
                    
                    # Map technical entity types
                    mapped_entity_group = self._map_technical_entity_type(entity_group, entity_text)
                    
                    processed_entity = {
                        'entity_group': mapped_entity_group,
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'technical_ner_model',
                        'original_label': entity_group
                    }
                    
                    # Boost confidence for technical context
                    if self._is_technical_context(text, start_pos, end_pos):
                        processed_entity['score'] = min(1.0, score * 1.15)
                        processed_entity['technical_context'] = True
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found technical entity: {mapped_entity_group} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
            
            # Add pattern-based technical entities
            pattern_entities = self._detect_technical_patterns(text)
            processed_results.extend(pattern_entities)
            
            # Remove duplicates and overlaps
            processed_results = self._remove_overlapping_entities(processed_results)
            
            processing_time = time.time() - start_time
            logger.debug(f"Technical NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during technical NER prediction: {e}", exc_info=True)
            return []
    
    def _detect_technical_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect technical entities using regex patterns"""
        pattern_entities = []
        
        for pattern_name, pattern in self._technical_patterns.items():
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                entity_text = match.group()
                
                # Skip very short matches unless they're specific patterns
                if len(entity_text) < 4 and pattern_name not in ['port', 'ip_address']:
                    continue
                
                # Map pattern to entity type
                entity_type = self._pattern_to_entity_type(pattern_name, entity_text)
                
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_pattern_confidence(pattern_name, entity_text)
                
                pattern_entity = {
                    'entity_group': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'score': confidence,
                    'word': entity_text,
                    'detector': 'technical_pattern_detector',
                    'pattern_type': pattern_name
                }
                
                pattern_entities.append(pattern_entity)
                
                logger.debug(f"Found technical pattern: {entity_type} '{entity_text}' "
                           f"({start_pos}:{end_pos}) pattern={pattern_name}")
        
        return pattern_entities
    
    def _pattern_to_entity_type(self, pattern_name: str, entity_text: str) -> str:
        """Map pattern name to entity type"""
        pattern_mappings = {
            'api_key': 'API_KEY',
            'uuid': 'UUID',
            'ip_address': 'IP_ADDRESS',
            'mac_address': 'MAC_ADDRESS',
            'version': 'VERSION',
            'port': 'PORT',
            'hash': 'HASH',
            'token': 'AUTHENTICATION_TOKEN'
        }
        
        return pattern_mappings.get(pattern_name, 'TECHNICAL_IDENTIFIER')
    
    def _calculate_pattern_confidence(self, pattern_name: str, entity_text: str) -> float:
        """Calculate confidence score for pattern-based detection"""
        base_confidence = {
            'api_key': 0.85,
            'uuid': 0.95,
            'ip_address': 0.90,
            'mac_address': 0.95,
            'version': 0.80,
            'port': 0.75,
            'hash': 0.85,
            'token': 0.90
        }
        
        confidence = base_confidence.get(pattern_name, 0.70)
        
        # Adjust based on entity text characteristics
        if pattern_name == 'api_key':
            # Longer keys are more likely to be real API keys
            if len(entity_text) > 32:
                confidence += 0.05
            elif len(entity_text) < 20:
                confidence -= 0.10
        
        elif pattern_name == 'hash':
            # Standard hash lengths get higher confidence
            if len(entity_text) in [32, 40, 64, 128]:
                confidence += 0.05
        
        return min(1.0, max(0.1, confidence))
    
    def _map_technical_entity_type(self, original_type: str, entity_text: str) -> str:
        """Map technical entity types to standard categories"""
        technical_mappings = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION', 
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'MISCELLANEOUS': 'TECHNICAL_IDENTIFIER',
            'MISC': 'TECHNICAL_IDENTIFIER'
        }
        
        normalized_type = original_type.upper().strip()
        
        # Check for direct mapping
        if normalized_type in technical_mappings:
            return technical_mappings[normalized_type]
        
        # Pattern-based detection for entity text
        entity_upper = entity_text.upper()
        
        # Check for technical identifiers
        if any(pattern in entity_upper for pattern in ['API', 'KEY', 'TOKEN', 'SECRET']):
            return 'API_KEY'
        
        if any(pattern in entity_upper for pattern in ['UUID', 'GUID']):
            return 'UUID'
        
        if re.match(r'^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$', entity_upper):
            return 'UUID'
        
        return original_type if original_type else 'TECHNICAL_IDENTIFIER'
    
    def _is_technical_context(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if the entity appears in a technical context"""
        # Get context around the entity
        context_size = 50
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        # Technical context indicators
        technical_indicators = [
            'api', 'key', 'token', 'secret', 'config', 'configuration',
            'server', 'database', 'system', 'application', 'service',
            'endpoint', 'url', 'uri', 'protocol', 'port', 'host',
            'authentication', 'authorization', 'credential', 'access',
            'version', 'build', 'release', 'deployment', 'environment',
            'log', 'debug', 'error', 'exception', 'trace', 'monitoring',
            'network', 'connection', 'socket', 'tcp', 'udp', 'http',
            'ssl', 'tls', 'certificate', 'encryption', 'hash', 'algorithm'
        ]
        
        return any(indicator in context for indicator in technical_indicators)
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        filtered_entities = []
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in filtered_entities:
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # There's an overlap
                    if entity['score'] > existing['score']:
                        # Replace existing with current entity
                        filtered_entities.remove(existing)
                        filtered_entities.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return sorted(filtered_entities, key=lambda x: x['start'])
    
    def get_info(self) -> Dict[str, Any]:
        """Get technical model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "technical_ner",
            "domain": "technical",
            "supported_entities": [
                "API_KEY", "UUID", "IP_ADDRESS", "MAC_ADDRESS", "VERSION",
                "PORT", "HASH", "AUTHENTICATION_TOKEN", "TECHNICAL_IDENTIFIER",
                "PERSON", "ORGANIZATION", "LOCATION"
            ],
            "pattern_count": len(self._technical_patterns),
            "specialization": "Technical documentation and system configuration PII detection"
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading technical NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_technical_ner_model(model_name: str = None) -> TechnicalNERModel:
    """Create and return a TechnicalNERModel instance"""
    return TechnicalNERModel(model_name)