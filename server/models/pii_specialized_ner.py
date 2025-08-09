"""
PII Specialized NER Model Wrapper

Specialized for comprehensive PII detection with enhanced patterns and
context-aware recognition for sensitive personal information.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("PIISpecializedNER")

class PIISpecializedNERModel:
    """
    PII Specialized Named Entity Recognition model wrapper.
    
    Enhanced for detecting various types of personally identifiable information:
    - Personal names and identifiers
    - Contact information (email, phone, address)
    - Government identifiers (SSN, passport, license)
    - Financial information (credit cards, bank accounts)
    - Biometric and health identifiers
    - Online identifiers and usernames
    """
    
    def __init__(self, model_name: str = None):
        # Use general BERT model as base for PII detection
        self.model_name = model_name or "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        # PII-specific patterns for enhanced detection
        self._pii_patterns = self._compile_pii_patterns()
        
        logger.info(f"PIISpecializedNERModel initialized with model: {self.model_name}")
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for PII entity detection"""
        patterns = {
            # Government identifiers
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'passport': re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
            'license': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
            
            # Financial
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'bank_account': re.compile(r'\b\d{8,17}\b'),
            'routing_number': re.compile(r'\b\d{9}\b'),
            
            # Contact information
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            
            # Online identifiers
            'username': re.compile(r'@[A-Za-z0-9_]{3,15}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            
            # Dates and ages
            'date': re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'),
            'age': re.compile(r'\b(?:age\s+)?(\d{1,3})\s*(?:years?\s*old|y\.?o\.?)\b', re.IGNORECASE),
            
            # Identifiers
            'employee_id': re.compile(r'\b(?:emp|employee|staff)[-\s]?(?:id|number)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE),
            'student_id': re.compile(r'\b(?:student|roll)[-\s]?(?:id|number|no)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE),
            'case_number': re.compile(r'\b(?:case|ticket|ref)[-\s]?(?:number|no|#)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE),
            
            # Medical
            'medical_record': re.compile(r'\b(?:mrn|medical[-\s]record)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE),
            'insurance_id': re.compile(r'\b(?:insurance|policy)[-\s]?(?:id|number)[-\s:]*([A-Z0-9]{4,15})\b', re.IGNORECASE),
            
            # Vehicle
            'license_plate': re.compile(r'\b[A-Z0-9]{2,3}[-\s]?[A-Z0-9]{3,4}\b'),
            'vin': re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
            
            # Biometric placeholders (for documents mentioning biometric data)
            'fingerprint': re.compile(r'\b(?:fingerprint|biometric)[-\s]?(?:id|data)\b', re.IGNORECASE),
            'dna': re.compile(r'\b(?:dna|genetic)[-\s]?(?:profile|data|sequence)\b', re.IGNORECASE)
        }
        return patterns
    
    def load(self) -> bool:
        """Load the PII specialized NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading PII specialized NER model: {self.model_name}")
            start_time = time.time()
            
            self.pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple",
                device=-1  # Use CPU for now
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"PII specialized NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PII specialized NER model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict PII entities in the given text with specialized patterns.
        
        Args:
            text: Input text to analyze for PII
            
        Returns:
            List of detected PII entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - PII specialized model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing PII text of length {len(text)}")
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
                    
                    # Map PII entity types
                    mapped_entity_group = self._map_pii_entity_type(entity_group, entity_text)
                    
                    processed_entity = {
                        'entity_group': mapped_entity_group,
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'pii_specialized_ner_model',
                        'original_label': entity_group
                    }
                    
                    # Boost confidence for PII context
                    if self._is_pii_context(text, start_pos, end_pos):
                        processed_entity['score'] = min(1.0, score * 1.1)
                        processed_entity['pii_context'] = True
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found PII entity: {mapped_entity_group} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
            
            # Add pattern-based PII entities
            pattern_entities = self._detect_pii_patterns(text)
            processed_results.extend(pattern_entities)
            
            # Remove duplicates and overlaps
            processed_results = self._remove_overlapping_entities(processed_results)
            
            processing_time = time.time() - start_time
            logger.debug(f"PII specialized NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during PII specialized NER prediction: {e}", exc_info=True)
            return []
    
    def _detect_pii_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities using specialized regex patterns"""
        pattern_entities = []
        
        for pattern_name, pattern in self._pii_patterns.items():
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                entity_text = match.group()
                
                # Use captured group if available (for patterns with groups)
                if match.groups():
                    captured_text = match.group(1)
                    if captured_text:
                        # Adjust positions for captured group
                        group_start = match.start(1)
                        group_end = match.end(1)
                        start_pos = group_start
                        end_pos = group_end
                        entity_text = captured_text
                
                # Skip very short matches for certain patterns
                if len(entity_text) < 3 and pattern_name not in ['age', 'zip_code']:
                    continue
                
                # Map pattern to entity type
                entity_type = self._pattern_to_pii_entity_type(pattern_name, entity_text)
                
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_pii_pattern_confidence(pattern_name, entity_text, text, start_pos)
                
                pattern_entity = {
                    'entity_group': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'score': confidence,
                    'word': entity_text,
                    'detector': 'pii_pattern_detector',
                    'pattern_type': pattern_name
                }
                
                pattern_entities.append(pattern_entity)
                
                logger.debug(f"Found PII pattern: {entity_type} '{entity_text}' "
                           f"({start_pos}:{end_pos}) pattern={pattern_name}")
        
        return pattern_entities
    
    def _pattern_to_pii_entity_type(self, pattern_name: str, entity_text: str) -> str:
        """Map pattern name to PII entity type"""
        pattern_mappings = {
            'ssn': 'SSN',
            'passport': 'PASSPORT_NUMBER',
            'license': 'DRIVER_LICENSE',
            'credit_card': 'CREDIT_CARD',
            'bank_account': 'BANK_ACCOUNT',
            'routing_number': 'ROUTING_NUMBER',
            'email': 'EMAIL_ADDRESS',
            'phone': 'PHONE_NUMBER',
            'zip_code': 'ZIP_CODE',
            'username': 'USERNAME',
            'ip_address': 'IP_ADDRESS',
            'url': 'URL',
            'date': 'DATE_TIME',
            'age': 'AGE',
            'employee_id': 'EMPLOYEE_ID',
            'student_id': 'STUDENT_ID',
            'case_number': 'CASE_NUMBER',
            'medical_record': 'MEDICAL_RECORD_NUMBER',
            'insurance_id': 'INSURANCE_ID',
            'license_plate': 'LICENSE_PLATE',
            'vin': 'VIN',
            'fingerprint': 'BIOMETRIC_IDENTIFIER',
            'dna': 'BIOMETRIC_IDENTIFIER'
        }
        
        return pattern_mappings.get(pattern_name, 'PII_IDENTIFIER')
    
    def _calculate_pii_pattern_confidence(self, pattern_name: str, entity_text: str, 
                                        full_text: str, start_pos: int) -> float:
        """Calculate confidence score for PII pattern-based detection"""
        base_confidence = {
            'ssn': 0.95,
            'passport': 0.90,
            'license': 0.85,
            'credit_card': 0.90,
            'bank_account': 0.80,
            'routing_number': 0.85,
            'email': 0.95,
            'phone': 0.90,
            'zip_code': 0.85,
            'username': 0.80,
            'ip_address': 0.90,
            'url': 0.95,
            'date': 0.75,
            'age': 0.80,
            'employee_id': 0.85,
            'student_id': 0.85,
            'case_number': 0.80,
            'medical_record': 0.90,
            'insurance_id': 0.85,
            'license_plate': 0.85,
            'vin': 0.95,
            'fingerprint': 0.90,
            'dna': 0.90
        }
        
        confidence = base_confidence.get(pattern_name, 0.70)
        
        # Adjust based on context
        context_size = 30
        context_start = max(0, start_pos - context_size)
        context_end = min(len(full_text), start_pos + len(entity_text) + context_size)
        context = full_text[context_start:context_end].lower()
        
        # Boost confidence for certain contextual clues
        if pattern_name == 'ssn' and any(word in context for word in ['social', 'security', 'ssn']):
            confidence += 0.05
        elif pattern_name == 'credit_card' and any(word in context for word in ['card', 'credit', 'payment']):
            confidence += 0.05
        elif pattern_name == 'phone' and any(word in context for word in ['phone', 'call', 'contact', 'mobile']):
            confidence += 0.05
        elif pattern_name == 'email' and any(word in context for word in ['email', 'contact', 'send', '@']):
            confidence += 0.05
        
        # Reduce confidence for common false positives
        if pattern_name == 'date' and any(word in context for word in ['version', 'build', 'release']):
            confidence -= 0.15
        elif pattern_name == 'phone' and any(word in context for word in ['port', 'extension', 'ext']):
            confidence -= 0.10
        
        return min(1.0, max(0.1, confidence))
    
    def _map_pii_entity_type(self, original_type: str, entity_text: str) -> str:
        """Map PII entity types to standard categories"""
        pii_mappings = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'MISCELLANEOUS': 'PII_IDENTIFIER',
            'MISC': 'PII_IDENTIFIER'
        }
        
        normalized_type = original_type.upper().strip()
        
        # Check for direct mapping
        if normalized_type in pii_mappings:
            return pii_mappings[normalized_type]
        
        return original_type if original_type else 'PII_IDENTIFIER'
    
    def _is_pii_context(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if the entity appears in a PII-sensitive context"""
        # Get context around the entity
        context_size = 40
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        # PII context indicators
        pii_indicators = [
            'personal', 'private', 'confidential', 'sensitive', 'protected',
            'identity', 'identification', 'id', 'number', 'account', 'record',
            'contact', 'address', 'phone', 'email', 'social', 'security',
            'credit', 'card', 'bank', 'financial', 'medical', 'health',
            'insurance', 'policy', 'license', 'passport', 'driver',
            'employee', 'student', 'patient', 'customer', 'client',
            'name', 'birth', 'age', 'date', 'ssn', 'dob', 'gender'
        ]
        
        return any(indicator in context for indicator in pii_indicators)
    
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
            for i, existing in enumerate(filtered_entities):
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # There's an overlap
                    if entity['score'] > existing['score']:
                        # Replace existing with current entity
                        filtered_entities[i] = entity
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return sorted(filtered_entities, key=lambda x: x['start'])
    
    def get_info(self) -> Dict[str, Any]:
        """Get PII specialized model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "pii_specialized_ner",
            "domain": "pii_specialized",
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "SSN", "PASSPORT_NUMBER",
                "DRIVER_LICENSE", "CREDIT_CARD", "BANK_ACCOUNT", "ROUTING_NUMBER",
                "EMAIL_ADDRESS", "PHONE_NUMBER", "ZIP_CODE", "USERNAME", "IP_ADDRESS",
                "URL", "DATE_TIME", "AGE", "EMPLOYEE_ID", "STUDENT_ID", "CASE_NUMBER",
                "MEDICAL_RECORD_NUMBER", "INSURANCE_ID", "LICENSE_PLATE", "VIN",
                "BIOMETRIC_IDENTIFIER", "PII_IDENTIFIER"
            ],
            "pattern_count": len(self._pii_patterns),
            "specialization": "Comprehensive PII detection with enhanced pattern recognition"
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading PII specialized NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_pii_specialized_ner_model(model_name: str = None) -> PIISpecializedNERModel:
    """Create and return a PIISpecializedNERModel instance"""
    return PIISpecializedNERModel(model_name)