"""
Legal NER Model Wrapper

Specialized for legal document entity recognition including:
- Legal entities and parties
- Case numbers and legal references
- Court information
- Legal professional names
- Contract and agreement identifiers
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("LegalNER")

class LegalNERModel:
    """
    Legal Named Entity Recognition model wrapper.
    
    Specialized for detecting legal entities such as:
    - Legal parties (plaintiff, defendant, counsel)
    - Case numbers and docket numbers
    - Court names and jurisdictions
    - Legal document references
    - Contract and agreement identifiers
    - Legal professional information
    """
    
    def __init__(self, model_name: str = None):
        # Use legal-focused model or fall back to general model
        self.model_name = model_name or "nlpaueb/legal-bert-base-uncased"
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        # Legal-specific patterns
        self._legal_patterns = self._compile_legal_patterns()
        
        logger.info(f"LegalNERModel initialized with model: {self.model_name}")
    
    def _compile_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for legal entity detection"""
        patterns = {
            'case_number': re.compile(r'\b(?:case|docket)[-\s]?(?:no|number)[-\s:]*([A-Z0-9-]{4,20})\b', re.IGNORECASE),
            'court': re.compile(r'\b(?:court|tribunal|judge)[-\s]?(?:of|in)?\s*([A-Z][A-Za-z\s]{5,30})\b', re.IGNORECASE),
            'legal_citation': re.compile(r'\b\d+\s+[A-Z][A-Za-z\.]+\s+\d+\b'),
            'contract_id': re.compile(r'\b(?:contract|agreement)[-\s]?(?:no|number|id)[-\s:]*([A-Z0-9-]{4,15})\b', re.IGNORECASE),
            'bar_number': re.compile(r'\b(?:bar|attorney)[-\s]?(?:no|number)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE),
            'statute': re.compile(r'\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+\b', re.IGNORECASE)
        }
        return patterns
    
    def load(self) -> bool:
        """Load the legal NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading legal NER model: {self.model_name}")
            start_time = time.time()
            
            try:
                self.pipeline = pipeline(
                    "ner", 
                    model=self.model_name, 
                    aggregation_strategy="simple",
                    device=-1  # Use CPU for now
                )
            except Exception as e:
                logger.warning(f"Failed to load legal model {self.model_name}, falling back to general model: {e}")
                # Fallback to general BERT model
                self.model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                self.pipeline = pipeline(
                    "ner", 
                    model=self.model_name, 
                    aggregation_strategy="simple",
                    device=-1
                )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Legal NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load legal NER model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict legal named entities in the given text.
        
        Args:
            text: Input text to analyze for legal entities
            
        Returns:
            List of detected legal entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - legal model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing legal text of length {len(text)}")
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
                    
                    # Map legal entity types
                    mapped_entity_group = self._map_legal_entity_type(entity_group, entity_text)
                    
                    processed_entity = {
                        'entity_group': mapped_entity_group,
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'legal_ner_model',
                        'original_label': entity_group
                    }
                    
                    # Boost confidence for legal context
                    if self._is_legal_context(text, start_pos, end_pos):
                        processed_entity['score'] = min(1.0, score * 1.1)
                        processed_entity['legal_context'] = True
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found legal entity: {mapped_entity_group} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
            
            # Add pattern-based legal entities
            pattern_entities = self._detect_legal_patterns(text)
            processed_results.extend(pattern_entities)
            
            # Remove duplicates and overlaps
            processed_results = self._remove_overlapping_entities(processed_results)
            
            processing_time = time.time() - start_time
            logger.debug(f"Legal NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during legal NER prediction: {e}", exc_info=True)
            return []
    
    def _detect_legal_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect legal entities using regex patterns"""
        pattern_entities = []
        
        for pattern_name, pattern in self._legal_patterns.items():
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                entity_text = match.group()
                
                # Use captured group if available
                if match.groups():
                    captured_text = match.group(1)
                    if captured_text:
                        group_start = match.start(1)
                        group_end = match.end(1)
                        start_pos = group_start
                        end_pos = group_end
                        entity_text = captured_text
                
                # Skip very short matches
                if len(entity_text) < 3:
                    continue
                
                # Map pattern to entity type
                entity_type = self._pattern_to_legal_entity_type(pattern_name)
                
                # Calculate confidence
                confidence = self._calculate_legal_pattern_confidence(pattern_name, entity_text)
                
                pattern_entity = {
                    'entity_group': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'score': confidence,
                    'word': entity_text,
                    'detector': 'legal_pattern_detector',
                    'pattern_type': pattern_name
                }
                
                pattern_entities.append(pattern_entity)
                
                logger.debug(f"Found legal pattern: {entity_type} '{entity_text}' "
                           f"({start_pos}:{end_pos}) pattern={pattern_name}")
        
        return pattern_entities
    
    def _pattern_to_legal_entity_type(self, pattern_name: str) -> str:
        """Map pattern name to legal entity type"""
        pattern_mappings = {
            'case_number': 'CASE_NUMBER',
            'court': 'COURT',
            'legal_citation': 'LEGAL_CITATION',
            'contract_id': 'CONTRACT_ID',
            'bar_number': 'BAR_NUMBER',
            'statute': 'STATUTE_REFERENCE'
        }
        
        return pattern_mappings.get(pattern_name, 'LEGAL_IDENTIFIER')
    
    def _calculate_legal_pattern_confidence(self, pattern_name: str, entity_text: str) -> float:
        """Calculate confidence score for legal pattern-based detection"""
        base_confidence = {
            'case_number': 0.90,
            'court': 0.85,
            'legal_citation': 0.95,
            'contract_id': 0.85,
            'bar_number': 0.90,
            'statute': 0.95
        }
        
        return base_confidence.get(pattern_name, 0.80)
    
    def _map_legal_entity_type(self, original_type: str, entity_text: str) -> str:
        """Map legal entity types to standard categories"""
        legal_mappings = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'MISCELLANEOUS': 'LEGAL_ENTITY',
            'MISC': 'LEGAL_ENTITY'
        }
        
        normalized_type = original_type.upper().strip()
        
        # Check for direct mapping
        if normalized_type in legal_mappings:
            return legal_mappings[normalized_type]
        
        return original_type if original_type else 'LEGAL_ENTITY'
    
    def _is_legal_context(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if the entity appears in a legal context"""
        # Get context around the entity
        context_size = 50
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        # Legal context indicators
        legal_indicators = [
            'court', 'judge', 'jury', 'trial', 'case', 'lawsuit', 'litigation',
            'plaintiff', 'defendant', 'counsel', 'attorney', 'lawyer', 'legal',
            'contract', 'agreement', 'clause', 'statute', 'law', 'regulation',
            'jurisdiction', 'docket', 'filing', 'motion', 'brief', 'evidence',
            'testimony', 'witness', 'verdict', 'judgment', 'appeal', 'hearing',
            'deposition', 'subpoena', 'warrant', 'injunction', 'settlement'
        ]
        
        return any(indicator in context for indicator in legal_indicators)
    
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
        """Get legal model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "legal_ner",
            "domain": "legal",
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "CASE_NUMBER", "COURT",
                "LEGAL_CITATION", "CONTRACT_ID", "BAR_NUMBER", "STATUTE_REFERENCE",
                "LEGAL_ENTITY", "LEGAL_IDENTIFIER"
            ],
            "pattern_count": len(self._legal_patterns),
            "specialization": "Legal document and case file PII detection"
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading legal NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_legal_ner_model(model_name: str = None) -> LegalNERModel:
    """Create and return a LegalNERModel instance"""
    return LegalNERModel(model_name)