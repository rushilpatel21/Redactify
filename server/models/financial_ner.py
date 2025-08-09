"""
Financial NER Model Wrapper

Specialized for financial document entity recognition including:
- Account numbers and financial identifiers
- Credit card and payment information
- Banking details
- Investment and trading information
- Financial institution names
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("FinancialNER")

class FinancialNERModel:
    """
    Financial Named Entity Recognition model wrapper.
    
    Specialized for detecting financial entities such as:
    - Account numbers and routing numbers
    - Credit card and payment card information
    - Banking and financial institution details
    - Investment account identifiers
    - Trading and transaction information
    - Financial professional information
    """
    
    def __init__(self, model_name: str = None):
        # Use financial-focused model or fall back to general model
        self.model_name = model_name or "ProsusAI/finbert"
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        # Financial-specific patterns
        self._financial_patterns = self._compile_financial_patterns()
        
        logger.info(f"FinancialNERModel initialized with model: {self.model_name}")
    
    def _compile_financial_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for financial entity detection"""
        patterns = {
            'account_number': re.compile(r'\b(?:account|acct)[-\s]?(?:no|number)[-\s:]*([0-9]{6,17})\b', re.IGNORECASE),
            'routing_number': re.compile(r'\b(?:routing|aba)[-\s]?(?:no|number)[-\s:]*([0-9]{9})\b', re.IGNORECASE),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'swift_code': re.compile(r'\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b'),
            'iban': re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]?){0,16}\b'),
            'cusip': re.compile(r'\b[0-9]{3}[0-9A-Z]{5}[0-9]\b'),
            'isin': re.compile(r'\b[A-Z]{2}[0-9A-Z]{9}[0-9]\b'),
            'ticker_symbol': re.compile(r'\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b'),
            'amount': re.compile(r'\$\s*[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?'),
            'tax_id': re.compile(r'\b(?:ein|tax[-\s]id)[-\s:]*([0-9]{2}-[0-9]{7})\b', re.IGNORECASE),
            'portfolio_id': re.compile(r'\b(?:portfolio|fund)[-\s]?(?:id|number)[-\s:]*([A-Z0-9]{4,12})\b', re.IGNORECASE)
        }
        return patterns
    
    def load(self) -> bool:
        """Load the financial NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading financial NER model: {self.model_name}")
            start_time = time.time()
            
            try:
                self.pipeline = pipeline(
                    "ner", 
                    model=self.model_name, 
                    aggregation_strategy="simple",
                    device=-1  # Use CPU for now
                )
            except Exception as e:
                logger.warning(f"Failed to load financial model {self.model_name}, falling back to general model: {e}")
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
            
            logger.info(f"Financial NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load financial NER model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict financial named entities in the given text.
        
        Args:
            text: Input text to analyze for financial entities
            
        Returns:
            List of detected financial entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - financial model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing financial text of length {len(text)}")
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
                    
                    # Map financial entity types
                    mapped_entity_group = self._map_financial_entity_type(entity_group, entity_text)
                    
                    processed_entity = {
                        'entity_group': mapped_entity_group,
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'financial_ner_model',
                        'original_label': entity_group
                    }
                    
                    # Boost confidence for financial context
                    if self._is_financial_context(text, start_pos, end_pos):
                        processed_entity['score'] = min(1.0, score * 1.1)
                        processed_entity['financial_context'] = True
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found financial entity: {mapped_entity_group} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
            
            # Add pattern-based financial entities
            pattern_entities = self._detect_financial_patterns(text)
            processed_results.extend(pattern_entities)
            
            # Remove duplicates and overlaps
            processed_results = self._remove_overlapping_entities(processed_results)
            
            processing_time = time.time() - start_time
            logger.debug(f"Financial NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during financial NER prediction: {e}", exc_info=True)
            return []
    
    def _detect_financial_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect financial entities using regex patterns"""
        pattern_entities = []
        
        for pattern_name, pattern in self._financial_patterns.items():
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
                
                # Skip very short matches for certain patterns
                if len(entity_text) < 4 and pattern_name not in ['amount']:
                    continue
                
                # Map pattern to entity type
                entity_type = self._pattern_to_financial_entity_type(pattern_name)
                
                # Calculate confidence
                confidence = self._calculate_financial_pattern_confidence(pattern_name, entity_text)
                
                pattern_entity = {
                    'entity_group': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'score': confidence,
                    'word': entity_text,
                    'detector': 'financial_pattern_detector',
                    'pattern_type': pattern_name
                }
                
                pattern_entities.append(pattern_entity)
                
                logger.debug(f"Found financial pattern: {entity_type} '{entity_text}' "
                           f"({start_pos}:{end_pos}) pattern={pattern_name}")
        
        return pattern_entities
    
    def _pattern_to_financial_entity_type(self, pattern_name: str) -> str:
        """Map pattern name to financial entity type"""
        pattern_mappings = {
            'account_number': 'ACCOUNT_NUMBER',
            'routing_number': 'ROUTING_NUMBER',
            'credit_card': 'CREDIT_CARD',
            'swift_code': 'SWIFT_CODE',
            'iban': 'IBAN',
            'cusip': 'CUSIP',
            'isin': 'ISIN',
            'ticker_symbol': 'TICKER_SYMBOL',
            'amount': 'MONETARY_AMOUNT',
            'tax_id': 'TAX_ID',
            'portfolio_id': 'PORTFOLIO_ID'
        }
        
        return pattern_mappings.get(pattern_name, 'FINANCIAL_IDENTIFIER')
    
    def _calculate_financial_pattern_confidence(self, pattern_name: str, entity_text: str) -> float:
        """Calculate confidence score for financial pattern-based detection"""
        base_confidence = {
            'account_number': 0.85,
            'routing_number': 0.95,
            'credit_card': 0.90,
            'swift_code': 0.95,
            'iban': 0.95,
            'cusip': 0.90,
            'isin': 0.90,
            'ticker_symbol': 0.80,
            'amount': 0.85,
            'tax_id': 0.90,
            'portfolio_id': 0.85
        }
        
        confidence = base_confidence.get(pattern_name, 0.75)
        
        # Adjust based on entity characteristics
        if pattern_name == 'credit_card':
            # Validate credit card using Luhn algorithm (simplified check)
            if self._is_valid_credit_card(entity_text):
                confidence += 0.05
        
        elif pattern_name == 'ticker_symbol':
            # Common ticker symbols get higher confidence
            if entity_text.upper() in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']:
                confidence += 0.10
        
        return min(1.0, max(0.1, confidence))
    
    def _is_valid_credit_card(self, card_number: str) -> bool:
        """Simple Luhn algorithm check for credit card validation"""
        # Remove any spaces or dashes
        card_number = re.sub(r'[-\s]', '', card_number)
        
        if not card_number.isdigit():
            return False
        
        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10 == 0
        
        return luhn_check(card_number)
    
    def _map_financial_entity_type(self, original_type: str, entity_text: str) -> str:
        """Map financial entity types to standard categories"""
        financial_mappings = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'MISCELLANEOUS': 'FINANCIAL_ENTITY',
            'MISC': 'FINANCIAL_ENTITY'
        }
        
        normalized_type = original_type.upper().strip()
        
        # Check for direct mapping
        if normalized_type in financial_mappings:
            return financial_mappings[normalized_type]
        
        # Pattern-based detection for entity text
        entity_upper = entity_text.upper()
        
        # Check for financial institutions
        if any(word in entity_upper for word in ['BANK', 'CREDIT', 'FINANCIAL', 'INVESTMENT']):
            return 'ORGANIZATION'
        
        return original_type if original_type else 'FINANCIAL_ENTITY'
    
    def _is_financial_context(self, text: str, start_pos: int, end_pos: int) -> bool:
        """Check if the entity appears in a financial context"""
        # Get context around the entity
        context_size = 50
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        # Financial context indicators
        financial_indicators = [
            'bank', 'account', 'credit', 'debit', 'card', 'payment', 'transaction',
            'financial', 'money', 'dollar', 'currency', 'investment', 'portfolio',
            'trading', 'stock', 'bond', 'fund', 'asset', 'liability', 'equity',
            'loan', 'mortgage', 'interest', 'rate', 'balance', 'deposit', 'withdrawal',
            'transfer', 'wire', 'ach', 'routing', 'swift', 'iban', 'cusip', 'isin',
            'broker', 'dealer', 'exchange', 'market', 'ticker', 'symbol', 'price'
        ]
        
        return any(indicator in context for indicator in financial_indicators)
    
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
        """Get financial model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "financial_ner",
            "domain": "financial",
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "ACCOUNT_NUMBER", "ROUTING_NUMBER",
                "CREDIT_CARD", "SWIFT_CODE", "IBAN", "CUSIP", "ISIN", "TICKER_SYMBOL",
                "MONETARY_AMOUNT", "TAX_ID", "PORTFOLIO_ID", "FINANCIAL_ENTITY",
                "FINANCIAL_IDENTIFIER"
            ],
            "pattern_count": len(self._financial_patterns),
            "specialization": "Financial document and transaction PII detection"
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading financial NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_financial_ner_model(model_name: str = None) -> FinancialNERModel:
    """Create and return a FinancialNERModel instance"""
    return FinancialNERModel(model_name)