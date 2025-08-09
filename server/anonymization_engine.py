"""
Anonymization Engine for Redactify MCP Server

This module provides comprehensive text anonymization capabilities including:
- Full redaction with pseudonymization
- Partial masking with format preservation
- Custom anonymization strategies
- Batch processing support
"""

import os
import logging
import hashlib
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger("AnonymizationEngine")

class AnonymizationEngine:
    """
    Comprehensive anonymization engine that supports multiple strategies:
    - Full redaction with hash-based pseudonyms
    - Partial masking with format preservation
    - Custom anonymization rules per entity type
    - Batch processing capabilities
    """
    
    def __init__(self):
        self.presidio_anonymizer = None
        self.config = {}
        self.pseudonymize_types = set()
        self.entity_type_mapping = {}
        
        # Initialize components
        self._load_configuration()
        self._load_presidio_anonymizer()
        
        logger.info("AnonymizationEngine initialized successfully")
    
    def _load_configuration(self):
        """Load anonymization configuration"""
        base_dir = os.path.dirname(__file__)
        
        # Load configuration
        self.config = {
            "partial_mask_char": os.environ.get("PARTIAL_MASK_CHAR", "*"),
            "preserve_format": True,
            "default_strategy": "pseudonymize",  # pseudonymize, mask, redact
            "hash_algorithm": "md5",
            "hash_length": 6
        }
        
        # Load pseudonymization types
        try:
            pseudonymize_path = os.path.join(base_dir, 'pseudonymize_types.json')
            with open(pseudonymize_path, 'r', encoding='utf-8') as f:
                pseudonymize_data = json.load(f)
                self.pseudonymize_types = set(pseudonymize_data)
                logger.info(f"Loaded {len(self.pseudonymize_types)} pseudonymization types")
        except Exception as e:
            logger.warning(f"Could not load pseudonymization types: {e}")
            self.pseudonymize_types = set()
        
        # Load entity type mapping
        try:
            mapping_path = os.path.join(base_dir, 'entity_type_mapping.json')
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.entity_type_mapping = json.load(f)
                logger.info(f"Loaded {len(self.entity_type_mapping)} entity type mappings")
        except Exception as e:
            logger.warning(f"Could not load entity type mapping: {e}")
            self.entity_type_mapping = {}
    
    def _load_presidio_anonymizer(self):
        """Initialize Presidio Anonymizer"""
        try:
            self.presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio Anonymizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Presidio Anonymizer: {e}")
            self.presidio_anonymizer = None
    
    def anonymize_text(self, text: str, entities: List[Dict[str, Any]], 
                      strategy: str = "pseudonymize", 
                      preserve_format: bool = True,
                      custom_rules: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Anonymize text using detected entities and specified strategy.
        
        Args:
            text: Original text to anonymize
            entities: List of detected entities with positions and types
            strategy: Anonymization strategy ('pseudonymize', 'mask', 'redact', 'custom')
            preserve_format: Whether to preserve original format where possible
            custom_rules: Custom anonymization rules per entity type
            
        Returns:
            Dictionary containing anonymized text and metadata
        """
        if not text:
            return {
                "anonymized_text": "",
                "entities_processed": [],
                "strategy_used": strategy,
                "processing_metadata": {
                    "original_length": 0,
                    "anonymized_length": 0,
                    "entities_count": 0
                }
            }
        
        try:
            logger.info(f"Anonymizing text of length {len(text)} with {len(entities)} entities using strategy: {strategy}")
            
            # Sort entities by start position (reverse order for replacement)
            sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
            
            # Process entities and build anonymized text
            anonymized_text = text
            processed_entities = []
            
            for entity in sorted_entities:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                entity_type = entity.get('entity_group', 'UNKNOWN')
                original_text = text[start:end] if start < len(text) and end <= len(text) else ""
                
                if not original_text:
                    continue
                
                # Determine anonymization method
                anonymized_value = self._anonymize_entity(
                    original_text, entity_type, strategy, preserve_format, custom_rules
                )
                
                # Replace in text
                anonymized_text = anonymized_text[:start] + anonymized_value + anonymized_text[end:]
                
                # Track processed entity
                processed_entity = {
                    **entity,
                    'original_text': original_text,
                    'anonymized_text': anonymized_value,
                    'anonymization_method': self._get_anonymization_method(entity_type, strategy, custom_rules)
                }
                processed_entities.append(processed_entity)
                
                logger.debug(f"Anonymized {entity_type} '{original_text}' -> '{anonymized_value}'")
            
            # Calculate metadata
            metadata = {
                "original_length": len(text),
                "anonymized_length": len(anonymized_text),
                "entities_count": len(processed_entities),
                "strategy_used": strategy,
                "preserve_format": preserve_format,
                "custom_rules_applied": bool(custom_rules)
            }
            
            logger.info(f"Anonymization completed: {len(processed_entities)} entities processed")
            
            return {
                "anonymized_text": anonymized_text,
                "entities_processed": processed_entities,
                "strategy_used": strategy,
                "processing_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error during text anonymization: {e}", exc_info=True)
            return {
                "anonymized_text": text,  # Return original on error
                "entities_processed": [],
                "strategy_used": strategy,
                "error": str(e),
                "processing_metadata": {
                    "original_length": len(text),
                    "anonymized_length": len(text),
                    "entities_count": 0
                }
            }
    
    def _anonymize_entity(self, entity_text: str, entity_type: str, 
                         strategy: str, preserve_format: bool,
                         custom_rules: Optional[Dict[str, str]] = None) -> str:
        """Anonymize a single entity based on strategy and type"""
        
        # Check for custom rules first
        if custom_rules and entity_type in custom_rules:
            return custom_rules[entity_type]
        
        # Apply strategy-based anonymization
        if strategy == "pseudonymize":
            return self._pseudonymize_value(entity_text, entity_type)
        elif strategy == "mask":
            if preserve_format:
                return self._smart_partial_mask(entity_text, entity_type)
            else:
                return self._full_mask_token(entity_text, entity_type)
        elif strategy == "redact":
            return f"[REDACTED-{entity_type}]"
        elif strategy == "custom":
            # Use entity-type specific logic
            return self._custom_anonymize(entity_text, entity_type, preserve_format)
        else:
            # Default to pseudonymization
            return self._pseudonymize_value(entity_text, entity_type)
    
    def _get_anonymization_method(self, entity_type: str, strategy: str, 
                                 custom_rules: Optional[Dict[str, str]] = None) -> str:
        """Get the anonymization method used for an entity"""
        if custom_rules and entity_type in custom_rules:
            return "custom_rule"
        elif strategy == "pseudonymize":
            return "pseudonymize"
        elif strategy == "mask":
            return "partial_mask" if self.config.get("preserve_format") else "full_mask"
        elif strategy == "redact":
            return "redact"
        elif strategy == "custom":
            return f"custom_{entity_type.lower()}"
        else:
            return "pseudonymize"
    
    def _pseudonymize_value(self, value: str, entity_type: str) -> str:
        """Create a pseudonymized value using hash"""
        hash_algo = self.config.get("hash_algorithm", "md5")
        hash_length = self.config.get("hash_length", 6)
        
        if hash_algo == "md5":
            hash_obj = hashlib.md5(value.encode('utf-8'))
        elif hash_algo == "sha256":
            hash_obj = hashlib.sha256(value.encode('utf-8'))
        else:
            hash_obj = hashlib.md5(value.encode('utf-8'))
        
        hash_value = hash_obj.hexdigest()[:hash_length]
        return f"[{entity_type.upper()}-{hash_value}]"
    
    def _full_mask_token(self, token: str, entity_type: str) -> str:
        """Fully mask a token"""
        if entity_type and entity_type.upper() in self.pseudonymize_types:
            return self._pseudonymize_value(token, entity_type)
        return '*' * len(token)
    
    def _smart_partial_mask(self, text: str, entity_type: str) -> str:
        """Apply smart partial masking based on entity type"""
        mask_char = self.config.get("partial_mask_char", "*")
        
        if not text:
            return text
        
        entity_type_upper = entity_type.upper() if entity_type else ""
        
        if entity_type_upper == "EMAIL_ADDRESS":
            return self._mask_email(text)
        elif entity_type_upper == "PHONE_NUMBER":
            return self._mask_phone(text)
        elif entity_type_upper == "URL":
            return self._partial_mask_url(text)
        elif entity_type_upper == "CREDIT_CARD":
            return self._mask_credit_card(text)
        elif entity_type_upper == "SSN":
            return self._mask_ssn(text)
        elif entity_type_upper in ["PASSWORD", "API_KEY", "AUTHENTICATION_TOKEN"]:
            return self._mask_sensitive_token(text)
        elif entity_type_upper == "DATE_TIME":
            return self._mask_date(text)
        else:
            return self._partial_mask_token(text)
    
    def _mask_email(self, email: str) -> str:
        """Mask email address preserving format"""
        mask_char = self.config.get("partial_mask_char", "*")
        try:
            local, domain = email.split("@")
        except ValueError:
            return self._partial_mask_token(email)
        
        # Mask local part
        if len(local) > 4:
            local_masked = local[0:2] + mask_char * (len(local) - 4) + local[-2:]
        else:
            local_masked = local[0] + mask_char * (len(local) - 1)
        
        # Mask domain but preserve TLD
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            tld = domain_parts[-1]
            domain_name = '.'.join(domain_parts[:-1])
            if len(domain_name) > 5:
                domain_masked = domain_name[0:2] + mask_char * (len(domain_name) - 2)
            else:
                domain_masked = mask_char * len(domain_name)
            masked_domain = domain_masked + '.' + tld
        else:
            masked_domain = mask_char * len(domain)
        
        return local_masked + "@" + masked_domain
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number preserving format"""
        mask_char = self.config.get("partial_mask_char", "*")
        digits_only = re.sub(r'[^0-9+]', '', phone)
        
        if len(digits_only) <= 4:
            return mask_char * len(phone)
        
        # Handle international prefix
        if digits_only.startswith('+'):
            prefix_end = digits_only.find('9')
            if prefix_end != -1 and prefix_end < 4:
                prefix = digits_only[:prefix_end+1]
                main_number = digits_only[prefix_end+1:]
            else:
                prefix = '+'
                main_number = digits_only[1:]
        else:
            prefix = ''
            main_number = digits_only
        
        # Mask main number but keep last 4 digits
        if len(main_number) > 4:
            masked_number = mask_char * (len(main_number) - 4) + main_number[-4:]
        else:
            masked_number = mask_char * len(main_number)
        
        masked_digits = prefix + masked_number
        
        # Reconstruct with original formatting
        result = ''
        digit_index = 0
        for char in phone:
            if char.isdigit() or char == '+':
                if digit_index < len(masked_digits):
                    result += masked_digits[digit_index]
                    digit_index += 1
                else:
                    result += mask_char
            else:
                result += char
        
        return result
    
    def _mask_credit_card(self, card: str) -> str:
        """Mask credit card number preserving format"""
        mask_char = self.config.get("partial_mask_char", "*")
        digits_only = re.sub(r'[^0-9]', '', card)
        
        if len(digits_only) >= 4:
            masked_digits = mask_char * (len(digits_only) - 4) + digits_only[-4:]
        else:
            masked_digits = mask_char * len(digits_only)
        
        # Reconstruct with original formatting
        result = ''
        digit_index = 0
        for char in card:
            if char.isdigit():
                if digit_index < len(masked_digits):
                    result += masked_digits[digit_index]
                    digit_index += 1
                else:
                    result += mask_char
            else:
                result += char
        
        return result
    
    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN preserving format"""
        mask_char = self.config.get("partial_mask_char", "*")
        if len(ssn) > 4:
            return mask_char * (len(ssn) - 4) + ssn[-4:]
        else:
            return mask_char * len(ssn)
    
    def _mask_sensitive_token(self, token: str) -> str:
        """Mask sensitive tokens like passwords and API keys"""
        mask_char = self.config.get("partial_mask_char", "*")
        if len(token) > 8:
            return token[:2] + mask_char * (len(token) - 2)
        else:
            return mask_char * len(token)
    
    def _mask_date(self, date: str) -> str:
        """Mask date preserving some format"""
        mask_char = self.config.get("partial_mask_char", "*")
        if len(date) > 6 and re.search(r'\d{4}', date):
            # Mask year in dates
            date_parts = re.split(r'[-/\s:]', date)
            if len(date_parts) > 2 and len(date_parts[0]) == 4:
                date_parts[0] = mask_char * 4
                original_separators = re.findall(r'[-/\s:]', date)
                reconstructed = date_parts[0]
                for i, part in enumerate(date_parts[1:]):
                    if i < len(original_separators):
                        reconstructed += original_separators[i] + part
                    else:
                        reconstructed += "-" + part
                return reconstructed
            elif len(date_parts) > 2 and len(date_parts[-1]) == 4:
                date_parts[-1] = mask_char * 4
                return re.sub(r'\d{4}', mask_char * 4, date)
        
        return self._partial_mask_token(date)
    
    def _partial_mask_url(self, url: str) -> str:
        """Partially mask URL preserving structure"""
        mask_char = self.config.get("partial_mask_char", "*")
        try:
            parsed = urlparse(url)
        except Exception:
            return self._partial_mask_token(url)
        
        scheme, netloc, path, params, query, fragment = (
            parsed.scheme, parsed.netloc, parsed.path, 
            parsed.params, parsed.query, parsed.fragment
        )
        
        # Mask domain but preserve TLD
        if ':' in netloc:
            domain, port = netloc.split(':', 1)
            port = ':' + port
        else:
            domain, port = netloc, ''
        
        parts = domain.split('.')
        masked_parts = []
        for i, part in enumerate(parts):
            if i == len(parts) - 1 and len(parts) > 1:
                # Keep TLD
                masked_parts.append(part)
            elif len(part) > 3:
                masked_parts.append(part[0:2] + mask_char * (len(part) - 2))
            else:
                masked_parts.append(mask_char * len(part))
        
        masked_netloc = '.'.join(masked_parts) + port
        
        # Mask path segments
        if path:
            path_segments = path.split('/')
            masked_segments = []
            for segment in path_segments:
                if not segment:
                    masked_segments.append(segment)
                    continue
                
                # Keep common path segments
                if segment.lower() in ['api', 'v1', 'v2', 'v3', 'dashboard', 'login', 'public', 'static']:
                    masked_segments.append(segment)
                elif len(segment) >= 5:
                    masked_segments.append(segment[0:2] + mask_char * (len(segment) - 2))
                else:
                    masked_segments.append(mask_char * len(segment))
            
            masked_path = '/'.join(masked_segments)
        else:
            masked_path = path
        
        return urlunparse((scheme, masked_netloc, masked_path, params, query, fragment))
    
    def _partial_mask_token(self, token: str) -> str:
        """Apply partial masking to a generic token"""
        n = len(token)
        mask_char = self.config.get("partial_mask_char", "*")
        
        if n <= 2:
            return mask_char * n
        elif n <= 5:
            return token[0] + mask_char * (n - 1)
        elif n <= 10:
            return token[0:2] + mask_char * (n - 4) + token[-2:]
        else:
            return token[0:2] + mask_char * (n - 5) + token[-3:]
    
    def _custom_anonymize(self, entity_text: str, entity_type: str, preserve_format: bool) -> str:
        """Apply custom anonymization logic based on entity type"""
        entity_type_upper = entity_type.upper() if entity_type else ""
        
        # Custom logic for specific entity types
        if entity_type_upper in ["MEDICAL_RECORD_NUMBER", "PATIENT_ID"]:
            return f"[MRN-{hashlib.md5(entity_text.encode()).hexdigest()[:6]}]"
        elif entity_type_upper in ["CASE_NUMBER", "LEGAL_CITATION"]:
            return f"[CASE-{hashlib.md5(entity_text.encode()).hexdigest()[:6]}]"
        elif entity_type_upper in ["ACCOUNT_NUMBER", "ROUTING_NUMBER"]:
            return f"[ACCT-{hashlib.md5(entity_text.encode()).hexdigest()[:6]}]"
        elif entity_type_upper in ["API_KEY", "AUTHENTICATION_TOKEN"]:
            return f"[TOKEN-{hashlib.md5(entity_text.encode()).hexdigest()[:8]}]"
        else:
            # Default to pseudonymization
            return self._pseudonymize_value(entity_text, entity_type)
    
    def batch_anonymize(self, texts: List[str], entities_list: List[List[Dict[str, Any]]], 
                       strategy: str = "pseudonymize", 
                       preserve_format: bool = True,
                       custom_rules: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Anonymize multiple texts in batch.
        
        Args:
            texts: List of texts to anonymize
            entities_list: List of entity lists corresponding to each text
            strategy: Anonymization strategy
            preserve_format: Whether to preserve format
            custom_rules: Custom anonymization rules
            
        Returns:
            List of anonymization results
        """
        if len(texts) != len(entities_list):
            raise ValueError("Number of texts must match number of entity lists")
        
        results = []
        for i, (text, entities) in enumerate(zip(texts, entities_list)):
            logger.debug(f"Processing batch item {i+1}/{len(texts)}")
            result = self.anonymize_text(text, entities, strategy, preserve_format, custom_rules)
            results.append(result)
        
        logger.info(f"Batch anonymization completed: {len(results)} texts processed")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get anonymization engine statistics"""
        return {
            "presidio_anonymizer_loaded": self.presidio_anonymizer is not None,
            "pseudonymize_types_count": len(self.pseudonymize_types),
            "entity_type_mappings": len(self.entity_type_mapping),
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')},
            "supported_strategies": ["pseudonymize", "mask", "redact", "custom"],
            "supported_entity_types": list(self.pseudonymize_types) if self.pseudonymize_types else []
        }

# Global anonymization engine instance
_anonymization_engine: Optional[AnonymizationEngine] = None

def get_anonymization_engine() -> AnonymizationEngine:
    """Get the global anonymization engine instance"""
    global _anonymization_engine
    if _anonymization_engine is None:
        _anonymization_engine = AnonymizationEngine()
    return _anonymization_engine