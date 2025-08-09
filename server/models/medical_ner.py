"""
Medical NER Model Wrapper

Converted from the original a2a_ner_medical/medical_ner_agent.py
Specialized for medical entity recognition including patient information,
medical conditions, treatments, and healthcare-related PII.
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("MedicalNER")

class MedicalNERModel:
    """
    Medical Named Entity Recognition model wrapper.
    
    Specialized for detecting medical entities such as:
    - Patient names and identifiers
    - Medical conditions and diagnoses
    - Medications and treatments
    - Healthcare provider information
    - Medical record numbers
    - Dates related to medical care
    """
    
    def __init__(self, model_name: str = None):
        # Use fine-tuned medical model if available, otherwise fall back to general model
        base_dir = os.path.dirname(os.path.dirname(__file__))
        fine_tuned_path = os.path.join(base_dir, "a2a_ner_medical", "fine_tuned_model")
        
        if model_name:
            self.model_name = model_name
        elif os.path.exists(fine_tuned_path):
            self.model_name = fine_tuned_path
            logger.info(f"Using fine-tuned medical model: {fine_tuned_path}")
        else:
            # Fallback to a medical-focused model from HuggingFace
            self.model_name = os.environ.get("A2A_MEDICAL_MODEL", "obi/deid_roberta_i2b2")
            logger.info(f"Using medical model from HuggingFace: {self.model_name}")
        
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        logger.info(f"MedicalNERModel initialized with model: {self.model_name}")
    
    def load(self) -> bool:
        """Load the medical NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading medical NER model: {self.model_name}")
            start_time = time.time()
            
            # Try to load the model
            self.pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple",
                device=-1  # Use CPU for now
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Medical NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load medical NER model: {e}", exc_info=True)
            
            # Try fallback to general model if medical model fails
            if "fine_tuned_model" in str(self.model_name) or "obi/deid_roberta_i2b2" in str(self.model_name):
                logger.warning("Attempting fallback to general BERT model for medical NER")
                try:
                    self.model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                    self.pipeline = pipeline(
                        "ner", 
                        model=self.model_name, 
                        aggregation_strategy="simple",
                        device=-1
                    )
                    self.load_time = time.time() - start_time
                    self.is_loaded = True
                    logger.info(f"Fallback medical NER model loaded in {self.load_time:.2f}s")
                    return True
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
            
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict medical named entities in the given text.
        
        Args:
            text: Input text to analyze for medical entities
            
        Returns:
            List of detected medical entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - medical model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing medical text of length {len(text)}")
            start_time = time.time()
            
            # Run NER prediction
            raw_results = self.pipeline(text)
            
            # Process results with medical-specific enhancements
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
                    
                    # Map medical entity types to standard PII types
                    mapped_entity_group = self._map_medical_entity_type(entity_group, entity_text)
                    
                    processed_entity = {
                        'entity_group': mapped_entity_group,
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'medical_ner_model',
                        'original_label': entity_group  # Keep original for reference
                    }
                    
                    # Add medical-specific confidence boost for certain patterns
                    if self._is_medical_context(text, start_pos, end_pos):
                        processed_entity['score'] = min(1.0, score * 1.1)
                        processed_entity['medical_context'] = True
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found medical entity: {mapped_entity_group} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
                else:
                    logger.warning(f"Invalid medical entity span: {start_pos}:{end_pos} for text length {len(text)}")
            
            processing_time = time.time() - start_time
            logger.debug(f"Medical NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during medical NER prediction: {e}", exc_info=True)
            return []
    
    def _map_medical_entity_type(self, original_type: str, entity_text: str) -> str:
        """
        Map medical-specific entity types to standard PII categories.
        
        Args:
            original_type: Original entity type from the model
            entity_text: The actual entity text
            
        Returns:
            Mapped entity type
        """
        # Common medical entity mappings
        medical_mappings = {
            # Patient information
            'PATIENT': 'PERSON',
            'NAME': 'PERSON', 
            'PERSON': 'PERSON',
            
            # Medical identifiers
            'ID': 'MEDICAL_RECORD_NUMBER',
            'MEDICALRECORD': 'MEDICAL_RECORD_NUMBER',
            'MRN': 'MEDICAL_RECORD_NUMBER',
            'PATIENTID': 'MEDICAL_RECORD_NUMBER',
            
            # Healthcare providers
            'DOCTOR': 'PERSON',
            'PHYSICIAN': 'PERSON',
            'NURSE': 'PERSON',
            'PROVIDER': 'PERSON',
            
            # Organizations
            'HOSPITAL': 'ORGANIZATION',
            'CLINIC': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'ORG': 'ORGANIZATION',
            
            # Locations
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'CITY': 'LOCATION',
            'STATE': 'LOCATION',
            
            # Dates
            'DATE': 'DATE_TIME',
            'TIME': 'DATE_TIME',
            'AGE': 'AGE',
            
            # Contact information
            'PHONE': 'PHONE_NUMBER',
            'EMAIL': 'EMAIL_ADDRESS',
            
            # Medical conditions (keep as medical info)
            'CONDITION': 'MEDICAL_CONDITION',
            'DIAGNOSIS': 'MEDICAL_CONDITION',
            'MEDICATION': 'MEDICATION',
            'TREATMENT': 'TREATMENT',
            
            # Miscellaneous
            'MISC': 'MISCELLANEOUS',
            'MISCELLANEOUS': 'MISCELLANEOUS'
        }
        
        # Normalize the original type
        normalized_type = original_type.upper().strip()
        
        # Check for direct mapping
        if normalized_type in medical_mappings:
            return medical_mappings[normalized_type]
        
        # Pattern-based detection for entity text
        entity_upper = entity_text.upper()
        
        # Check for medical record number patterns
        if any(pattern in entity_upper for pattern in ['MRN', 'RECORD', 'PATIENT ID']):
            return 'MEDICAL_RECORD_NUMBER'
        
        # Check for medication patterns
        if any(pattern in entity_upper for pattern in ['MG', 'ML', 'TABLET', 'CAPSULE', 'DOSE']):
            return 'MEDICATION'
        
        # Default mapping
        return original_type if original_type else 'UNKNOWN'
    
    def _is_medical_context(self, text: str, start_pos: int, end_pos: int) -> bool:
        """
        Check if the entity appears in a medical context.
        
        Args:
            text: Full text
            start_pos: Entity start position
            end_pos: Entity end position
            
        Returns:
            True if entity appears in medical context
        """
        # Get context around the entity
        context_size = 50
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), end_pos + context_size)
        context = text[context_start:context_end].lower()
        
        # Medical context indicators
        medical_indicators = [
            'patient', 'doctor', 'physician', 'nurse', 'hospital', 'clinic',
            'medical', 'diagnosis', 'treatment', 'medication', 'prescription',
            'surgery', 'procedure', 'examination', 'test', 'lab', 'result',
            'condition', 'symptom', 'disease', 'illness', 'health', 'care',
            'record', 'chart', 'visit', 'appointment', 'admission', 'discharge',
            'mrn', 'dob', 'age', 'allergies', 'history'
        ]
        
        return any(indicator in context for indicator in medical_indicators)
    
    def get_info(self) -> Dict[str, Any]:
        """Get medical model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "medical_ner",
            "domain": "medical",
            "supported_entities": [
                "PERSON", "MEDICAL_RECORD_NUMBER", "ORGANIZATION", "LOCATION",
                "DATE_TIME", "AGE", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "MEDICAL_CONDITION", "MEDICATION", "TREATMENT"
            ],
            "specialization": "Healthcare and medical document PII detection"
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading medical NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_medical_ner_model(model_name: str = None) -> MedicalNERModel:
    """Create and return a MedicalNERModel instance"""
    return MedicalNERModel(model_name)