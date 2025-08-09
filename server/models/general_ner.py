"""
General NER Model Wrapper

Converted from the original a2a_ner_general/general_ner_agent.py
Now works as an internal model component instead of a standalone service.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from transformers import pipeline
import numpy as np

logger = logging.getLogger("GeneralNER")

class GeneralNERModel:
    """
    General-purpose Named Entity Recognition model wrapper.
    
    This class encapsulates the BERT-based NER model that was previously
    running as a separate microservice.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.pipeline = None
        self.load_time = None
        self.is_loaded = False
        
        logger.info(f"GeneralNERModel initialized with model: {self.model_name}")
    
    def load(self) -> bool:
        """Load the NER model pipeline"""
        if self.is_loaded:
            return True
        
        try:
            logger.info(f"Loading general NER model: {self.model_name}")
            start_time = time.time()
            
            self.pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple",
                device=-1  # Use CPU for now
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"General NER model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load general NER model: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict named entities in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities with metadata
        """
        if not self.is_loaded:
            if not self.load():
                logger.error("Cannot predict - model not loaded")
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"Processing text of length {len(text)}")
            start_time = time.time()
            
            # Run NER prediction
            raw_results = self.pipeline(text)
            
            # Process results
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
                    
                    processed_entity = {
                        'entity_group': item.get('entity_group', 'UNKNOWN'),
                        'start': start_pos,
                        'end': end_pos,
                        'score': score,
                        'word': entity_text,
                        'detector': 'general_ner_model'
                    }
                    
                    processed_results.append(processed_entity)
                    
                    logger.debug(f"Found entity: {item.get('entity_group')} '{entity_text}' "
                               f"({start_pos}:{end_pos}) score={score:.3f}")
                else:
                    logger.warning(f"Invalid entity span: {start_pos}:{end_pos} for text length {len(text)}")
            
            processing_time = time.time() - start_time
            logger.debug(f"General NER processing completed in {processing_time:.3f}s, "
                        f"found {len(processed_results)} entities")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during general NER prediction: {e}", exc_info=True)
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "model_type": "transformers_ner",
            "domain": "general",
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "MISCELLANEOUS"
            ]
        }
    
    def unload(self):
        """Unload the model to free memory"""
        if self.is_loaded:
            logger.info("Unloading general NER model")
            self.pipeline = None
            self.is_loaded = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.is_loaded:
            self.unload()

# Factory function for model creation
def create_general_ner_model(model_name: str = None) -> GeneralNERModel:
    """Create and return a GeneralNERModel instance"""
    return GeneralNERModel(model_name)