"""
Centralized Model Management for Redactify MCP Server

This module provides efficient model loading, caching, and resource management
for all NER models and the text classifier.
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from transformers import pipeline
import google.generativeai as genai
import psutil
import threading

logger = logging.getLogger("ModelManager")

@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    model_path: str
    task: str
    domain: str
    priority: int = 1  # Higher priority models stay in memory longer
    max_memory_mb: int = 1024  # Maximum memory usage estimate

@dataclass
class ModelInfo:
    """Runtime information about a loaded model"""
    config: ModelConfig
    model: Any
    last_used: float
    load_time: float
    memory_usage_mb: float
    usage_count: int = 0

class ModelManager:
    """
    Centralized model manager with lazy loading, caching, and resource monitoring.
    
    Features:
    - Lazy loading: Models loaded only when first requested
    - LRU eviction: Least recently used models evicted when memory is low
    - Resource monitoring: Track memory usage and performance
    - Thread-safe: Safe for concurrent access
    """
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self._models: Dict[str, ModelInfo] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._lock = threading.RLock()
        self._gemini_model = None
        
        # Initialize model configurations
        self._setup_model_configs()
        
        # Initialize Gemini client
        self._setup_gemini_client()
        
        logger.info(f"ModelManager initialized with max memory: {max_memory_mb}MB")
    
    def _setup_model_configs(self):
        """Setup configurations for all available models"""
        base_dir = os.path.dirname(__file__)
        
        self._model_configs = {
            "general": ModelConfig(
                name="general",
                model_path=os.environ.get("A2A_GENERAL_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english"),
                task="ner",
                domain="general",
                priority=3,  # High priority - always used
                max_memory_mb=800
            ),
            "medical": ModelConfig(
                name="medical", 
                model_path=os.path.join(base_dir, "a2a_ner_medical", "fine_tuned_model"),
                task="ner",
                domain="medical",
                priority=2,
                max_memory_mb=900
            ),
            "technical": ModelConfig(
                name="technical",
                model_path=os.environ.get("A2A_TECHNICAL_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english"),
                task="ner", 
                domain="technical",
                priority=2,
                max_memory_mb=700
            ),
            "legal": ModelConfig(
                name="legal",
                model_path=os.environ.get("A2A_LEGAL_MODEL", "nlpaueb/legal-bert-base-uncased"),
                task="ner",
                domain="legal", 
                priority=1,
                max_memory_mb=800
            ),
            "financial": ModelConfig(
                name="financial",
                model_path=os.environ.get("A2A_FINANCIAL_MODEL", "ProsusAI/finbert"),
                task="ner",
                domain="financial",
                priority=1,
                max_memory_mb=750
            ),
            "pii_specialized": ModelConfig(
                name="pii_specialized",
                model_path=os.environ.get("A2A_PII_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english"),
                task="ner",
                domain="pii",
                priority=2,
                max_memory_mb=800
            )
        }
        
        logger.info(f"Configured {len(self._model_configs)} models")
    
    def _setup_gemini_client(self):
        """Initialize Gemini client for text classification"""
        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini classification will be disabled.")
            self._gemini_model = None
            return
        
        try:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self._gemini_model = None
    
    async def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The loaded model or None if loading failed
        """
        with self._lock:
            # Check if model is already loaded
            if model_name in self._models:
                model_info = self._models[model_name]
                model_info.last_used = time.time()
                model_info.usage_count += 1
                logger.debug(f"Retrieved cached model: {model_name}")
                return model_info.model
            
            # Check if model config exists
            if model_name not in self._model_configs:
                logger.error(f"Unknown model: {model_name}")
                return None
            
            # Load the model
            return await self._load_model(model_name)
    
    async def _load_model(self, model_name: str) -> Optional[Any]:
        """Load a model and add it to the cache"""
        config = self._model_configs[model_name]
        logger.info(f"Loading model: {model_name} from {config.model_path}")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Check if we need to free memory first
            await self._ensure_memory_available(config.max_memory_mb)
            
            # Load the model using appropriate wrapper
            model = await self._create_model_wrapper(model_name, config)
            
            if not model:
                logger.error(f"Failed to create model wrapper for {model_name}")
                return None
            
            load_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Create model info and cache it
            model_info = ModelInfo(
                config=config,
                model=model,
                last_used=time.time(),
                load_time=load_time,
                memory_usage_mb=memory_used
            )
            
            self._models[model_name] = model_info
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s, using {memory_used:.1f}MB")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            return None
    
    async def _create_model_wrapper(self, model_name: str, config: ModelConfig) -> Optional[Any]:
        """Create appropriate model wrapper based on domain"""
        try:
            if model_name == "general":
                from models.general_ner import create_general_ner_model
                wrapper = create_general_ner_model(config.model_path)
            elif model_name == "medical":
                from models.medical_ner import create_medical_ner_model
                wrapper = create_medical_ner_model(config.model_path)
            elif model_name == "technical":
                from models.technical_ner import create_technical_ner_model
                wrapper = create_technical_ner_model(config.model_path)
            elif model_name == "legal":
                from models.legal_ner import create_legal_ner_model
                wrapper = create_legal_ner_model(config.model_path)
            elif model_name == "financial":
                from models.financial_ner import create_financial_ner_model
                wrapper = create_financial_ner_model(config.model_path)
            elif model_name == "pii_specialized":
                from models.pii_specialized_ner import create_pii_specialized_ner_model
                wrapper = create_pii_specialized_ner_model(config.model_path)
            else:
                logger.error(f"Unknown model type: {model_name}")
                return None
            
            # Load the model
            if wrapper.load():
                return wrapper
            else:
                logger.error(f"Failed to load wrapper for {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model wrapper for {model_name}: {e}", exc_info=True)
            return None
    
    async def _ensure_memory_available(self, required_mb: int):
        """Ensure enough memory is available by evicting models if necessary"""
        current_usage = self._get_total_model_memory()
        available_memory = self.max_memory_mb - current_usage
        
        if available_memory >= required_mb:
            return  # Enough memory available
        
        logger.info(f"Need to free {required_mb - available_memory:.1f}MB memory")
        
        # Sort models by priority (lower first) and last used time
        models_by_eviction_priority = sorted(
            self._models.items(),
            key=lambda x: (x[1].config.priority, x[1].last_used)
        )
        
        freed_memory = 0
        for model_name, model_info in models_by_eviction_priority:
            if freed_memory >= (required_mb - available_memory):
                break
                
            logger.info(f"Evicting model: {model_name} (freed {model_info.memory_usage_mb:.1f}MB)")
            freed_memory += model_info.memory_usage_mb
            del self._models[model_name]
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_total_model_memory(self) -> float:
        """Get total memory used by loaded models"""
        return sum(model_info.memory_usage_mb for model_info in self._models.values())
    
    def get_gemini_model(self):
        """Get the Gemini model for text classification"""
        return self._gemini_model
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        with self._lock:
            stats = {
                "loaded_models": len(self._models),
                "total_memory_mb": self._get_total_model_memory(),
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization": self._get_total_model_memory() / self.max_memory_mb,
                "models": {}
            }
            
            for name, info in self._models.items():
                stats["models"][name] = {
                    "domain": info.config.domain,
                    "memory_mb": info.memory_usage_mb,
                    "load_time": info.load_time,
                    "usage_count": info.usage_count,
                    "last_used": info.last_used
                }
            
            return stats
    
    def get_available_models(self) -> List[str]:
        """Get list of all available model names"""
        return list(self._model_configs.keys())
    
    async def preload_models(self, model_names: List[str]):
        """Preload specified models for faster access"""
        logger.info(f"Preloading models: {model_names}")
        
        for model_name in model_names:
            if model_name in self._model_configs:
                await self.get_model(model_name)
            else:
                logger.warning(f"Cannot preload unknown model: {model_name}")
    
    async def cleanup(self):
        """Clean up resources"""
        with self._lock:
            logger.info(f"Cleaning up {len(self._models)} loaded models")
            self._models.clear()
            
            # Gemini doesn't need explicit cleanup
            pass

# Global model manager instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        max_memory = int(os.environ.get("MAX_MODEL_MEMORY_MB", "4096"))
        _model_manager = ModelManager(max_memory_mb=max_memory)
    return _model_manager