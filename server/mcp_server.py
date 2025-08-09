"""
Redactify MCP Server

A proper Model Context Protocol server that provides PII detection and anonymization
tools using specialized NER models and detection engines.
"""

import asyncio
import logging
import json
from typing import Any, Sequence
from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    LoggingLevel
)
import mcp.server.stdio
from detection_engine import get_detection_engine
from model_manager import get_model_manager
from anonymization_engine import get_anonymization_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedactifyMCPServer")

# Initialize global components
detection_engine = None
model_manager = None
anonymization_engine = None

def setup_server() -> Server:
    """Setup and configure the MCP server with all tools and resources"""
    global detection_engine, model_manager, anonymization_engine
    
    server = Server("redactify")
    
    # Initialize components
    detection_engine = get_detection_engine()
    model_manager = get_model_manager()
    anonymization_engine = get_anonymization_engine()
    
    logger.info("Redactify MCP Server initializing...")
    
    # Tool 1: Comprehensive PII Detection
    @server.call_tool()
    async def detect_pii(arguments: dict) -> Sequence[TextContent]:
        """
        Detect PII in text using multiple specialized models and detection methods.
        
        Args:
            text (str): The text to analyze for PII
            domains (list, optional): Specific domains to focus on (medical, technical, legal, financial, general)
            confidence_threshold (float, optional): Minimum confidence score for entities (0.0-1.0)
            include_context (bool, optional): Whether to include contextual information
            
        Returns:
            Detailed PII detection results with entities, confidence scores, and metadata
        """
        try:
            # Extract arguments
            text = arguments.get("text", "")
            domains = arguments.get("domains")
            confidence_threshold = arguments.get("confidence_threshold")
            include_context = arguments.get("include_context", True)
            
            if not text:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "No text provided for analysis",
                        "entities": [],
                        "processing_time": 0
                    }, indent=2)
                )]
            
            logger.info(f"Processing PII detection for text of length {len(text)}")
            
            # Override confidence threshold if provided
            if confidence_threshold is not None:
                original_threshold = detection_engine.config.get("entity_confidence_threshold")
                detection_engine.config["entity_confidence_threshold"] = confidence_threshold
            
            # Perform detection
            entities, domains_used = await detection_engine.detect_entities(text, domains)
            
            # Restore original threshold
            if confidence_threshold is not None and 'original_threshold' in locals():
                detection_engine.config["entity_confidence_threshold"] = original_threshold
            
            # Prepare response
            result = {
                "entities": entities,
                "domains_used": domains_used,
                "total_entities": len(entities),
                "entity_types": list(set(e.get("entity_group", "UNKNOWN") for e in entities)),
                "processing_metadata": {
                    "text_length": len(text),
                    "confidence_threshold": confidence_threshold or detection_engine.config.get("entity_confidence_threshold"),
                    "domains_requested": domains,
                    "domains_used": domains_used,
                    "include_context": include_context
                }
            }
            
            # Add context information if requested
            if include_context:
                result["context_info"] = {
                    "model_stats": model_manager.get_model_stats(),
                    "detection_stats": detection_engine.get_stats()
                }
            
            logger.info(f"PII detection completed: {len(entities)} entities found")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in detect_pii: {e}", exc_info=True)
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "error": f"Detection failed: {str(e)}",
                    "entities": [],
                    "processing_time": 0
                }, indent=2)
            )]
    
    # Tool 2: Text Classification
    @server.call_tool()
    async def classify_text(arguments: dict) -> Sequence[TextContent]:
        """
        Classify text to determine relevant domains for specialized PII detection.
        
        Args:
            text (str): The text to classify
            categories (list, optional): Specific categories to consider
            
        Returns:
            Classification results with confidence scores and recommended models
        """
        try:
            text = arguments.get("text", "")
            categories = arguments.get("categories")
            
            if not text:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "No text provided for classification",
                        "classifications": ["general"]
                    }, indent=2)
                )]
            
            logger.info(f"Classifying text of length {len(text)}")
            
            # Use the detection engine's classification method
            classifications = await detection_engine._classify_text(text)
            
            # Prepare response
            result = {
                "classifications": classifications,
                "primary_domain": classifications[0] if classifications else "general",
                "recommended_models": [],
                "processing_metadata": {
                    "text_length": len(text),
                    "categories_requested": categories,
                    "available_categories": ["medical", "technical", "legal", "financial", "general"]
                }
            }
            
            # Add recommended models based on classifications
            for domain in classifications:
                if domain in model_manager.get_available_models():
                    result["recommended_models"].append(domain)
            
            logger.info(f"Text classified as: {classifications}")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in classify_text: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Classification failed: {str(e)}",
                    "classifications": ["general"]
                }, indent=2)
            )]
    
    # Tool 3: System Health Check
    @server.call_tool()
    async def health_check(arguments: dict) -> Sequence[TextContent]:
        """
        Check the health and status of all system components.
        
        Returns:
            Comprehensive health status including model availability and system resources
        """
        try:
            logger.info("Performing system health check")
            
            # Get component status
            model_stats = model_manager.get_model_stats()
            detection_stats = detection_engine.get_stats()
            
            # Check model availability
            available_models = model_manager.get_available_models()
            loaded_models = list(model_stats.get("models", {}).keys())
            
            health_status = {
                "status": "healthy",
                "timestamp": detection_stats.get("timestamp"),
                "components": {
                    "detection_engine": {
                        "status": "ok" if detection_stats["presidio_loaded"] else "error",
                        "presidio_loaded": detection_stats["presidio_loaded"],
                        "regex_patterns": detection_stats["regex_patterns"],
                        "blocklist_size": detection_stats["blocklist_size"]
                    },
                    "model_manager": {
                        "status": "ok",
                        "loaded_models": len(loaded_models),
                        "available_models": len(available_models),
                        "memory_usage_mb": model_stats.get("total_memory_mb", 0),
                        "memory_utilization": f"{model_stats.get('memory_utilization', 0)*100:.1f}%"
                    }
                },
                "models": {
                    "available": available_models,
                    "loaded": loaded_models,
                    "details": model_stats.get("models", {})
                },
                "configuration": {
                    "confidence_threshold": detection_stats["config"].get("confidence_threshold"),
                    "max_workers": detection_stats["config"].get("max_workers"),
                    "enabled_features": {
                        "medical_pii": detection_stats["config"].get("enable_medical_pii"),
                        "technical_ner": detection_stats["config"].get("enable_technical_ner"),
                        "legal_ner": detection_stats["config"].get("enable_legal_ner"),
                        "financial_ner": detection_stats["config"].get("enable_financial_ner"),
                        "pii_specialized": detection_stats["config"].get("enable_pii_specialized")
                    }
                }
            }
            
            # Determine overall status
            if not detection_stats["presidio_loaded"]:
                health_status["status"] = "degraded"
                health_status["issues"] = ["Presidio Analyzer not loaded"]
            
            logger.info(f"Health check completed: {health_status['status']}")
            
            return [TextContent(
                type="text",
                text=json.dumps(health_status, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in health_check: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": f"Health check failed: {str(e)}"
                }, indent=2)
            )]
    
    # Tool 4: Text Anonymization
    @server.call_tool()
    async def anonymize_text(arguments: dict) -> Sequence[TextContent]:
        """
        Anonymize text using detected entities with various strategies.
        
        Args:
            text (str): The text to anonymize
            entities (list, optional): Pre-detected entities (if not provided, will detect automatically)
            strategy (str, optional): Anonymization strategy ('pseudonymize', 'mask', 'redact', 'custom')
            preserve_format (bool, optional): Whether to preserve original format
            custom_rules (dict, optional): Custom anonymization rules per entity type
            
        Returns:
            Anonymized text with metadata about the anonymization process
        """
        try:
            # Extract arguments
            text = arguments.get("text", "")
            entities = arguments.get("entities")
            strategy = arguments.get("strategy", "pseudonymize")
            preserve_format = arguments.get("preserve_format", True)
            custom_rules = arguments.get("custom_rules")
            
            if not text:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "No text provided for anonymization",
                        "anonymized_text": "",
                        "entities_processed": []
                    }, indent=2)
                )]
            
            logger.info(f"Processing anonymization for text of length {len(text)} with strategy: {strategy}")
            
            # If entities not provided, detect them first
            if entities is None:
                logger.info("No entities provided, detecting entities first")
                detected_entities, domains_used = await detection_engine.detect_entities(text)
                entities = detected_entities
            
            # Perform anonymization
            result = anonymization_engine.anonymize_text(
                text=text,
                entities=entities,
                strategy=strategy,
                preserve_format=preserve_format,
                custom_rules=custom_rules
            )
            
            # Add additional metadata
            result["processing_metadata"]["domains_used"] = getattr(detection_engine, '_last_domains_used', [])
            result["processing_metadata"]["detection_performed"] = arguments.get("entities") is None
            
            logger.info(f"Anonymization completed: {len(result['entities_processed'])} entities processed")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in anonymize_text: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Anonymization failed: {str(e)}",
                    "anonymized_text": arguments.get("text", ""),
                    "entities_processed": []
                }, indent=2)
            )]
    
    # Tool 5: Entity Verification
    @server.call_tool()
    async def verify_entities(arguments: dict) -> Sequence[TextContent]:
        """
        Verify and validate detected entities against the original text.
        
        Args:
            text (str): The original text
            entities (list): List of entities to verify
            confidence_threshold (float, optional): Minimum confidence for verification
            
        Returns:
            Verification results with entity validation status
        """
        try:
            text = arguments.get("text", "")
            entities = arguments.get("entities", [])
            confidence_threshold = arguments.get("confidence_threshold", 0.5)
            
            if not text or not entities:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Both text and entities are required for verification",
                        "verified_entities": [],
                        "verification_summary": {}
                    }, indent=2)
                )]
            
            logger.info(f"Verifying {len(entities)} entities against text of length {len(text)}")
            
            verified_entities = []
            verification_stats = {
                "total_entities": len(entities),
                "valid_entities": 0,
                "invalid_entities": 0,
                "low_confidence_entities": 0,
                "out_of_bounds_entities": 0
            }
            
            for entity in entities:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                score = entity.get('score', 0)
                entity_type = entity.get('entity_group', 'UNKNOWN')
                
                # Verify entity bounds
                if start < 0 or end > len(text) or start >= end:
                    verification_stats["out_of_bounds_entities"] += 1
                    verified_entity = {
                        **entity,
                        'verification_status': 'invalid',
                        'verification_reason': 'out_of_bounds',
                        'is_valid': False
                    }
                # Verify confidence
                elif score < confidence_threshold:
                    verification_stats["low_confidence_entities"] += 1
                    verified_entity = {
                        **entity,
                        'verification_status': 'low_confidence',
                        'verification_reason': f'confidence {score:.3f} below threshold {confidence_threshold}',
                        'is_valid': False,
                        'extracted_text': text[start:end]
                    }
                else:
                    verification_stats["valid_entities"] += 1
                    verified_entity = {
                        **entity,
                        'verification_status': 'valid',
                        'verification_reason': 'passed_all_checks',
                        'is_valid': True,
                        'extracted_text': text[start:end]
                    }
                
                verified_entities.append(verified_entity)
            
            # Calculate verification summary
            verification_summary = {
                **verification_stats,
                "validation_rate": verification_stats["valid_entities"] / verification_stats["total_entities"] if verification_stats["total_entities"] > 0 else 0,
                "confidence_threshold_used": confidence_threshold
            }
            
            result = {
                "verified_entities": verified_entities,
                "verification_summary": verification_summary,
                "text_length": len(text),
                "processing_metadata": {
                    "verification_performed": True,
                    "entities_verified": len(verified_entities)
                }
            }
            
            logger.info(f"Entity verification completed: {verification_stats['valid_entities']}/{verification_stats['total_entities']} entities valid")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in verify_entities: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Entity verification failed: {str(e)}",
                    "verified_entities": [],
                    "verification_summary": {}
                }, indent=2)
            )]
    
    # Tool 6: Batch Processing
    @server.call_tool()
    async def process_batch(arguments: dict) -> Sequence[TextContent]:
        """
        Process multiple texts in batch for detection, classification, or anonymization.
        
        Args:
            texts (list): List of texts to process
            operation (str): Operation to perform ('detect', 'classify', 'anonymize')
            options (dict, optional): Operation-specific options
            
        Returns:
            Batch processing results for all texts
        """
        try:
            texts = arguments.get("texts", [])
            operation = arguments.get("operation", "detect")
            options = arguments.get("options", {})
            
            if not texts:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "No texts provided for batch processing",
                        "results": []
                    }, indent=2)
                )]
            
            logger.info(f"Processing batch of {len(texts)} texts with operation: {operation}")
            
            results = []
            
            for i, text in enumerate(texts):
                logger.debug(f"Processing batch item {i+1}/{len(texts)}")
                
                try:
                    if operation == "detect":
                        entities, domains = await detection_engine.detect_entities(text, options.get("domains"))
                        result = {
                            "text_index": i,
                            "entities": entities,
                            "domains_used": domains,
                            "total_entities": len(entities),
                            "status": "success"
                        }
                    
                    elif operation == "classify":
                        classifications = await detection_engine._classify_text(text)
                        result = {
                            "text_index": i,
                            "classifications": classifications,
                            "primary_domain": classifications[0] if classifications else "general",
                            "status": "success"
                        }
                    
                    elif operation == "anonymize":
                        # First detect entities if not provided
                        entities = options.get("entities", {}).get(str(i))
                        if entities is None:
                            entities, _ = await detection_engine.detect_entities(text)
                        
                        # Then anonymize
                        anonymization_result = anonymization_engine.anonymize_text(
                            text=text,
                            entities=entities,
                            strategy=options.get("strategy", "pseudonymize"),
                            preserve_format=options.get("preserve_format", True),
                            custom_rules=options.get("custom_rules")
                        )
                        
                        result = {
                            "text_index": i,
                            **anonymization_result,
                            "status": "success"
                        }
                    
                    else:
                        result = {
                            "text_index": i,
                            "error": f"Unknown operation: {operation}",
                            "status": "error"
                        }
                    
                except Exception as item_error:
                    logger.error(f"Error processing batch item {i}: {item_error}")
                    result = {
                        "text_index": i,
                        "error": str(item_error),
                        "status": "error"
                    }
                
                results.append(result)
            
            # Calculate batch summary
            successful_items = sum(1 for r in results if r.get("status") == "success")
            batch_summary = {
                "total_texts": len(texts),
                "successful_items": successful_items,
                "failed_items": len(texts) - successful_items,
                "success_rate": successful_items / len(texts) if texts else 0,
                "operation_performed": operation
            }
            
            batch_result = {
                "results": results,
                "batch_summary": batch_summary,
                "processing_metadata": {
                    "batch_size": len(texts),
                    "operation": operation,
                    "options_used": options
                }
            }
            
            logger.info(f"Batch processing completed: {successful_items}/{len(texts)} items successful")
            
            return [TextContent(
                type="text",
                text=json.dumps(batch_result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in process_batch: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Batch processing failed: {str(e)}",
                    "results": []
                }, indent=2)
            )]
    
    # Tool 7: Model Management
    @server.call_tool()
    async def manage_models(arguments: dict) -> Sequence[TextContent]:
        """
        Manage model loading, unloading, and preloading operations.
        
        Args:
            action (str): Action to perform (preload, stats, available)
            models (list, optional): List of model names for preload action
            
        Returns:
            Results of the model management operation
        """
        try:
            action = arguments.get("action", "stats")
            models = arguments.get("models", [])
            
            logger.info(f"Model management action: {action}")
            
            if action == "preload":
                if not models:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "No models specified for preloading",
                            "available_models": model_manager.get_available_models()
                        }, indent=2)
                    )]
                
                await model_manager.preload_models(models)
                result = {
                    "action": "preload",
                    "models_requested": models,
                    "status": "completed",
                    "current_stats": model_manager.get_model_stats()
                }
                
            elif action == "stats":
                result = {
                    "action": "stats",
                    "model_stats": model_manager.get_model_stats()
                }
                
            elif action == "available":
                result = {
                    "action": "available",
                    "available_models": model_manager.get_available_models(),
                    "loaded_models": list(model_manager.get_model_stats().get("models", {}).keys())
                }
                
            else:
                result = {
                    "error": f"Unknown action: {action}",
                    "available_actions": ["preload", "stats", "available"]
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in manage_models: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Model management failed: {str(e)}"
                }, indent=2)
            )]
    
    # List available tools
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools"""
        return [
            Tool(
                name="detect_pii",
                description="Detect PII in text using multiple specialized models and detection methods",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze for PII"
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific domains to focus on (medical, technical, legal, financial, general)"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum confidence score for entities"
                        },
                        "include_context": {
                            "type": "boolean",
                            "description": "Whether to include contextual information"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="classify_text",
                description="Classify text to determine relevant domains for specialized PII detection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to classify"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific categories to consider"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="anonymize_text",
                description="Anonymize text using detected entities with various strategies",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to anonymize"
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Pre-detected entities (if not provided, will detect automatically)"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["pseudonymize", "mask", "redact", "custom"],
                            "description": "Anonymization strategy"
                        },
                        "preserve_format": {
                            "type": "boolean",
                            "description": "Whether to preserve original format"
                        },
                        "custom_rules": {
                            "type": "object",
                            "description": "Custom anonymization rules per entity type"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="verify_entities",
                description="Verify and validate detected entities against the original text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The original text"
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of entities to verify"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum confidence for verification"
                        }
                    },
                    "required": ["text", "entities"]
                }
            ),
            Tool(
                name="process_batch",
                description="Process multiple texts in batch for detection, classification, or anonymization",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to process"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["detect", "classify", "anonymize"],
                            "description": "Operation to perform"
                        },
                        "options": {
                            "type": "object",
                            "description": "Operation-specific options"
                        }
                    },
                    "required": ["texts", "operation"]
                }
            ),
            Tool(
                name="health_check",
                description="Check the health and status of all system components",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="manage_models",
                description="Manage model loading, unloading, and preloading operations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["preload", "stats", "available"],
                            "description": "Action to perform"
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model names for preload action"
                        }
                    },
                    "required": ["action"]
                }
            )
        ]
    
    logger.info("Redactify MCP Server setup completed")
    return server

async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Redactify MCP Server...")
    
    try:
        server = setup_server()
        
        # Run the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, 
                write_stream, 
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if model_manager:
            await model_manager.cleanup()
        logger.info("Redactify MCP Server shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())