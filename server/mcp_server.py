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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedactifyMCPServer")

# Initialize global components
detection_engine = None
model_manager = None

def setup_server() -> Server:
    """Setup and configure the MCP server with all tools and resources"""
    global detection_engine, model_manager
    
    server = Server("redactify")
    
    # Initialize components
    detection_engine = get_detection_engine()
    model_manager = get_model_manager()
    
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
    
    # Tool 4: Model Management
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