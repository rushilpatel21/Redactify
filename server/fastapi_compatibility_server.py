#!/usr/bin/env python3
"""
FastAPI Compatibility Server for Frontend Testing

This server provides the same API endpoints as the original server.py
but uses the new MCP server architecture internally.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Import our new MCP components
from detection_engine import get_detection_engine
from anonymization_engine import get_anonymization_engine
from model_manager import get_model_manager

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CompatibilityServer")

# Initialize FastAPI app
app = FastAPI(title="Redactify Compatibility Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONT_END_URL", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detection_engine = get_detection_engine()
anonymization_engine = get_anonymization_engine()
model_manager = get_model_manager()

logger.info("=== Redactify Compatibility Server v2.0 ===")
logger.info("Using new MCP architecture with FastAPI compatibility layer")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Redactify Compatibility Server",
        "version": "2.0",
        "architecture": "MCP with FastAPI compatibility",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check component health
        model_stats = model_manager.get_model_stats()
        detection_stats = detection_engine.get_stats()
        
        return {
            "status": "healthy",
            "components": {
                "detection_engine": "ok" if detection_stats["presidio_loaded"] else "error",
                "model_manager": "ok",
                "anonymization_engine": "ok"
            },
            "models": {
                "available": model_manager.get_available_models(),
                "loaded": list(model_stats.get("models", {}).keys())
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/anonymize")
async def anonymize_endpoint(request: Request):
    """
    Main anonymization endpoint - compatible with frontend
    """
    start_time = time.time()
    
    try:
        # Parse request
        data = await request.json()
        text = data.get("text", "")
        options = data.get("options", {})
        full_redaction = data.get("full_redaction", True)
        
        logger.info(f"--- /anonymize Request Received ---")
        logger.info(f"Text length: {len(text)}")
        logger.info(f"Full redaction: {full_redaction}")
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )
        
        # Step 1: Detect entities using new MCP architecture
        logger.info("Starting entity detection...")
        entities, domains = await detection_engine.detect_entities(text)
        
        # Filter entities based on options (if provided)
        if options:
            filtered_entities = []
            for entity in entities:
                entity_type = entity.get('entity_group', '').upper()
                # Map some entity types for compatibility
                if entity_type in ['PER', 'PERSON']:
                    entity_type = 'PERSON'
                elif entity_type in ['ORG', 'ORGANIZATION']:
                    entity_type = 'ORGANIZATION'
                elif entity_type in ['LOC', 'LOCATION']:
                    entity_type = 'LOCATION'
                
                # Check if this entity type is enabled
                if options.get(entity_type, True):
                    filtered_entities.append(entity)
            
            entities = filtered_entities
        
        logger.info(f"Found {len(entities)} entities after filtering")
        
        # Step 2: Anonymize text
        strategy = "pseudonymize" if full_redaction else "mask"
        anonymization_result = anonymization_engine.anonymize_text(
            text=text,
            entities=entities,
            strategy=strategy,
            preserve_format=not full_redaction
        )
        
        # Step 3: Prepare response in original format
        processing_time = time.time() - start_time
        
        response = {
            "anonymized_text": anonymization_result["anonymized_text"],
            "entities": entities,
            "processing_time": processing_time,
            "domains_detected": domains,
            "entities_processed": len(anonymization_result["entities_processed"]),
            "strategy_used": strategy,
            "metadata": {
                "total_entities": len(entities),
                "domains_used": domains,
                "detectors_used": list(set(e.get('detector', 'unknown') for e in entities))
            }
        }
        
        logger.info(f"Anonymization completed in {processing_time:.2f}s")
        logger.info(f"--- /anonymize Request End ---")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in /anonymize endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/detect")
async def detect_entities_endpoint(request: Request):
    """
    Entity detection endpoint - for debugging/testing
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )
        
        # Detect entities
        entities, domains = await detection_engine.detect_entities(text)
        
        return JSONResponse(content={
            "entities": entities,
            "domains_detected": domains,
            "total_entities": len(entities),
            "entity_types": list(set(e.get('entity_group', 'UNKNOWN') for e in entities))
        })
        
    except Exception as e:
        logger.error(f"Error in /detect endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/config")
async def get_config():
    """Get configuration information"""
    try:
        return {
            "version": "2.0",
            "architecture": "MCP",
            "available_models": model_manager.get_available_models(),
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", 
                "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS", 
                "URL", "DATE_TIME", "PASSWORD", "API_KEY"
            ],
            "anonymization_strategies": ["pseudonymize", "mask", "redact", "custom"]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Redactify Compatibility Server on port {port}")
    logger.info("This server provides FastAPI compatibility for the new MCP architecture")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )