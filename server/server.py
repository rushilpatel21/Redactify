#!/usr/bin/env python3
"""
Redactify Server - Main Entry Point

This is the main server that provides PII detection and anonymization services
using the new MCP architecture with improved performance and accuracy.
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
from mcp_client import MCPClientManager, MCPServerConfig
from model_manager import get_model_manager
from auto_mcp_manager import get_auto_mcp_manager

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedactifyServer")

# Initialize FastAPI app
app = FastAPI(title="Redactify Server", version="2.0")
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
auto_mcp_manager = get_auto_mcp_manager()
mcp_client_manager: Optional[MCPClientManager] = None

logger.info("=== Redactify Server v2.0 ===")
logger.info("Using true MCP architecture with distributed services")

@app.on_event("startup")
async def startup_event():
    """Initialize MCP servers and client connections"""
    global mcp_client_manager
    
    logger.info("=== Starting Redactify MCP System ===")
    
    # Step 1: Automatically start all MCP servers
    logger.info("Step 1: Starting MCP servers automatically...")
    success = await auto_mcp_manager.start_all_servers(timeout=180.0)  # 3 minutes timeout
    
    if not success:
        logger.error("Failed to start MCP servers! Some functionality may be limited.")
        # Continue anyway - the system can work with just Presidio and regex
    else:
        logger.info("✓ All MCP servers started successfully")
    
    # Step 2: Create MCP client manager
    logger.info("Step 2: Initializing MCP client connections...")
    mcp_client_manager = MCPClientManager()
    await mcp_client_manager.__aenter__()
    
    # Add MCP server configurations
    mcp_servers = [
        MCPServerConfig("general", port=3001),
        MCPServerConfig("medical", port=3002),
        MCPServerConfig("technical", port=3003),
        MCPServerConfig("legal", port=3004),
        MCPServerConfig("financial", port=3005),
        MCPServerConfig("pii_specialized", port=3006),
    ]
    
    for server_config in mcp_servers:
        mcp_client_manager.add_server(server_config)
    
    # Step 3: Update detection engine to use MCP clients
    detection_engine.set_mcp_client_manager(mcp_client_manager)
    
    logger.info("✓ MCP client connections established")
    logger.info("✓ Server startup complete")
    logger.info("=== Redactify MCP System Ready ===")
    
    # Start monitoring MCP servers
    auto_mcp_manager.monitoring_task = asyncio.create_task(auto_mcp_manager.start_monitoring())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup MCP servers and client connections"""
    global mcp_client_manager
    
    logger.info("=== Shutting Down Redactify MCP System ===")
    
    # Close MCP client connections
    if mcp_client_manager:
        logger.info("Closing MCP client connections...")
        await mcp_client_manager.__aexit__(None, None, None)
        logger.info("✓ MCP client connections closed")
    
    # Shutdown all MCP servers
    logger.info("Shutting down MCP servers...")
    await auto_mcp_manager.shutdown_all_servers()
    logger.info("✓ All MCP servers stopped")
    
    logger.info("=== Shutdown Complete ===")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Redactify Server is running",
        "version": "2.0",
        "architecture": "MCP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }

@app.post("/anonymize")
async def anonymize_endpoint(request: Request):
    """
    Main anonymization endpoint
    """
    start_time = time.time()
    
    try:
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
        
        # Step 3: Prepare response
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

@app.post("/anonymize_batch")
async def anonymize_batch_endpoint(request: Request):
    """
    Batch anonymization endpoint for processing multiple texts efficiently
    """
    start_time = time.time()
    
    try:
        data = await request.json()
        texts = data.get("texts", [])
        options = data.get("options", {})
        full_redaction = data.get("full_redaction", True)
        
        logger.info(f"--- /anonymize_batch Request Received ---")
        logger.info(f"Number of texts: {len(texts)}")
        logger.info(f"Full redaction: {full_redaction}")
        
        if not texts or not isinstance(texts, list):
            return JSONResponse(
                status_code=400,
                content={"error": "No texts provided or invalid format"}
            )
        
        # Step 1: Batch detect entities using new MCP architecture
        logger.info("Starting batch entity detection...")
        batch_results = await detection_engine.detect_entities_batch(texts)
        
        # Step 2: Process each result
        strategy = "pseudonymize" if full_redaction else "mask"
        batch_responses = []
        
        for i, (entities, domains) in enumerate(batch_results):
            text = texts[i]
            
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
            
            # Anonymize text
            anonymization_result = anonymization_engine.anonymize_text(
                text=text,
                entities=entities,
                strategy=strategy,
                preserve_format=not full_redaction
            )
            
            # Prepare individual response
            batch_responses.append({
                "anonymized_text": anonymization_result["anonymized_text"],
                "entities": entities,
                "domains_detected": domains,
                "entities_processed": len(anonymization_result["entities_processed"]),
                "metadata": {
                    "total_entities": len(entities),
                    "domains_used": domains,
                    "detectors_used": list(set(e.get('detector', 'unknown') for e in entities))
                }
            })
        
        processing_time = time.time() - start_time
        
        response = {
            "results": batch_responses,
            "batch_size": len(texts),
            "total_processing_time": processing_time,
            "average_time_per_text": processing_time / len(texts) if texts else 0,
            "strategy_used": strategy
        }
        
        logger.info(f"Batch anonymization completed in {processing_time:.2f}s")
        logger.info(f"Average time per text: {processing_time / len(texts):.2f}s")
        logger.info(f"--- /anonymize_batch Request End ---")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in /anonymize_batch endpoint: {e}", exc_info=True)
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
            "model_stats": model_manager.get_model_stats(),
            "supported_entities": [
                "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", 
                "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS", 
                "URL", "DATE_TIME", "PASSWORD", "API_KEY"
            ],
            "anonymization_strategies": ["pseudonymize", "mask", "redact", "custom"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "components": {
                "detection_engine": "ok",
                "anonymization_engine": "ok",
                "model_manager": "ok",
                "mcp_servers": "checking..."
            }
        }
        
        # Check model manager
        try:
            stats = model_manager.get_model_stats()
            health_status["components"]["model_manager"] = "ok"
            health_status["loaded_models"] = stats.get("loaded_models", 0)
        except Exception as e:
            health_status["components"]["model_manager"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check MCP servers
        try:
            mcp_health = await auto_mcp_manager.check_all_health()
            healthy_servers = sum(1 for is_healthy in mcp_health.values() if is_healthy)
            total_servers = len(mcp_health)
            
            health_status["mcp_servers"] = {
                "healthy": healthy_servers,
                "total": total_servers,
                "servers": mcp_health
            }
            health_status["components"]["mcp_servers"] = f"{healthy_servers}/{total_servers} healthy"
            
            if healthy_servers == 0:
                health_status["status"] = "degraded"
            elif healthy_servers < total_servers:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["mcp_servers"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

@app.get("/mcp-status")
async def mcp_status():
    """Get detailed MCP server status"""
    try:
        server_status = auto_mcp_manager.get_server_status()
        health_results = await auto_mcp_manager.check_all_health()
        
        # Combine status and health information
        detailed_status = {}
        for name, status in server_status.items():
            detailed_status[name] = {
                **status,
                "healthy": health_results.get(name, False),
                "url": f"http://localhost:{status['port']}"
            }
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "servers": detailed_status,
            "summary": {
                "total": len(detailed_status),
                "running": sum(1 for s in detailed_status.values() if s["running"]),
                "healthy": sum(1 for s in detailed_status.values() if s["healthy"])
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    
    print(f"=== Redactify Server v2.0 ===")
    print(f"Architecture: MCP (Model Context Protocol)")
    print(f"Auto-Management: MCP servers will start automatically")
    print(f"Serving on: http://0.0.0.0:{port}")
    print(f"Debug mode: {debug}")
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"Max workers: {os.environ.get('MAX_WORKERS', '8')}")
    print(f"Max model memory: {os.environ.get('MAX_MODEL_MEMORY_MB', '4096')}MB")
    print(f"Gemini API: {'✓' if os.environ.get('GEMINI_API_KEY') else '✗'}")
    print(f"")
    print(f"MCP Servers (auto-started):")
    print(f"  • General NER:      http://localhost:3001")
    print(f"  • Medical NER:      http://localhost:3002")
    print(f"  • Technical NER:    http://localhost:3003")
    print(f"  • Legal NER:        http://localhost:3004")
    print(f"  • Financial NER:    http://localhost:3005")
    print(f"  • PII Specialized:  http://localhost:3006")
    print(f"")
    print(f"Health Check: http://localhost:{port}/health")
    print(f"MCP Status:   http://localhost:{port}/mcp-status")
    print(f"=============================")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=debug
    )