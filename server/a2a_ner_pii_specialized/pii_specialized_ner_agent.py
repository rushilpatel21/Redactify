import os
import logging
import time
import uuid
import json
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np

# --- Enhanced Logging Setup ---
load_dotenv()
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("A2ASpecializedPII")
logger.info(f"Logging initialized at {log_level} level")

# --- Model Configuration ---
MODEL_NAME = os.environ.get("A2A_PII_SPECIAL_MODEL", "1-13-am/xlm-roberta-base-pii-finetuned")
AGENT_ID = "a2a_ner_pii_specialized"

# --- Load model at module initialization ---
logger.info(f"[{AGENT_ID}] Starting model load: {MODEL_NAME}")
ner_pipeline = None
try:
    logger.info(f"[{AGENT_ID}] Step 1: Setting up model pipeline")
    start_time = time.time()
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    load_time = time.time() - start_time
    logger.info(f"[{AGENT_ID}] Step 2: Model loaded successfully in {load_time:.2f}s")
except Exception as e:
    logger.error(f"[{AGENT_ID}] CRITICAL ERROR: Failed to load model {MODEL_NAME}: {e}", exc_info=True)
    logger.error(f"[{AGENT_ID}] Model loading stack trace", stack_info=True)

# --- MCP Server Setup ---
logger.info(f"[{AGENT_ID}] Initializing FastMCP server")
mcp = FastMCP(
    name="PiiSpecializedNERAgent", 
    version="1.0.0",
    description="Specialized PII Named Entity Recognition agent"
)
logger.info(f"[{AGENT_ID}] FastMCP initialized")

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect specialized PII entities in text."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: predict function called")
    
    text = inputs or ""
    if not text:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text provided, returning empty result")
        return {"entities": []}
    
    if ner_pipeline is None:
        logger.error(f"[{AGENT_ID}][{request_id}] NER pipeline not loaded. Cannot perform entity detection.")
        return {"entities": []}

    logger.info(f"[{AGENT_ID}][{request_id}] Processing text of length {len(text)}")
    logger.debug(f"[{AGENT_ID}][{request_id}] Text snippet (first 100 chars): {text[:100]}")
    
    try:
        # Step 1: Entity detection
        logger.info(f"[{AGENT_ID}][{request_id}] Step 1: Starting entity detection")
        start_time = time.time()
        raw_results = ner_pipeline(text)
        detection_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Step 2: Detection completed in {detection_time:.2f}s")
        
        # Step 2: Process results
        logger.info(f"[{AGENT_ID}][{request_id}] Step 3: Processing {len(raw_results)} entities")
        processed_results = []
        
        for i, item in enumerate(raw_results):
            logger.debug(f"[{AGENT_ID}][{request_id}] Processing entity {i+1}/{len(raw_results)}")
            
            # Type conversion and validation
            if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                item['score'] = float(item['score'])
                logger.debug(f"[{AGENT_ID}][{request_id}] Entity {i+1} score: {item['score']:.4f}")
            
            if 'start' in item: 
                item['start'] = int(item['start'])
            if 'end' in item:
                item['end'] = int(item['end'])
                
            # Logging the entity type and text
            if 'entity_group' in item and 'start' in item and 'end' in item:
                entity_text = text[item['start']:item['end']] if 0 <= item['start'] < len(text) and item['end'] <= len(text) else "INVALID_SPAN"
                logger.debug(f"[{AGENT_ID}][{request_id}] Entity {i+1}: {item.get('entity_group', 'UNKNOWN')} '{entity_text}' ({item['start']}:{item['end']})")
            
            # Add detector information
            item['detector'] = AGENT_ID
            processed_results.append(item)
        
        total_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Step 4: Processing completed, returning {len(processed_results)} entities in {total_time:.2f}s")
        
        # Group entities by type for better logging
        entity_types = {}
        for item in processed_results:
            entity_type = item.get('entity_group', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        logger.info(f"[{AGENT_ID}][{request_id}] Entity types found: {entity_types}")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function completed successfully")
        
        return {"entities": processed_results}
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error during NER detection: {e}", exc_info=True)
        logger.error(f"[{AGENT_ID}][{request_id}] Stack trace", stack_info=True)
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function failed")
        return {"entities": []}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the PII Specialized NER agent service."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: health_check function called")
    
    status = {
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }
    
    logger.info(f"[{AGENT_ID}][{request_id}] Health check result: {status}")
    logger.info(f"[{AGENT_ID}][{request_id}] EXIT: health_check function completed")
    return status

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    
    # Service configuration
    port = int(os.environ.get("A2A_PII_SPECIALIZED_PORT", 3006))
    logger.info(f"[{AGENT_ID}] Starting server on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="PII Specialized NER MCP Server")
    
    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        """MCP JSON-RPC endpoint that processes requests and forwards to appropriate tools"""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{AGENT_ID}][{request_id}] Received MCP request")
        
        try:
            # Parse request JSON
            logger.info(f"[{AGENT_ID}][{request_id}] Parsing request JSON")
            data = await request.json()
            logger.debug(f"[{AGENT_ID}][{request_id}] Request data: {json.dumps(data, indent=2)}")
            
            # Check if this is a JSON-RPC request
            if "jsonrpc" in data and "method" in data:
                method = data["method"]
                params = data.get("params", {})
                json_rpc_id = data.get("id")
                
                logger.info(f"[{AGENT_ID}][{request_id}] Processing JSON-RPC method: {method}")
                
                # Call the appropriate tool
                if method == "predict" and "inputs" in params:
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling predict with text length: {len(params['inputs'])}")
                    start_time = time.time()
                    result = await predict(inputs=params["inputs"], parameters=params.get("parameters"))
                    duration = time.time() - start_time
                    logger.info(f"[{AGENT_ID}][{request_id}] predict completed in {duration:.2f}s, found {len(result.get('entities', []))} entities")
                    
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    logger.info(f"[{AGENT_ID}][{request_id}] Returning response")
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                elif method == "health_check":
                    logger.info(f"[{AGENT_ID}][{request_id}] Calling health_check")
                    result = await health_check()
                    response_data = {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": json_rpc_id
                    }
                    logger.info(f"[{AGENT_ID}][{request_id}] Returning health check response")
                    return Response(content=json.dumps(response_data), media_type="application/json")
                
                else:
                    # Method not found
                    logger.warning(f"[{AGENT_ID}][{request_id}] Method not found: {method}")
                    return Response(
                        content=json.dumps({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method {method} not found"
                            },
                            "id": json_rpc_id
                        }),
                        media_type="application/json"
                    )
            else:
                # Not a JSON-RPC request
                logger.warning(f"[{AGENT_ID}][{request_id}] Invalid JSON-RPC request")
                return Response(
                    content=json.dumps({
                        "error": "Invalid JSON-RPC request"
                    }),
                    status_code=400,
                    media_type="application/json"
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"[{AGENT_ID}][{request_id}] JSON decode error: {e}")
            return Response(
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    },
                    "id": None
                }),
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"[{AGENT_ID}][{request_id}] Error processing MCP request: {e}", exc_info=True)
            return Response(
                content=json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": None
                }),
                media_type="application/json"
            )
    
    # Add /health endpoint for basic monitoring
    @app.get("/health")
    async def health():
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{AGENT_ID}][{request_id}] Received health check request")
        status = await health_check()
        logger.info(f"[{AGENT_ID}][{request_id}] Returning health status: {status['status']}")
        return status
    
    # Start the server
    logger.info(f"[{AGENT_ID}] Server initialization complete")
    logger.info(f"[{AGENT_ID}] Starting uvicorn server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)