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
logger = logging.getLogger("A2AMedicalNER")
logger.info(f"Logging initialized at {log_level} level")

# --- Model Configuration ---
MODEL_1_NAME = os.environ.get("A2A_MEDICAL_MODEL1", "obi/deid_roberta_i2b2")
MODEL_2_NAME = os.environ.get("A2A_MEDICAL_MODEL2", "theekshana/deid-roberta-i2b2-NER-medical-reports")
AGENT_ID = "a2a_ner_medical"

# --- Load models at module initialization ---
logger.info(f"[{AGENT_ID}] Starting models load")
pipelines = {}
models_loaded = False

try:
    logger.info(f"[{AGENT_ID}] Step 1: Loading model 1: {MODEL_1_NAME}")
    start_time = time.time()
    pipelines['model1'] = pipeline("ner", model=MODEL_1_NAME, aggregation_strategy="simple")
    load_time = time.time() - start_time
    logger.info(f"[{AGENT_ID}] Model 1 loaded successfully in {load_time:.2f}s")
except Exception as e:
    logger.error(f"[{AGENT_ID}] CRITICAL ERROR: Failed to load model {MODEL_1_NAME}: {e}", exc_info=True)
    logger.error(f"[{AGENT_ID}] Model 1 loading stack trace", stack_info=True)

try:
    logger.info(f"[{AGENT_ID}] Step 2: Loading model 2: {MODEL_2_NAME}")
    start_time = time.time()
    pipelines['model2'] = pipeline(
        "ner",
        model=MODEL_2_NAME,
        aggregation_strategy="simple",
        model_kwargs={"ignore_mismatched_sizes": True}
    )
    load_time = time.time() - start_time
    logger.info(f"[{AGENT_ID}] Model 2 loaded successfully in {load_time:.2f}s (with mismatched sizes ignored)")
except Exception as e:
    logger.error(f"[{AGENT_ID}] CRITICAL ERROR: Failed to load model {MODEL_2_NAME}: {e}", exc_info=True)
    logger.error(f"[{AGENT_ID}] Model 2 loading stack trace", stack_info=True)

models_loaded = len(pipelines) > 0
logger.info(f"[{AGENT_ID}] Models loading complete. Loaded {len(pipelines)}/2 models. Ready: {models_loaded}")

# --- MCP Server Setup ---
logger.info(f"[{AGENT_ID}] Initializing FastMCP server")
mcp = FastMCP(
    name="MedicalNERAgent", 
    version="1.0.0",
    description="Medical Named Entity Recognition with dual models"
)
logger.info(f"[{AGENT_ID}] FastMCP initialized")

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect medical named entities using dual-model approach."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: predict function called")
    
    text = inputs or ""
    if not text:
        logger.warning(f"[{AGENT_ID}][{request_id}] Empty text provided, returning empty result")
        return {"entities": []}
    
    if not pipelines:
        logger.error(f"[{AGENT_ID}][{request_id}] No NER pipelines loaded. Cannot perform entity detection.")
        return {"entities": []}

    logger.info(f"[{AGENT_ID}][{request_id}] Processing text of length {len(text)}")
    logger.debug(f"[{AGENT_ID}][{request_id}] Text snippet (first 100 chars): {text[:100]}")
    
    all_entities = []
    try:
        # Step 1: Detect with all models
        logger.info(f"[{AGENT_ID}][{request_id}] Step 1: Starting entity detection with {len(pipelines)} model(s)")
        start_time = time.time()
        
        # Run each model in sequence
        for model_name, ner_pipe in pipelines.items():
            model_start_time = time.time()
            logger.info(f"[{AGENT_ID}][{request_id}] Running {model_name}")
            try:
                results = ner_pipe(text)
                model_duration = time.time() - model_start_time
                logger.info(f"[{AGENT_ID}][{request_id}] {model_name} completed in {model_duration:.2f}s, found {len(results)} entities")
                
                # Process and normalize this model's results
                for item in results:
                    # Type conversion and validation
                    if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                        item['score'] = float(item['score'])
                    if 'start' in item: 
                        item['start'] = int(item['start'])
                    if 'end' in item:
                        item['end'] = int(item['end'])
                    
                    # Add source model and detector information
                    item['detector'] = f"{AGENT_ID}_{model_name}"
                    
                    # Logging the entity type and text
                    if 'entity_group' in item and 'start' in item and 'end' in item:
                        entity_text = text[item['start']:item['end']] if 0 <= item['start'] < len(text) and item['end'] <= len(text) else "INVALID_SPAN"
                        logger.debug(f"[{AGENT_ID}][{request_id}] {model_name} entity: {item.get('entity_group', 'UNKNOWN')} '{entity_text}' ({item['start']}:{item['end']})")
                    
                    all_entities.append(item)
                    
            except Exception as model_error:
                logger.error(f"[{AGENT_ID}][{request_id}] Error with {model_name}: {model_error}", exc_info=True)
        
        detection_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Step 2: Detection completed in {detection_time:.2f}s across all models")
        
        # Step 3: Process and merge results
        logger.info(f"[{AGENT_ID}][{request_id}] Step 3: Consolidating {len(all_entities)} total entities")
        
        # Group entities by type for better logging
        entity_types = {}
        for item in all_entities:
            entity_type = item.get('entity_group', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
        
        total_time = time.time() - start_time
        logger.info(f"[{AGENT_ID}][{request_id}] Entity types found: {entity_types}")
        logger.info(f"[{AGENT_ID}][{request_id}] Step 4: Processing completed in {total_time:.2f}s")
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function completed successfully")
        
        return {"entities": all_entities}
    except Exception as e:
        logger.error(f"[{AGENT_ID}][{request_id}] Error during NER prediction: {e}", exc_info=True)
        logger.error(f"[{AGENT_ID}][{request_id}] Stack trace", stack_info=True)
        logger.info(f"[{AGENT_ID}][{request_id}] EXIT: predict function failed")
        return {"entities": []}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the Medical NER agent service."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{AGENT_ID}][{request_id}] ENTRY: health_check function called")
    
    status = {
        "status": "ok" if models_loaded else "error",
        "agent_id": AGENT_ID,
        "models_loaded_count": len(pipelines),
        "model_names": [MODEL_1_NAME, MODEL_2_NAME],
        "loaded_models": list(pipelines.keys()),
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
    port = int(os.environ.get("A2A_MEDICAL_PORT", 8003))
    logger.info(f"[{AGENT_ID}] Starting server on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="Medical NER MCP Server")
    
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