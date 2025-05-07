import os
import logging
import time
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("A2AMedicalNER")

# --- Model Configuration ---
MODEL_1_NAME = os.environ.get("A2A_MEDICAL_MODEL1", "obi/deid_roberta_i2b2")
MODEL_2_NAME = os.environ.get("A2A_MEDICAL_MODEL2", "theekshana/deid-roberta-i2b2-NER-medical-reports")
AGENT_ID = "a2a_ner_medical"

# --- Load models at module initialization ---
logger.info(f"[{AGENT_ID}] Loading models")
pipelines = {}
models_loaded = False

try:
    logger.info(f"[{AGENT_ID}] Loading model 1: {MODEL_1_NAME}")
    pipelines['model1'] = pipeline("ner", model=MODEL_1_NAME, aggregation_strategy="simple")
    logger.info(f"[{AGENT_ID}] Model 1 loaded successfully.")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_1_NAME}: {e}", exc_info=True)

try:
    logger.info(f"[{AGENT_ID}] Loading model 2: {MODEL_2_NAME}")
    pipelines['model2'] = pipeline(
        "ner",
        model=MODEL_2_NAME,
        aggregation_strategy="simple",
        model_kwargs={"ignore_mismatched_sizes": True}
    )
    logger.info(f"[{AGENT_ID}] Model 2 loaded successfully (with mismatched sizes ignored).")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_2_NAME}: {e}", exc_info=True)

models_loaded = len(pipelines) > 0

# --- MCP Server Setup ---
mcp = FastMCP(
    name="MedicalNERAgent", 
    version="1.0.0",
    description="Medical Named Entity Recognition with dual models"
)

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect medical named entities using dual-model approach."""
    text = inputs or ""
    if not models_loaded:
        logger.warning(f"[{AGENT_ID}] No models loaded. Cannot perform entity detection.")
        return {"entities": []}

    logger.info(f"[{AGENT_ID}] Received prediction request (text length: {len(text)}).")
    all_results = []
    total_duration = 0

    try:
        start_time = time.time()
        if 'model1' in pipelines:
            try:
                results1_raw = pipelines['model1'](text)
                results1_processed = []
                for item in results1_raw:
                    if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                        item['score'] = float(item['score'])
                    if 'start' in item: item['start'] = int(item['start'])
                    if 'end' in item:   item['end']   = int(item['end'])
                    item['detector'] = f"{AGENT_ID}_model1"
                    results1_processed.append(item)
                all_results.extend(results1_processed)
                logger.debug(f"[{AGENT_ID}] Model 1 found {len(results1_processed)} entities.")
            except Exception as e:
                logger.error(f"[{AGENT_ID}] Error running model 1: {e}", exc_info=True)

        if 'model2' in pipelines:
            try:
                results2_raw = pipelines['model2'](text)
                results2_processed = []
                for item in results2_raw:
                    if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                        item['score'] = float(item['score'])
                    if 'start' in item: item['start'] = int(item['start'])
                    if 'end' in item:   item['end']   = int(item['end'])
                    item['detector'] = f"{AGENT_ID}_model2"
                    results2_processed.append(item)
                all_results.extend(results2_processed)
                logger.debug(f"[{AGENT_ID}] Model 2 found {len(results2_processed)} entities.")
            except Exception as e:
                logger.error(f"[{AGENT_ID}] Error running model 2: {e}", exc_info=True)

        total_duration = time.time() - start_time
        logger.info(f"[{AGENT_ID}] Prediction completed in {total_duration:.2f}s, found {len(all_results)} potential entities.")
        
        return {"entities": all_results}
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER prediction: {e}", exc_info=True)
        return {"entities": []}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the Medical NER agent service."""
    return {
        "status": "ok" if models_loaded else "error",
        "agent_id": AGENT_ID,
        "models_loaded_count": len(pipelines),
        "model_names": [MODEL_1_NAME, MODEL_2_NAME]
    }

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    import json

    port = int(os.environ.get("A2A_MEDICAL_PORT", 8003))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="MedicalNER MCP Server")
    
    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        """MCP JSON-RPC endpoint that processes requests and forwards to appropriate tools"""
        try:
            data = await request.json()
            # Check if this is a JSON-RPC request
            if "jsonrpc" in data and "method" in data:
                method = data["method"]
                params = data.get("params", {})
                request_id = data.get("id")
                
                # Call the appropriate tool
                if method == "predict" and "inputs" in params:
                    result = await predict(inputs=params["inputs"], parameters=params.get("parameters"))
                    return Response(
                        content=json.dumps({
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": request_id
                        }),
                        media_type="application/json"
                    )
                elif method == "health_check":
                    result = await health_check()
                    return Response(
                        content=json.dumps({
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": request_id
                        }),
                        media_type="application/json"
                    )
                else:
                    # Method not found
                    return Response(
                        content=json.dumps({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method {method} not found"
                            },
                            "id": request_id
                        }),
                        media_type="application/json"
                    )
            else:
                # Not a JSON-RPC request
                return Response(
                    content=json.dumps({
                        "error": "Invalid JSON-RPC request"
                    }),
                    status_code=400,
                    media_type="application/json"
                )
        except Exception as e:
            logger.error(f"Error processing MCP request: {e}", exc_info=True)
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
        status = await health_check()
        return status
    
    uvicorn.run(app, host="0.0.0.0", port=port)