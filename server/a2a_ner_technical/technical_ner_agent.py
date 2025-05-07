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
logger = logging.getLogger("A2ATechnicalNER")

# --- Model Configuration ---
MODEL_NAME = os.environ.get("A2A_TECHNICAL_MODEL", "Jean-Baptiste/roberta-large-ner-english")
AGENT_ID = "a2a_ner_technical"

# --- Load model at module initialization ---
logger.info(f"[{AGENT_ID}] Loading model: {MODEL_NAME}")
ner_pipeline = None
try:
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    logger.info(f"[{AGENT_ID}] Model loaded successfully.")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_NAME}: {e}", exc_info=True)

# --- MCP Server Setup ---
mcp = FastMCP(
    name="TechnicalNERAgent", 
    version="1.0.0",
    description="Technical Named Entity Recognition agent"
)

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect technical named entities in text."""
    text = inputs or ""
    if ner_pipeline is None:
        logger.warning(f"[{AGENT_ID}] NER pipeline not loaded. Cannot perform entity detection.")
        return {"entities": []}

    if not text:
        return {"entities": []}

    logger.info(f"[{AGENT_ID}] Detecting entities in text length {len(text)}")
    try:
        start_time = time.time()
        raw_results = ner_pipeline(text)
        duration = time.time() - start_time
        logger.info(f"[{AGENT_ID}] Detection completed in {duration:.2f}s, found {len(raw_results)} entities.")

        processed_results = []
        for item in raw_results:
            if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                item['score'] = float(item['score'])
            if 'start' in item: item['start'] = int(item['start'])
            if 'end' in item:   item['end']   = int(item['end'])
            item['detector'] = AGENT_ID
            processed_results.append(item)
        
        return {"entities": processed_results}
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER detection: {e}", exc_info=True)
        return {"entities": []}

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the Technical NER agent service."""
    return {
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME
    }

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    import json

    port = int(os.environ.get("A2A_TECHNICAL_PORT", 8004))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="TechnicalNER MCP Server")
    
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