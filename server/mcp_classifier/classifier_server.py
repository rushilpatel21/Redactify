import os
import logging
import time
import torch
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from transformers import pipeline
from dotenv import load_dotenv
import json

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPClassifierServer")

# --- Model Configuration ---
MODEL_NAME = os.environ.get("MCP_MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
CANDIDATE_LABELS = ["medical", "technical", "general"]
CONFIDENCE_THRESHOLD = float(os.environ.get("MCP_THRESHOLD", 0.50))

# --- Load model at module initialization ---
logger.info(f"Loading zero-shot model '{MODEL_NAME}'")
classifier_pipeline = None
try:
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading zero-shot model '{MODEL_NAME}' on device: {'GPU' if device == 0 else 'CPU'}")
    classifier_pipeline = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device
    )
    logger.info("Zero-shot classification pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load zero-shot model '{MODEL_NAME}': {e}", exc_info=True)
    classifier_pipeline = None

# --- MCP Server Setup ---
mcp = FastMCP(
    name="ZeroShotClassifier",
    version="1.0.0",
    description="Zero-shot text classifier for determining text categories"
)

# --- Classification Logic ---
def model_classify(text: str) -> List[str]:
    """Classifies text using the zero-shot model, returning a list of relevant categories."""
    if not classifier_pipeline:
        logger.error("Classifier pipeline not loaded. Cannot perform classification.")
        return ["general"]

    if not text:
        return ["general"]

    try:
        max_length = 512
        truncated_text = text[:max_length] if len(text) > max_length else text

        results = classifier_pipeline(
            truncated_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=True
        )

        scores = results['scores']
        labels = results['labels']
        classifications = []

        for label, score in zip(labels, scores):
            logger.debug(f"Label: {label}, Score: {score:.4f}")
            if score >= CONFIDENCE_THRESHOLD:
                classifications.append(label)

        if not classifications or classifications == ["general"]:
            if "general" not in classifications:
                classifications.append("general")
        else:
            if "general" in classifications and len(classifications) > 1:
                classifications.remove("general")

        return sorted(list(set(classifications)))

    except Exception as e:
        logger.error(f"Error during model classification: {e}", exc_info=True)
        return ["general"]

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify text using zero-shot classification.
    
    Args:
        inputs: The text to classify
        parameters: Optional parameters (not used currently)
        
    Returns:
        Dictionary containing classifications
    """
    start_time = time.time()
    text = inputs or ""
    classifications = model_classify(text)
    duration = time.time() - start_time
    logger.info(f"MCP predict tool: classified in {duration:.4f}s")
    return {"classifications": classifications}

@mcp.tool()
async def health_check() -> Dict[str, str]:
    """Check the health of the classifier service."""
    model_status = "loaded" if classifier_pipeline else "failed_to_load"
    return {
        "status": "ok", 
        "service": "ZeroShotClassifier", 
        "model_status": model_status
    }

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    
    port = int(os.environ.get("MCP_CLASSIFIER_PORT", 8001))
    logger.info(f"Starting Zero-Shot Text Classifier MCP Server on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="Zero-Shot Classifier MCP Server")
    
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