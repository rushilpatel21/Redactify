import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("A2ATechnicalNER")

app = FastAPI()

# --- Agent Configuration ---
MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"
AGENT_ID = "a2a_ner_technical"

# --- Load Model ---
ner_pipeline = None
try:
    logger.info(f"[{AGENT_ID}] Loading model: {MODEL_NAME}")
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    logger.info(f"[{AGENT_ID}] Model loaded successfully.")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_NAME}: {e}", exc_info=True)

# --- Model Context Definitions ---
class ModelContextRequest(BaseModel):
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    inputs: str
    parameters: Optional[Dict[str, Any]] = None

class ModelContextResponse(BaseModel):
    model_id: str
    model_version: str
    outputs: Dict[str, Any]

# --- API Endpoint (MCP Simulation) ---
@app.post("/predict", response_model=ModelContextResponse)
async def predict(request: ModelContextRequest):
    """Receives text and returns detected entities from this agent's model."""
    text = request.inputs or ""
    if ner_pipeline is None:
        raise HTTPException(503, "Model not available")

    if not text:
        return ModelContextResponse(
            model_id=request.model_id or AGENT_ID,
            model_version=request.model_version or MODEL_NAME,
            outputs={"entities": []}
        )

    logger.info(f"[{AGENT_ID}] Received detection request (text length: {len(text)}).")
    try:
        start_time = time.time()
        raw_results = ner_pipeline(text)
        duration = time.time() - start_time
        logger.info(f"[{AGENT_ID}] Detection completed in {duration:.2f}s, found {len(raw_results)} entities.")

        # --- FIX: Convert float32 scores and add detector ---
        processed_results = []
        for item in raw_results:
            # Ensure score exists and is a numeric type before casting
            if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                item['score'] = float(item['score'])  # Cast to standard float
            if 'start' in item: item['start'] = int(item['start'])
            if 'end' in item:   item['end']   = int(item['end'])
            item['detector'] = AGENT_ID
            processed_results.append(item)
        # --- End FIX ---

        return ModelContextResponse(
            model_id=request.model_id or AGENT_ID,
            model_version=request.model_version or MODEL_NAME,
            outputs={"entities": processed_results}
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER detection: {e}", exc_info=True)
        # Ensure the error message itself is serializable
        error_message = f"Detection failed in agent {AGENT_ID}: {str(e)}"
        raise HTTPException(500, error_message)

@app.get("/health", summary="Health Check")
async def health_check():
    """Basic health check for the agent."""
    return {
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME
    }

# --- Run (for development) ---
if __name__ == "__main__":
    port = int(os.environ.get("A2A_TECHNICAL_PORT", 8004))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)