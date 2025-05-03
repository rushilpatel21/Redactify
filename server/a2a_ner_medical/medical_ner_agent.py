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
logger = logging.getLogger("A2AMedicalNER")

app = FastAPI()

# --- Agent Configuration ---
MODEL_1_NAME = "obi/deid_roberta_i2b2"
MODEL_2_NAME = "theekshana/deid-roberta-i2b2-NER-medical-reports"
AGENT_ID = "a2a_ner_medical"

# --- Local Definitions ---
class ModelContextRequest(BaseModel):
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    inputs: str
    parameters: Optional[Dict[str, Any]] = None

class ModelContextResponse(BaseModel):
    model_id: str
    model_version: str
    outputs: Dict[str, Any]

# --- Load Models ---
pipelines = {}
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

# --- API Endpoint (MCP Simulation) ---
@app.post("/predict", response_model=ModelContextResponse)
async def predict(request: ModelContextRequest):
    """Receives text and returns detected entities from medical models."""
    text = request.inputs or ""
    if not models_loaded:
        raise HTTPException(503, "No models available")

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

        return ModelContextResponse(
            model_id=request.model_id or AGENT_ID,
            model_version=request.model_version or "multi",
            outputs={"entities": all_results}
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER prediction: {e}", exc_info=True)
        error_message = f"Prediction failed in agent {AGENT_ID}: {str(e)}"
        raise HTTPException(500, error_message)

@app.get("/health", summary="Health Check")
async def health_check():
    """Basic health check for the agent."""
    return {
        "status": "ok" if models_loaded else "error",
        "agent_id": AGENT_ID,
        "models_loaded_count": len(pipelines),
        "model_names": [MODEL_1_NAME, MODEL_2_NAME]
    }

# --- Run (for development) ---
if __name__ == "__main__":
    port = int(os.environ.get("A2A_MEDICAL_PORT", 8003))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")