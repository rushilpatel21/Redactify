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
logger = logging.getLogger("A2AGeneralNER")

app = FastAPI()
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
AGENT_ID    = "a2a_ner_general"

class ModelContextRequest(BaseModel):
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    inputs: str
    parameters: Optional[Dict[str, Any]] = None

class ModelContextResponse(BaseModel):
    model_id: str
    model_version: str
    outputs: Dict[str, Any]

ner_pipeline = None
try:
    logger.info(f"[{AGENT_ID}] Loading model: {MODEL_NAME}")
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    logger.info(f"[{AGENT_ID}] Model loaded successfully.")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_NAME}: {e}", exc_info=True)

@app.post("/predict", response_model=ModelContextResponse)
async def predict(request: ModelContextRequest):
    text = request.inputs or ""
    if ner_pipeline is None:
        raise HTTPException(503, "Model not available")
    if not text:
        return ModelContextResponse(
            model_id=request.model_id or AGENT_ID,
            model_version=request.model_version or MODEL_NAME,
            outputs={"entities": []}
        )
    logger.info(f"[{AGENT_ID}] Detecting entities in text length {len(text)}")
    try:
        start = time.time()
        raw = ner_pipeline(text)
        logger.info(f"[{AGENT_ID}] Found {len(raw)} spans in {time.time()-start:.2f}s")
        results = []
        for item in raw:
            if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                item['score'] = float(item['score'])
            if 'start' in item: item['start'] = int(item['start'])
            if 'end' in item:   item['end']   = int(item['end'])
            item['detector'] = AGENT_ID
            results.append(item)
        return ModelContextResponse(
            model_id=request.model_id or AGENT_ID,
            model_version=request.model_version or MODEL_NAME,
            outputs={"entities": results}
        )
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {e}")

@app.get("/health", summary="Health Check")
async def health():
    return {
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("A2A_GENERAL_PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)