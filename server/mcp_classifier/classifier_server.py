import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Any, Dict, Optional
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPClassifierServer")

# --- Model Configuration ---
MODEL_NAME = os.environ.get("MCP_MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
CANDIDATE_LABELS = ["medical", "technical", "general"]
CONFIDENCE_THRESHOLD = float(os.environ.get("MCP_THRESHOLD", 0.50))

# --- Global Variables ---
classifier_pipeline = None
app = FastAPI(title="Zero-Shot Text Classifier MCP Server")

# --- Model Loading ---
@app.on_event("startup")
async def load_model():
    """Load the zero-shot classification pipeline at startup."""
    global classifier_pipeline
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

# --- Classification Logic ---
def model_classify(text: str) -> List[str]:
    """
    Classifies text using the zero-shot model, returning a list of relevant categories.
    """
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

# --- MCP Models ---
class ModelContextRequest(BaseModel):
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    inputs: str
    parameters: Optional[Dict[str, Any]] = None

class ModelContextResponse(BaseModel):
    model_id: str
    model_version: str
    outputs: Dict[str, Any]

# --- MCP Predict Endpoint ---
@app.post(
    "/predict",
    response_model=ModelContextResponse,
    summary="MCP Predict: Zero-Shot Classification",
    description="MCP protocol endpoint wrapping zero-shot classify"
)
async def predict(request: ModelContextRequest):
    start_time = time.time()
    text = request.inputs or ""
    if not text:
        classifications = ["general"]
    else:
        classifications = model_classify(text)
    duration = time.time() - start_time
    logger.info(f"MCP Predict: model={request.model_id or MODEL_NAME} classified in {duration:.4f}s")
    return ModelContextResponse(
        model_id=request.model_id or MODEL_NAME,
        model_version=request.model_version or MODEL_NAME,
        outputs={"classifications": classifications}
    )

# --- Health Check ---
@app.get("/health", summary="Health Check")
async def health():
    """Basic health check, indicates if model is loaded."""
    model_status = "loaded" if classifier_pipeline else "failed_to_load"
    return {"status": "ok", "service": "Zero-Shot Text Classifier MCP Server", "model_status": model_status}

# --- Run (for development) ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_CLASSIFIER_PORT", 8001))
    logger.info(f"Starting Zero-Shot Text Classifier MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)