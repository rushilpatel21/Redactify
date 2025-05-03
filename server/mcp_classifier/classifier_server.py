import os
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
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

# --- API Models ---
class TextInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    classifications: List[str]

# --- API Endpoint ---
@app.post("/classify",
          response_model=ClassificationOutput,
          summary="Classify input text using Zero-Shot Model",
          description=f"Returns a list of relevant classifications from {CANDIDATE_LABELS} based on model confidence (threshold: {CONFIDENCE_THRESHOLD}).")
async def classify_endpoint(data: TextInput):
    """Receives text and returns its classification(s) using the model."""
    start_time = time.time()
    if not data.text:
        return {"classifications": ["general"]}

    try:
        classification_list = model_classify(data.text)
        duration = time.time() - start_time
        text_snippet = data.text[:100].replace('\n', ' ') + ('...' if len(data.text) > 100 else '')
        logger.info(f"Classified snippet '{text_snippet}' as: {classification_list} in {duration:.4f}s")
        return {"classifications": classification_list}
    except Exception as e:
        logger.error(f"Unexpected error in /classify endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during classification: {e}")

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