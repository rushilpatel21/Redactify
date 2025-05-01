import os
import logging
import time # Added for logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPClassifierServer")

app = FastAPI(title="Text Classifier MCP Server")

# --- Simple Keyword-Based Classification ---
# In a real scenario, load a pre-trained fast classification model here
MEDICAL_KEYWORDS = {"patient", "doctor", "hospital", "diagnosis", "medical", "clinic", "treatment", "prescription", "mrn", "hipaa", "phi"}
TECHNICAL_KEYWORDS = {"code", "software", "api", "server", "debug", "technical", "error", "python", "java", "javascript", "bug", "feature", "github", "gitlab", "token", "key", "secret", "password", "credential"}

def simple_classify(text: str) -> str:
    """Classifies text based on keywords."""
    if not text:
        return "general"
    # Limit text length for performance if needed
    text_to_scan = text[:5000].lower() # Scan first 5000 chars
    text_lower_words = set(text_to_scan.split())

    # Prioritize technical if sensitive keywords are present
    if text_lower_words.intersection(TECHNICAL_KEYWORDS):
        return "technical"
    elif text_lower_words.intersection(MEDICAL_KEYWORDS):
        return "medical"
    else:
        return "general"

# --- API Models ---
class TextInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    classification: str

# --- API Endpoint (MCP Simulation) ---
@app.post("/classify",
          response_model=ClassificationOutput,
          summary="Classify input text",
          description="Simulates an MCP endpoint to get text classification (e.g., general, medical, technical).")
async def classify_endpoint(data: TextInput):
    """Receives text and returns its classification."""
    start_time = time.time()
    if not data.text:
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        classification = simple_classify(data.text)
        duration = time.time() - start_time
        logger.info(f"Classified text snippet as: {classification} in {duration:.4f}s")
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during classification: {e}")

@app.get("/health", summary="Health Check")
async def health():
    """Basic health check."""
    return {"status": "ok", "service": "Text Classifier MCP Server"}

# --- Run (for development) ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_CLASSIFIER_PORT", 8001))
    logger.info(f"Starting Text Classifier MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)