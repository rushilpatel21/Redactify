import os
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Set

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPClassifierServer")

app = FastAPI(title="Text Classifier MCP Server")

# --- Simple Keyword-Based Classification ---
MEDICAL_KEYWORDS = {
    "patient", "doctor", "hospital", "diagnosis", "medical", "clinic", "treatment",
    "prescription", "mrn", "hipaa", "phi", "physician", "nurse", "ward", "ecg",
    "medication", "symptoms", "admission", "discharge", "radiology", "pathology"
}
TECHNICAL_KEYWORDS = {
    "code", "software", "api", "server", "debug", "technical", "error", "python",
    "java", "javascript", "bug", "feature", "github", "gitlab", "token", "key",
    "secret", "password", "credential", "ip address", "mac address", "network",
    "database", "query", "stacktrace", "deployment", "ssh", "url", "http"
}

def simple_classify(text: str) -> List[str]:
    """
    Classifies text based on keywords, returning a list of relevant categories.
    """
    if not text:
        return ["general"]

    # Limit text length for performance if needed
    text_to_scan = text.lower() # Scan first 10000 chars
    text_lower_words = set(text_to_scan.split())

    classifications: Set[str] = set() # Use a set to avoid duplicates

    # Check for medical keywords
    if text_lower_words.intersection(MEDICAL_KEYWORDS):
        classifications.add("medical")

    # Check for technical keywords
    if text_lower_words.intersection(TECHNICAL_KEYWORDS):
        classifications.add("technical")

    # If no specific category is found, default to general
    if not classifications:
        classifications.add("general")

    return sorted(list(classifications)) # Return sorted list

# --- API Models ---
class TextInput(BaseModel):
    text: str

class ClassificationOutput(BaseModel):
    classifications: List[str]

# --- API Endpoint (MCP Simulation) ---
@app.post("/classify",
          response_model=ClassificationOutput,
          summary="Classify input text",
          description="Returns a list of relevant classifications (e.g., ['general', 'medical']).")
async def classify_endpoint(data: TextInput):
    """Receives text and returns its classification(s)."""
    start_time = time.time()
    if not data.text:
        return {"classifications": ["general"]}
    try:
        classification_list = simple_classify(data.text)
        duration = time.time() - start_time
        logger.info(f"Classified text snippet as: {classification_list} in {duration:.4f}s")
        return {"classifications": classification_list}
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