import os
import logging
import time
from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np # Add numpy import

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("A2ASpecializedPII")

app = Flask(__name__)

# --- Agent Configuration ---
MODEL_NAME = "1-13-am/xlm-roberta-base-pii-finetuned"
AGENT_ID = "a2a_ner_pii_specialized"

# --- Load Model ---
ner_pipeline = None
try:
    logger.info(f"[{AGENT_ID}] Loading model: {MODEL_NAME}")
    ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    logger.info(f"[{AGENT_ID}] Model loaded successfully.")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_NAME}: {e}", exc_info=True)

# --- API Endpoint (A2A Simulation) ---
@app.route("/detect", methods=["POST"])
def detect():
    """Receives text and returns detected entities from this agent's model."""
    if ner_pipeline is None:
        logger.error(f"[{AGENT_ID}] Model not loaded, cannot process request.")
        return jsonify({"error": f"Model for agent {AGENT_ID} is not available"}), 503

    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    if not text:
        return jsonify({"entities": []})

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
                 item['score'] = float(item['score']) # Cast to standard float
            item['detector'] = AGENT_ID
            processed_results.append(item)
        # --- End FIX ---

        return jsonify({"entities": processed_results})
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER detection: {e}", exc_info=True)
        # Ensure the error message itself is serializable
        error_message = f"Detection failed in agent {AGENT_ID}: {str(e)}"
        return jsonify({"error": error_message}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check for the agent."""
    return jsonify({
        "status": "ok" if ner_pipeline else "error",
        "agent_id": AGENT_ID,
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME
    })

# --- Run (for development) ---
if __name__ == "__main__":
    port = int(os.environ.get("A2A_PII_SPECIALIZED_PORT", 8005))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)