import os
import logging
import time
from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np # Add numpy import

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("A2AMedicalNER")

app = Flask(__name__)

# --- Agent Configuration ---
MODEL_1_NAME = "obi/deid_roberta_i2b2"
MODEL_2_NAME = "theekshana/deid-roberta-i2b2-NER-medical-reports"
AGENT_ID = "a2a_ner_medical"

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
        model_kwargs={"ignore_mismatched_sizes": True} # Keep this
    )
    logger.info(f"[{AGENT_ID}] Model 2 loaded successfully (with mismatched sizes ignored).")
except Exception as e:
    logger.error(f"[{AGENT_ID}] Failed to load model {MODEL_2_NAME}: {e}", exc_info=True)

models_loaded = len(pipelines) > 0

# --- API Endpoint (A2A Simulation) ---
@app.route("/detect", methods=["POST"])
def detect():
    """Receives text and returns detected entities from medical models."""
    if not models_loaded:
        logger.error(f"[{AGENT_ID}] No models loaded, cannot process request.")
        return jsonify({"error": f"No models available for agent {AGENT_ID}"}), 503

    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    if not text:
        return jsonify({"entities": []})

    logger.info(f"[{AGENT_ID}] Received detection request (text length: {len(text)}).")
    all_results = []
    total_duration = 0

    try:
        start_time = time.time()
        if 'model1' in pipelines:
            try:
                results1_raw = pipelines['model1'](text)
                # --- FIX: Process results1 ---
                results1_processed = []
                for item in results1_raw:
                    if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                        item['score'] = float(item['score'])
                    item['detector'] = f"{AGENT_ID}_model1"
                    results1_processed.append(item)
                # --- End FIX ---
                all_results.extend(results1_processed)
                logger.debug(f"[{AGENT_ID}] Model 1 found {len(results1_processed)} entities.")
            except Exception as e:
                logger.error(f"[{AGENT_ID}] Error running model 1: {e}", exc_info=True)

        if 'model2' in pipelines:
            try:
                results2_raw = pipelines['model2'](text)
                # --- FIX: Process results2 ---
                results2_processed = []
                for item in results2_raw:
                    if 'score' in item and isinstance(item['score'], (int, float, np.floating)):
                        item['score'] = float(item['score'])
                    item['detector'] = f"{AGENT_ID}_model2"
                    results2_processed.append(item)
                # --- End FIX ---
                all_results.extend(results2_processed)
                logger.debug(f"[{AGENT_ID}] Model 2 found {len(results2_processed)} entities.")
            except Exception as e:
                logger.error(f"[{AGENT_ID}] Error running model 2: {e}", exc_info=True)

        total_duration = time.time() - start_time
        logger.info(f"[{AGENT_ID}] Detection completed in {total_duration:.2f}s, found {len(all_results)} potential entities.")

        # Return the combined and processed results
        return jsonify({"entities": all_results})
    except Exception as e:
        logger.error(f"[{AGENT_ID}] Error during NER detection: {e}", exc_info=True)
        # Ensure the error message itself is serializable
        error_message = f"Detection failed in agent {AGENT_ID}: {str(e)}"
        return jsonify({"error": error_message}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check for the agent."""
    return jsonify({
        "status": "ok" if models_loaded else "error",
        "agent_id": AGENT_ID,
        "models_loaded_count": len(pipelines),
        "model_names": [MODEL_1_NAME, MODEL_2_NAME]
    })

# --- Run (for development) ---
if __name__ == "__main__":
    port = int(os.environ.get("A2A_MEDICAL_PORT", 8003))
    logger.info(f"Starting {AGENT_ID} on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)