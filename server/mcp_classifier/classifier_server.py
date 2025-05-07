import os
import logging
import time
import uuid
import json
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPLLMClassifier")

# --- LLM Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("LLM_MODEL_NAME", "gpt-4-turbo")
DOCUMENT_CATEGORIES = ["medical", "technical", "general"]

# Initialize OpenAI client
client = None
try:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info(f"OpenAI client configured with model {OPENAI_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)

# --- MCP Server Setup ---
mcp = FastMCP(
    name="LLMClassifier",
    version="1.0.0",
    description="LLM-based text classifier for intelligent PII detection routing"
)

# --- Classification Logic ---
async def llm_classify(text: str) -> List[str]:
    """Classifies text using LLM reasoning, returning a list of relevant categories."""
    if not OPENAI_API_KEY or not client:
        logger.error("OpenAI API key not set or client initialization failed. Cannot perform LLM classification.")
        return ["general"]

    if not text:
        return ["general"]

    try:
        max_length = 4000
        truncated_text = text[:max_length] if len(text) > max_length else text

        prompt = f"""
        You are a document classifier that helps route text to specialized PII detection models.
        
        Analyze the following text and determine which categories it belongs to from: {', '.join(DOCUMENT_CATEGORIES)}.
        Multiple categories can apply if the text contains mixed content.
        Always include "general" as a fallback category if no specific category applies.
        
        Return ONLY a JSON array of category names, nothing else.
        
        Text to classify:
        {truncated_text}
        """

        # Using the new OpenAI client API (v1.0+)
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )

        # Try to parse the response as JSON array
        content = response.choices[0].message.content.strip()
        try:
            # Remove any markdown code formatting if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            classifications = json.loads(content)
            if not isinstance(classifications, list):
                classifications = ["general"]
                
            # Ensure all classifications are valid
            classifications = [c.lower() for c in classifications if c.lower() in DOCUMENT_CATEGORIES]
            
            # Always include "general" if empty
            if not classifications:
                classifications.append("general")
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {content}")
            classifications = ["general"]
            
        return sorted(list(set(classifications)))

    except Exception as e:
        logger.error(f"Error during LLM classification: {e}", exc_info=True)
        return ["general"]

# --- MCP Tools ---
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify text using LLM-based analysis to determine which PII detection models to use.
    
    Args:
        inputs: The text to classify
        parameters: Optional parameters (not used currently)
        
    Returns:
        Dictionary containing classifications
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] Classifying document of length {len(inputs)}")
    
    text = inputs or ""
    classifications = await llm_classify(text)
    
    duration = time.time() - start_time
    logger.info(f"[{request_id}] MCP predict tool: classified as {classifications} in {duration:.4f}s")
    
    return {
        "classifications": classifications,
        "model_used": OPENAI_MODEL,
        "processing_time": duration
    }

@mcp.tool()
async def health_check() -> Dict[str, str]:
    """Check the health of the LLM classifier service."""
    model_status = "initialized" if OPENAI_API_KEY else "missing_api_key"
    return {
        "status": "ok" if model_status == "initialized" else "error", 
        "service": "LLMClassifier", 
        "model": OPENAI_MODEL,
        "model_status": model_status
    }

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    
    port = int(os.environ.get("MCP_CLASSIFIER_PORT", 8001))
    logger.info(f"Starting LLM Text Classifier MCP Server on port {port}")
    
    # Create a regular FastAPI app
    app = FastAPI(title="LLM Classifier MCP Server")
    
    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        """MCP JSON-RPC endpoint that processes requests and forwards to appropriate tools"""
        request_id = str(uuid.uuid4())[:8]
        try:
            data = await request.json()
            logger.info(f"[{request_id}] Received MCP request")
            
            # Check if this is a JSON-RPC request
            if "jsonrpc" in data and "method" in data:
                method = data["method"]
                params = data.get("params", {})
                request_id = data.get("id")
                
                # Call the appropriate tool
                if method == "predict" and "inputs" in params:
                    logger.info(f"[{request_id}] Processing predict request for text of length {len(params['inputs'])}")
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
            logger.error(f"[{request_id}] Error processing MCP request: {e}", exc_info=True)
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