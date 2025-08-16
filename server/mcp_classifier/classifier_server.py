import os
import logging
import time
import uuid
import json
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPLLMClassifier")

# --- LLM Configuration ---
GEMINI_API_KEY = "AIzaSyCptLRj8vNEYCt541zcSh3vUzQO1mNp6rU"
DOCUMENT_CATEGORIES = ["medical", "technical", "general"]

# Initialize Gemini client
gemini_model = None
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini client configured successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)

# --- MCP Server Setup ---
mcp = FastMCP(
    name="LLMClassifier",
    version="1.0.0",
    description="LLM-based text classifier for intelligent PII detection routing"
)

# --- Classification Logic ---
async def llm_classify(text: str) -> List[str]:
    """Classifies text using Gemini LLM reasoning, returning a list of relevant categories."""
    if not gemini_model:
        logger.error("Gemini model not initialized. Cannot perform LLM classification.")
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

        # Using Gemini to generate response
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()

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
            logger.warning(f"Failed to parse Gemini response as JSON: {content}")
            classifications = ["general"]
            
        return sorted(list(set(classifications)))

    except Exception as e:
        logger.error(f"Error during Gemini classification: {e}", exc_info=True)
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
        "model_used": "gemini-1.5-flash",
        "processing_time": duration
    }

@mcp.tool()
async def classify_document_type(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Classify document type and determine appropriate processing strategy.
    
    Analyzes text to determine:
    - Document type (medical, legal, financial, technical, general)
    - Confidence level
    - Recommended detection domains
    - Processing strategy
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] Classifying document type for text of length {len(inputs)}")
    
    text = inputs or ""
    if not text:
        return {
            "document_type": "unknown",
            "confidence": 0.0,
            "recommended_domains": ["general"],
            "processing_strategy": "conservative"
        }
    
    # Use LLM for detailed document classification
    try:
        prompt = f"""
Analyze this text and classify the document type. Consider the content, terminology, and context.

Text to analyze:
"{text[:1500]}..."

Classify into one of these categories:
- medical: Healthcare records, clinical notes, patient information
- legal: Legal documents, contracts, court records, case files
- financial: Banking, financial records, tax documents, payment info
- technical: Software documentation, API docs, technical specifications
- general: Business communications, news, general correspondence

Return JSON format:
{{
  "document_type": "category",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "recommended_domains": ["domain1", "domain2"],
  "processing_strategy": "conservative|balanced|aggressive"
}}
"""
        
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        
        # Parse JSON response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
        else:
            raise ValueError("No JSON found in response")
        
        result = json.loads(json_str)
        
        duration = time.time() - start_time
        logger.info(f"[{request_id}] Document classified as {result.get('document_type')} in {duration:.4f}s")
        
        return {
            **result,
            "processing_time": duration,
            "tool_used": "classify_document_type"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in document classification: {e}")
        # Fallback classification
        return {
            "document_type": "general",
            "confidence": 0.3,
            "reasoning": "Fallback due to classification error",
            "recommended_domains": ["general"],
            "processing_strategy": "balanced",
            "processing_time": time.time() - start_time,
            "tool_used": "classify_document_type"
        }

@mcp.tool()
async def assess_sensitivity_level(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Assess document sensitivity level and recommend anonymization approach.
    
    Analyzes text to determine:
    - Sensitivity level (low, medium, high, critical)
    - Risk factors present
    - Recommended anonymization strategy
    - Confidence thresholds
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] Assessing sensitivity level for text of length {len(inputs)}")
    
    text = inputs or ""
    if not text:
        return {
            "sensitivity_level": "unknown",
            "confidence": 0.0,
            "risk_factors": [],
            "recommendations": {}
        }
    
    try:
        prompt = f"""
Analyze this text for sensitivity level and privacy risks. Consider what types of personal or sensitive information might be present.

Text to analyze:
"{text[:1500]}..."

Assess the sensitivity and return JSON:
{{
  "sensitivity_level": "low|medium|high|critical",
  "confidence": 0.0-1.0,
  "risk_factors": ["list of specific risks found"],
  "recommendations": {{
    "anonymization_strategy": "conservative|balanced|aggressive",
    "confidence_threshold": 0.0-1.0,
    "special_handling": ["any special considerations"]
  }},
  "reasoning": "brief explanation of assessment"
}}

Sensitivity levels:
- low: General business content, public information
- medium: Internal communications, some personal references
- high: Personal data, financial info, health references
- critical: Highly sensitive PII, medical records, legal documents
"""
        
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        
        # Parse JSON response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
        else:
            raise ValueError("No JSON found in response")
        
        result = json.loads(json_str)
        
        duration = time.time() - start_time
        logger.info(f"[{request_id}] Sensitivity assessed as {result.get('sensitivity_level')} in {duration:.4f}s")
        
        return {
            **result,
            "processing_time": duration,
            "tool_used": "assess_sensitivity_level"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in sensitivity assessment: {e}")
        # Fallback assessment
        return {
            "sensitivity_level": "medium",
            "confidence": 0.3,
            "risk_factors": ["unknown"],
            "recommendations": {
                "anonymization_strategy": "balanced",
                "confidence_threshold": 0.5,
                "special_handling": []
            },
            "reasoning": "Fallback due to assessment error",
            "processing_time": time.time() - start_time,
            "tool_used": "assess_sensitivity_level"
        }

@mcp.tool()
async def health_check() -> Dict[str, str]:
    """Check the health of the LLM classifier service."""
    model_status = "initialized" if gemini_model else "not_initialized"
    return {
        "status": "ok" if model_status == "initialized" else "error", 
        "service": "LLMClassifier", 
        "model": "gemini-1.5-flash",
        "model_status": model_status
    }

# --- Run (for development) ---
if __name__ == "__main__":
    from fastapi import FastAPI, Request, Response
    import uvicorn
    
    port = int(os.environ.get("MCP_CLASSIFIER_PORT", 3007))
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