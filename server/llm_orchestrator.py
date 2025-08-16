"""
LLM Orchestrator for True MCP Implementation

This module implements an LLM-driven orchestration layer that makes intelligent
decisions about which MCP tools to use for PII detection and anonymization.
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from dataclasses import dataclass

logger = logging.getLogger("LLMOrchestrator")

@dataclass
class ToolCall:
    """Represents a tool call decision made by the LLM"""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str

@dataclass
class ExecutionPlan:
    """Represents the complete execution plan from the LLM"""
    tool_calls: List[ToolCall]
    strategy: str  # "conservative", "balanced", "aggressive"
    confidence: float
    reasoning: str

class LLMOrchestrator:
    """
    LLM-driven orchestration for MCP tool selection and execution.
    
    This class implements TRUE MCP behavior where an LLM makes intelligent
    decisions about which tools to use based on the user request and content.
    """
    
    def __init__(self, mcp_client_manager=None):
        self.mcp_client_manager = mcp_client_manager
        self.llm_model = None
        self._initialize_llm()
        
        # Available MCP tools registry
        self.available_tools = self._build_tool_registry()
        
        logger.info("LLMOrchestrator initialized with LLM-driven tool selection")
    
    def _initialize_llm(self):
        """Initialize the LLM for orchestration decisions"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found. LLM orchestration will be disabled.")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("LLM model initialized for orchestration")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
    
    def _build_tool_registry(self) -> Dict[str, Dict]:
        """Build registry of available MCP tools with descriptions"""
        return {
            # Analysis Tools
            "classify_document_type": {
                "description": "Classify document type and determine processing strategy",
                "server": "classifier",
                "parameters": ["text"],
                "returns": "document_type, confidence, recommended_domains"
            },
            "assess_sensitivity_level": {
                "description": "Assess document sensitivity and recommend anonymization level",
                "server": "classifier", 
                "parameters": ["text"],
                "returns": "sensitivity_level, risk_factors, recommendations"
            },
            
            # Detection Tools
            "general_ner_predict": {
                "description": "Detect standard entities (PERSON, ORG, LOCATION)",
                "server": "a2a_ner_general",
                "parameters": ["text", "confidence_threshold"],
                "returns": "entities with positions and confidence scores"
            },
            "medical_ner_predict": {
                "description": "Detect medical entities and PHI (patient info, conditions, treatments)",
                "server": "a2a_ner_medical",
                "parameters": ["text", "confidence_threshold"],
                "returns": "medical entities and PHI with high accuracy"
            },
            "financial_ner_predict": {
                "description": "Detect financial PII (SSN, account numbers, credit cards)",
                "server": "a2a_ner_financial",
                "parameters": ["text", "confidence_threshold"],
                "returns": "financial entities and sensitive numbers"
            },
            "legal_ner_predict": {
                "description": "Detect legal entities (case numbers, court names, legal references)",
                "server": "a2a_ner_legal",
                "parameters": ["text", "confidence_threshold"],
                "returns": "legal entities and case information"
            },
            "technical_ner_predict": {
                "description": "Detect technical secrets (API keys, IPs, URLs, passwords)",
                "server": "a2a_ner_technical",
                "parameters": ["text", "confidence_threshold"],
                "returns": "technical entities and security-sensitive data"
            },
            "pii_specialized_predict": {
                "description": "Specialized PII detection with high precision",
                "server": "a2a_ner_pii_specialized",
                "parameters": ["text", "confidence_threshold"],
                "returns": "high-confidence PII entities"
            },
            
            # Processing Tools
            "merge_overlapping_entities": {
                "description": "Merge overlapping entity detections from multiple models",
                "server": "a2a_ner_general",
                "parameters": ["text", "entities", "merge_strategy", "overlap_threshold"],
                "returns": "merged entities without overlaps"
            },
            
            # Anonymization Tools
            "pseudonymize_entities": {
                "description": "Replace entities with consistent hash-based pseudonyms",
                "server": "a2a_ner_general",
                "parameters": ["text", "entities", "strategy", "preserve_format"],
                "returns": "anonymized text with pseudonymized entities"
            },
            "mask_entities": {
                "description": "Partially mask entities while preserving format (e.g., J*** S***)",
                "server": "a2a_ner_general", 
                "parameters": ["text", "entities", "mask_char", "show_partial"],
                "returns": "text with entities partially masked"
            },
            "redact_entities": {
                "description": "Completely remove or redact entities from text",
                "server": "a2a_ner_general",
                "parameters": ["text", "entities", "redaction_style", "placeholder"],
                "returns": "text with entities completely redacted"
            }
        }
    
    async def process_anonymization_request(self, user_request: str, text: str, options: Dict = None) -> Dict[str, Any]:
        """
        Main entry point for LLM-driven anonymization.
        
        The LLM analyzes the request and text, then decides which tools to use.
        """
        start_time = time.time()
        
        logger.info(f"LLM Orchestrator processing request: {user_request[:100]}...")
        logger.info(f"Text length: {len(text)} characters")
        
        try:
            # Step 1: LLM analyzes the request and creates execution plan
            plan = await self.create_execution_plan(user_request, text, options)
            
            if not plan:
                logger.error("Failed to create execution plan")
                return self._fallback_processing(text, options)
            
            logger.info(f"LLM created plan with {len(plan.tool_calls)} tool calls")
            logger.info(f"Strategy: {plan.strategy}, Confidence: {plan.confidence}")
            
            # Step 2: Execute the plan using MCP tools
            results = await self.execute_plan(plan, text)
            
            processing_time = time.time() - start_time
            
            return {
                "anonymized_text": results.get("anonymized_text", text),
                "entities": results.get("entities", []),
                "execution_plan": {
                    "tool_calls": [{"tool": tc.tool_name, "reasoning": tc.reasoning} for tc in plan.tool_calls],
                    "strategy": plan.strategy,
                    "llm_confidence": plan.confidence,
                    "llm_reasoning": plan.reasoning
                },
                "processing_time": processing_time,
                "orchestration_method": "llm_driven"
            }
            
        except Exception as e:
            logger.error(f"Error in LLM orchestration: {e}", exc_info=True)
            return self._fallback_processing(text, options)
    
    async def create_execution_plan(self, user_request: str, text: str, options: Dict = None) -> Optional[ExecutionPlan]:
        """
        Use LLM to analyze the request and create an intelligent execution plan.
        """
        if not self.llm_model:
            logger.warning("LLM not available, cannot create execution plan")
            return None
        
        # Build the prompt for LLM decision making
        prompt = self._build_orchestration_prompt(user_request, text, options)
        
        try:
            logger.info("Asking LLM to create execution plan...")
            response = self.llm_model.generate_content(prompt)
            content = response.text.strip()
            
            logger.debug(f"LLM response: {content}")
            
            # Parse LLM response into execution plan
            plan = self._parse_llm_response(content)
            
            if plan:
                logger.info(f"LLM plan parsed successfully: {plan.strategy} strategy with {len(plan.tool_calls)} tools")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error getting LLM execution plan: {e}")
            return None
    
    def _build_orchestration_prompt(self, user_request: str, text: str, options: Dict = None) -> str:
        """Build the prompt for LLM orchestration decisions"""
        
        # Truncate text for prompt if too long
        text_sample = text[:2000] + "..." if len(text) > 2000 else text
        
        tools_description = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.available_tools.items()
        ])
        
        prompt = f"""
You are an AI orchestrator for PII anonymization. You must decide which tools to use based on the user request and text content.

AVAILABLE MCP TOOLS:
{tools_description}

USER REQUEST: "{user_request}"

TEXT TO ANALYZE (first 2000 chars):
"{text_sample}"

USER OPTIONS: {json.dumps(options or {}, indent=2)}

TASK: Create an execution plan by analyzing:
1. What type of document is this? (medical, legal, financial, technical, general)
2. What domains of PII are likely present?
3. What level of anonymization does the user want?
4. Which detection tools should be used and in what order?
5. What confidence thresholds are appropriate?

RESPONSE FORMAT (JSON):
{{
  "strategy": "conservative|balanced|aggressive",
  "confidence": 0.0-1.0,
  "reasoning": "Your analysis of the text and why you chose this approach",
  "tool_calls": [
    {{
      "tool_name": "tool_name_from_available_tools",
      "parameters": {{"param1": "value1"}},
      "reasoning": "Why this tool is needed"
    }}
  ]
}}

GUIDELINES:
- Conservative: High confidence thresholds, fewer tools, focus on precision
- Balanced: Moderate thresholds, comprehensive detection, good balance  
- Aggressive: Lower thresholds, all relevant tools, focus on recall

WORKFLOW STEPS:
1. ANALYZE: Start with classify_document_type and/or assess_sensitivity_level if uncertain
2. DETECT: Use appropriate *_ner_predict tools based on document type
3. PROCESS: Use merge_overlapping_entities if multiple detection tools were used
4. ANONYMIZE: Choose ONE anonymization method:
   - pseudonymize_entities: For complete anonymization with pseudonyms
   - mask_entities: For partial masking while preserving readability
   - redact_entities: For complete removal/redaction

IMPORTANT:
- Always include at least one detection tool AND one anonymization tool
- Use merge_overlapping_entities when using multiple detection tools
- Consider user preferences and document sensitivity
- Explain your reasoning clearly

Create your execution plan:
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[ExecutionPlan]:
        """Parse LLM response into structured execution plan"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                logger.error("No JSON found in LLM response")
                return None
            
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["strategy", "confidence", "reasoning", "tool_calls"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field in LLM response: {field}")
                    return None
            
            # Parse tool calls
            tool_calls = []
            for tc_data in data["tool_calls"]:
                if "tool_name" not in tc_data:
                    logger.warning("Tool call missing tool_name, skipping")
                    continue
                
                tool_call = ToolCall(
                    tool_name=tc_data["tool_name"],
                    parameters=tc_data.get("parameters", {}),
                    reasoning=tc_data.get("reasoning", "No reasoning provided")
                )
                tool_calls.append(tool_call)
            
            return ExecutionPlan(
                tool_calls=tool_calls,
                strategy=data["strategy"],
                confidence=float(data["confidence"]),
                reasoning=data["reasoning"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    async def execute_plan(self, plan: ExecutionPlan, text: str) -> Dict[str, Any]:
        """Execute the LLM-generated plan using MCP tools"""
        logger.info(f"Executing plan with {len(plan.tool_calls)} tool calls")
        
        all_entities = []
        execution_results = []
        current_text = text
        context = {"original_text": text}
        
        for i, tool_call in enumerate(plan.tool_calls):
            logger.info(f"Executing tool {i+1}/{len(plan.tool_calls)}: {tool_call.tool_name}")
            logger.debug(f"Tool reasoning: {tool_call.reasoning}")
            
            try:
                # Pass context to tools that might need it
                # For anonymization tools, use the current_text (which may have been updated by previous tools)
                text_to_use = current_text if tool_call.tool_name in ["pseudonymize_entities", "mask_entities", "redact_entities"] else text
                result = await self._execute_single_tool(tool_call, text_to_use, context)
                
                success = result is not None
                execution_results.append({
                    "tool": tool_call.tool_name,
                    "success": success,
                    "reasoning": tool_call.reasoning
                })
                
                if result:
                    # Handle different types of results
                    if "entities" in result:
                        # Detection tool result
                        new_entities = result["entities"]
                        all_entities.extend(new_entities)
                        execution_results[-1]["entities_found"] = len(new_entities)
                        
                        # Update context with detected entities
                        context["detected_entities"] = all_entities
                        
                    elif "anonymized_text" in result:
                        # Anonymization tool result
                        current_text = result["anonymized_text"]
                        execution_results[-1]["entities_processed"] = result.get("entities_processed", 0)
                        
                        # Update context with anonymized text
                        context["current_text"] = current_text
                        
                    elif "merged_count" in result:
                        # Entity merging result
                        all_entities = result["entities"]
                        execution_results[-1]["entities_merged"] = result.get("merges_performed", 0)
                        execution_results[-1]["final_count"] = result.get("merged_count", 0)
                        
                        # Update context with merged entities
                        context["detected_entities"] = all_entities
                    
                    # Store additional result info
                    for key in ["document_type", "sensitivity_level", "classifications"]:
                        if key in result:
                            context[key] = result[key]
                            execution_results[-1][key] = result[key]
                    
                else:
                    execution_results[-1]["error"] = "Tool returned no result"
                    
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.tool_name}: {e}")
                execution_results.append({
                    "tool": tool_call.tool_name,
                    "success": False,
                    "error": str(e),
                    "reasoning": tool_call.reasoning
                })
        
        # Determine final result
        final_result = {
            "execution_results": execution_results,
            "tools_executed": len(execution_results),
            "context": context
        }
        
        # If we have anonymized text, use it; otherwise return entities for manual processing
        if current_text != text:
            final_result["anonymized_text"] = current_text
            final_result["entities"] = all_entities
            logger.info(f"Anonymization completed: '{text}' -> '{current_text}'")
        else:
            final_result["entities"] = all_entities
            # If no anonymization was performed but we have entities, try a fallback
            if all_entities:
                logger.warning("No anonymization tool was executed, entities detected but text unchanged")
                logger.info(f"Detected {len(all_entities)} entities but no anonymization applied")
                
                # Fallback: Apply basic pseudonymization if we have entities
                try:
                    from anonymization_engine import get_anonymization_engine
                    anonymization_engine = get_anonymization_engine()
                    
                    fallback_result = anonymization_engine.anonymize_text(
                        text=text,
                        entities=all_entities,
                        strategy="pseudonymize",
                        preserve_format=False
                    )
                    
                    if fallback_result and fallback_result.get("anonymized_text") != text:
                        final_result["anonymized_text"] = fallback_result["anonymized_text"]
                        logger.info(f"Applied fallback anonymization: '{text}' -> '{fallback_result['anonymized_text']}'")
                    else:
                        final_result["anonymized_text"] = text
                        logger.warning("Fallback anonymization also failed")
                        
                except Exception as e:
                    logger.error(f"Fallback anonymization failed: {e}")
                    final_result["anonymized_text"] = text
            else:
                final_result["anonymized_text"] = text
        
        return final_result
    
    async def _execute_single_tool(self, tool_call: ToolCall, text: str, context: Dict = None) -> Optional[Dict]:
        """Execute a single MCP tool call"""
        tool_info = self.available_tools.get(tool_call.tool_name)
        if not tool_info:
            logger.error(f"Unknown tool: {tool_call.tool_name}")
            return None
        
        server_name = tool_info["server"]
        
        # Handle different tool types
        if tool_call.tool_name.endswith("_predict"):
            # This is a prediction tool
            return await self._call_prediction_tool(server_name, text, tool_call.parameters)
        elif tool_call.tool_name.startswith("classify_") or tool_call.tool_name.startswith("assess_"):
            # This is a classification/analysis tool
            return await self._call_analysis_tool(server_name, tool_call.tool_name, text, tool_call.parameters)
        elif tool_call.tool_name in ["pseudonymize_entities", "mask_entities", "redact_entities", "merge_overlapping_entities"]:
            # This is an anonymization/processing tool
            return await self._call_anonymization_tool(server_name, tool_call.tool_name, text, tool_call.parameters, context)
        else:
            logger.warning(f"Unknown tool type: {tool_call.tool_name}")
            return None
    
    async def _call_prediction_tool(self, server_name: str, text: str, parameters: Dict) -> Optional[Dict]:
        """Call a prediction tool on an MCP server"""
        if not self.mcp_client_manager:
            logger.error("MCP client manager not available")
            return None
        
        try:
            client = self.mcp_client_manager.get_client(server_name)
            result = await client.predict(text, parameters)
            return result
        except Exception as e:
            logger.error(f"Error calling prediction tool on {server_name}: {e}")
            return None
    
    async def _call_analysis_tool(self, server_name: str, tool_name: str, text: str, parameters: Dict) -> Optional[Dict]:
        """Call an analysis tool (classification, assessment, etc.)"""
        # For now, route to classifier server or implement basic logic
        if server_name == "classifier" and self.mcp_client_manager:
            try:
                # Try to call the classifier server if it exists
                client = self.mcp_client_manager.get_client("mcp_classifier")
                # Call the classify method with the tool name
                result = await client.predict(text, {"tool_type": tool_name, **parameters})
                return result
            except Exception as e:
                logger.debug(f"Classifier server not available, using fallback: {e}")
        
        # Fallback: basic analysis
        return self._basic_analysis(tool_name, text, parameters)
    
    def _basic_analysis(self, tool_name: str, text: str, parameters: Dict) -> Dict:
        """Basic analysis fallback when specialized tools aren't available"""
        if tool_name == "classify_document_type":
            # Simple keyword-based classification
            text_lower = text.lower()
            if any(word in text_lower for word in ["patient", "medical", "diagnosis", "treatment"]):
                return {"document_type": "medical", "confidence": 0.7, "recommended_domains": ["medical", "general"]}
            elif any(word in text_lower for word in ["account", "ssn", "credit", "financial"]):
                return {"document_type": "financial", "confidence": 0.7, "recommended_domains": ["financial", "general"]}
            elif any(word in text_lower for word in ["case", "court", "legal", "attorney"]):
                return {"document_type": "legal", "confidence": 0.7, "recommended_domains": ["legal", "general"]}
            else:
                return {"document_type": "general", "confidence": 0.5, "recommended_domains": ["general"]}
        
        elif tool_name == "assess_sensitivity_level":
            # Simple sensitivity assessment
            sensitive_patterns = ["ssn", "social security", "credit card", "password", "api key"]
            sensitivity = "high" if any(pattern in text.lower() for pattern in sensitive_patterns) else "medium"
            return {"sensitivity_level": sensitivity, "confidence": 0.6}
        
        return {"result": "basic_analysis", "confidence": 0.3}
    
    async def _call_anonymization_tool(self, server_name: str, tool_name: str, text: str, parameters: Dict, context: Dict = None) -> Optional[Dict]:
        """Call an anonymization tool on an MCP server"""
        if not self.mcp_client_manager:
            logger.error("MCP client manager not available")
            return None
        
        try:
            client = self.mcp_client_manager.get_client(server_name)
            
            # For anonymization tools, we need to pass the detected entities
            tool_parameters = parameters.copy()
            
            # If no entities provided in parameters, get them from context
            if "entities" not in tool_parameters and context and "detected_entities" in context:
                tool_parameters["entities"] = context["detected_entities"]
                logger.info(f"Using {len(context['detected_entities'])} entities from context for {tool_name}")
            
            # If still no entities, we can't anonymize
            if "entities" not in tool_parameters or not tool_parameters["entities"]:
                logger.warning(f"No entities available for anonymization tool {tool_name}")
                return {"anonymized_text": text, "entities_processed": 0, "error": "No entities to anonymize"}
            
            logger.info(f"Calling {tool_name} with {len(tool_parameters['entities'])} entities")
            
            # Create the request payload for the anonymization tool
            request_payload = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": {
                    "inputs": text,
                    "parameters": tool_parameters
                },
                "id": str(uuid.uuid4())
            }
            
            # Make direct JSON-RPC call
            async with client.session.post(
                f"{client.config.url}/mcp",
                json=request_payload,
                timeout=client.config.timeout
            ) as response:
                data = await response.json()
                
                if "result" in data:
                    result = data["result"]
                    logger.info(f"{tool_name} processed {result.get('entities_processed', 0)} entities")
                    return result
                elif "error" in data:
                    logger.error(f"MCP tool error: {data['error']}")
                    return None
                else:
                    logger.error(f"Invalid MCP response: {data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling anonymization tool {tool_name} on {server_name}: {e}")
            return None
    
    def _fallback_processing(self, text: str, options: Dict = None) -> Dict[str, Any]:
        """Fallback to original processing when LLM orchestration fails"""
        logger.info("Using fallback processing (non-LLM)")
        
        # This would call the original detection_engine logic
        return {
            "anonymized_text": text,  # Placeholder
            "entities": [],
            "execution_plan": {"fallback": True},
            "processing_time": 0.0,
            "orchestration_method": "fallback"
        }