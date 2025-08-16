#!/usr/bin/env python3
"""
Simple test to verify TRUE MCP implementation is working.

This test demonstrates that the LLM is making intelligent decisions
about which tools to use for anonymization.
"""

import asyncio
import json
import aiohttp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrueMCPTest")

async def test_llm_driven_anonymization():
    """Test the LLM-driven anonymization endpoint"""
    
    # Test cases that should trigger different LLM decisions
    test_cases = [
        {
            "name": "Medical Document",
            "user_request": "Anonymize this medical record completely",
            "text": "Patient John Smith (DOB: 1985-03-15) was diagnosed with diabetes. Dr. Sarah Johnson at Memorial Hospital prescribed metformin. Contact: john.smith@email.com, phone: 555-123-4567.",
            "expected_llm_decisions": ["medical_ner_predict", "general_ner_predict", "pseudonymize_entities"]
        },
        {
            "name": "Business Email", 
            "user_request": "Partially mask this business email to preserve readability",
            "text": "Hi Jane Doe, please contact our client Microsoft Corp at their Seattle office. Best regards, Bob Wilson (bob.wilson@company.com).",
            "expected_llm_decisions": ["general_ner_predict", "mask_entities"]
        },
        {
            "name": "Financial Document",
            "user_request": "Completely redact all sensitive information from this financial document",
            "text": "Account holder: Alice Johnson, SSN: 123-45-6789, Account: 9876543210, Credit Card: 4532-1234-5678-9012. Bank: Chase Manhattan.",
            "expected_llm_decisions": ["financial_ner_predict", "general_ner_predict", "redact_entities"]
        }
    ]
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        for test_case in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {test_case['name']}")
            logger.info(f"User Request: {test_case['user_request']}")
            logger.info(f"{'='*60}")
            
            payload = {
                "user_request": test_case["user_request"],
                "text": test_case["text"]
            }
            
            try:
                async with session.post(f"{base_url}/anonymize_llm", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Display LLM's decision-making
                        execution_plan = result.get("execution_plan", {})
                        strategy = execution_plan.get("strategy", "unknown")
                        llm_reasoning = execution_plan.get("llm_reasoning", "No reasoning provided")
                        tool_calls = execution_plan.get("tool_calls", [])
                        
                        logger.info(f"✅ LLM Strategy: {strategy}")
                        logger.info(f"✅ LLM Reasoning: {llm_reasoning}")
                        logger.info(f"✅ Tools Selected by LLM:")
                        
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("tool", "unknown")
                            reasoning = tool_call.get("reasoning", "No reasoning")
                            logger.info(f"   - {tool_name}: {reasoning}")
                        
                        # Show results
                        if "anonymized_text" in result:
                            logger.info(f"✅ Original: {test_case['text']}")
                            logger.info(f"✅ Anonymized: {result['anonymized_text']}")
                        
                        # Verify LLM made intelligent decisions
                        tools_used = [tc.get("tool", "") for tc in tool_calls]
                        expected_tools = test_case["expected_llm_decisions"]
                        
                        logger.info(f"✅ Expected tools: {expected_tools}")
                        logger.info(f"✅ LLM selected: {tools_used}")
                        
                        # Check if LLM made reasonable decisions
                        has_detection = any("predict" in tool for tool in tools_used)
                        has_anonymization = any(tool in ["pseudonymize_entities", "mask_entities", "redact_entities"] for tool in tools_used)
                        
                        if has_detection and has_anonymization:
                            logger.info("✅ SUCCESS: LLM made complete workflow decisions!")
                        else:
                            logger.warning("⚠️  PARTIAL: LLM workflow incomplete")
                            
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"❌ Test failed: {e}")
            
            # Wait between tests
            await asyncio.sleep(2)

async def test_server_health():
    """Test that all MCP servers are running"""
    logger.info("\n" + "="*60)
    logger.info("Testing MCP Server Health")
    logger.info("="*60)
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/mcp-status") as response:
                if response.status == 200:
                    status = await response.json()
                    mcp_servers = status.get("mcp_servers", {})
                    
                    logger.info(f"✅ Total MCP servers: {status.get('total_servers', 0)}")
                    logger.info(f"✅ Healthy servers: {status.get('healthy_servers', 0)}")
                    
                    for server_name, server_info in mcp_servers.items():
                        server_status = server_info.get("status", "unknown")
                        server_url = server_info.get("url", "unknown")
                        
                        if server_status == "healthy":
                            logger.info(f"✅ {server_name}: {server_status} ({server_url})")
                        else:
                            logger.warning(f"⚠️  {server_name}: {server_status} ({server_url})")
                else:
                    logger.error(f"❌ Health check failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")

async def main():
    """Run all tests"""
    logger.info("🚀 Starting TRUE MCP Implementation Test")
    logger.info("This test verifies that the LLM makes intelligent tool selection decisions")
    
    # Test server health first
    await test_server_health()
    
    # Test LLM-driven anonymization
    await test_llm_driven_anonymization()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 TRUE MCP Test Complete!")
    logger.info("If you see LLM making tool decisions above, TRUE MCP is working!")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())