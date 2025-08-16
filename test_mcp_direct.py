#!/usr/bin/env python3
"""
Direct MCP Test - Bypasses LLM to test MCP servers directly
This proves the MCP infrastructure is working correctly.
"""

import asyncio
import json
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DirectMCPTest")

async def test_mcp_servers_directly():
    """Test MCP servers directly without LLM orchestration"""
    
    # Test data
    test_text = "Patient John Smith was treated by Dr. Sarah Johnson at Memorial Hospital"
    
    # MCP servers to test
    mcp_servers = [
        {"name": "General NER", "port": 3001},
        {"name": "Medical NER", "port": 3002},
        {"name": "Technical NER", "port": 3003},
        {"name": "Legal NER", "port": 3004},
        {"name": "Financial NER", "port": 3005},
        {"name": "PII Specialized", "port": 3006},
        {"name": "Classifier", "port": 3007}
    ]
    
    async with aiohttp.ClientSession() as session:
        logger.info("🚀 Testing MCP Servers Directly")
        logger.info("="*60)
        
        for server in mcp_servers:
            name = server["name"]
            port = server["port"]
            url = f"http://localhost:{port}/mcp"
            
            logger.info(f"\n📡 Testing {name} (Port {port})")
            
            # Test 1: Health Check
            health_payload = {
                "jsonrpc": "2.0",
                "method": "health_check",
                "params": {},
                "id": "health-test"
            }
            
            try:
                async with session.post(url, json=health_payload, timeout=5) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "result" in result:
                            status = result["result"].get("status", "unknown")
                            logger.info(f"  ✅ Health: {status}")
                        else:
                            logger.warning(f"  ⚠️  Health: Invalid response")
                    else:
                        logger.error(f"  ❌ Health: HTTP {response.status}")
                        continue
            except Exception as e:
                logger.error(f"  ❌ Health: {e}")
                continue
            
            # Test 2: Predict (for NER servers)
            if "NER" in name:
                predict_payload = {
                    "jsonrpc": "2.0",
                    "method": "predict",
                    "params": {
                        "inputs": test_text,
                        "parameters": {"confidence_threshold": 0.5}
                    },
                    "id": "predict-test"
                }
                
                try:
                    start_time = asyncio.get_event_loop().time()
                    async with session.post(url, json=predict_payload, timeout=30) as response:
                        end_time = asyncio.get_event_loop().time()
                        duration = end_time - start_time
                        
                        if response.status == 200:
                            result = await response.json()
                            if "result" in result and "entities" in result["result"]:
                                entities = result["result"]["entities"]
                                logger.info(f"  ✅ Predict: {len(entities)} entities in {duration:.2f}s")
                                
                                # Show first few entities
                                for i, entity in enumerate(entities[:3]):
                                    entity_type = entity.get("entity_group", "UNKNOWN")
                                    word = entity.get("word", "")
                                    score = entity.get("score", 0)
                                    logger.info(f"    - {entity_type}: '{word}' (confidence: {score:.3f})")
                                
                                if len(entities) > 3:
                                    logger.info(f"    ... and {len(entities) - 3} more entities")
                            else:
                                logger.warning(f"  ⚠️  Predict: Invalid response format")
                        else:
                            logger.error(f"  ❌ Predict: HTTP {response.status}")
                except Exception as e:
                    logger.error(f"  ❌ Predict: {e}")
            
            # Test 3: Anonymization Tools (for General NER)
            if name == "General NER":
                logger.info(f"  🔧 Testing anonymization tools...")
                
                # First get some entities
                predict_payload = {
                    "jsonrpc": "2.0",
                    "method": "predict",
                    "params": {"inputs": test_text},
                    "id": "get-entities"
                }
                
                try:
                    async with session.post(url, json=predict_payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            entities = result.get("result", {}).get("entities", [])
                            
                            if entities:
                                # Test pseudonymization
                                pseudo_payload = {
                                    "jsonrpc": "2.0",
                                    "method": "pseudonymize_entities",
                                    "params": {
                                        "inputs": test_text,
                                        "parameters": {
                                            "entities": entities,
                                            "strategy": "hash"
                                        }
                                    },
                                    "id": "pseudo-test"
                                }
                                
                                async with session.post(url, json=pseudo_payload, timeout=10) as pseudo_response:
                                    if pseudo_response.status == 200:
                                        pseudo_result = await pseudo_response.json()
                                        if "result" in pseudo_result:
                                            anonymized = pseudo_result["result"].get("anonymized_text", "")
                                            processed = pseudo_result["result"].get("entities_processed", 0)
                                            logger.info(f"  ✅ Pseudonymize: {processed} entities processed")
                                            logger.info(f"    Original: {test_text}")
                                            logger.info(f"    Anonymized: {anonymized}")
                                        else:
                                            logger.warning(f"  ⚠️  Pseudonymize: Invalid response")
                                    else:
                                        logger.error(f"  ❌ Pseudonymize: HTTP {pseudo_response.status}")
                            else:
                                logger.info(f"  ℹ️  No entities found for anonymization test")
                except Exception as e:
                    logger.error(f"  ❌ Anonymization test: {e}")

async def main():
    logger.info("🧪 Direct MCP Server Testing")
    logger.info("This bypasses LLM quota limits and tests MCP infrastructure directly")
    logger.info("")
    
    await test_mcp_servers_directly()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 Direct MCP Test Complete!")
    logger.info("If you see entities and anonymization above, TRUE MCP is working!")

if __name__ == "__main__":
    asyncio.run(main())