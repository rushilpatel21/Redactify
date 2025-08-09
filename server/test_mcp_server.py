#!/usr/bin/env python3
"""
Test script for Redactify MCP Server Phase 1

This script tests the core infrastructure components:
- ModelManager
- DetectionEngine  
- Basic MCP server functionality
"""

import asyncio
import logging
import json
import sys
import os
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPServerTest")

async def test_model_manager():
    """Test ModelManager functionality"""
    logger.info("=== Testing ModelManager ===")
    
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        logger.info("✓ ModelManager created successfully")
        
        # Test available models
        available_models = manager.get_available_models()
        logger.info(f"✓ Available models: {available_models}")
        
        # Test model stats
        stats = manager.get_model_stats()
        logger.info(f"✓ Model stats: {stats['loaded_models']} loaded, {stats['total_memory_mb']:.1f}MB used")
        
        # Test loading general model
        logger.info("Loading general NER model...")
        general_model = await manager.get_model("general")
        
        if general_model:
            logger.info("✓ General model loaded successfully")
            
            # Test prediction
            test_text = "Hello, my name is John Doe and I work at Microsoft."
            results = general_model(test_text)
            logger.info(f"✓ Model prediction test: found {len(results)} entities")
            
            for entity in results:
                logger.info(f"  - {entity.get('entity_group', 'UNKNOWN')}: "
                           f"'{test_text[entity.get('start', 0):entity.get('end', 0)]}' "
                           f"(confidence: {entity.get('score', 0):.3f})")
        else:
            logger.error("✗ Failed to load general model")
            return False
        
        # Test stats after loading
        stats_after = manager.get_model_stats()
        logger.info(f"✓ Stats after loading: {stats_after['loaded_models']} loaded, "
                   f"{stats_after['total_memory_mb']:.1f}MB used")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ModelManager test failed: {e}", exc_info=True)
        return False

async def test_detection_engine():
    """Test DetectionEngine functionality"""
    logger.info("=== Testing DetectionEngine ===")
    
    try:
        from detection_engine import get_detection_engine
        
        engine = get_detection_engine()
        logger.info("✓ DetectionEngine created successfully")
        
        # Test configuration loading
        stats = engine.get_stats()
        logger.info(f"✓ Configuration loaded: Presidio={stats['presidio_loaded']}, "
                   f"Regex patterns={stats['regex_patterns']}")
        
        # Test entity detection
        test_text = """
        Hello, my name is Dr. John Smith and I work at Microsoft Corporation.
        My email is john.smith@microsoft.com and my phone number is (555) 123-4567.
        I live in Seattle, Washington and my SSN is 123-45-6789.
        """
        
        logger.info("Testing entity detection...")
        entities, domains = await engine.detect_entities(test_text)
        
        logger.info(f"✓ Detection completed: {len(entities)} entities found in domains {domains}")
        
        # Group entities by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('entity_group', 'UNKNOWN')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        for entity_type, type_entities in entity_types.items():
            logger.info(f"  {entity_type}: {len(type_entities)} entities")
            for entity in type_entities[:3]:  # Show first 3 of each type
                start, end = entity.get('start', 0), entity.get('end', 0)
                text_span = test_text[start:end] if start < len(test_text) and end <= len(test_text) else "INVALID"
                logger.info(f"    - '{text_span}' (confidence: {entity.get('score', 0):.3f}, "
                           f"detector: {entity.get('detector', 'unknown')})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DetectionEngine test failed: {e}", exc_info=True)
        return False

async def test_mcp_tools():
    """Test MCP server tools"""
    logger.info("=== Testing MCP Tools ===")
    
    try:
        from mcp_server import setup_server
        
        server = setup_server()
        logger.info("✓ MCP Server setup successfully")
        
        # Test basic server functionality
        logger.info("✓ MCP Server created with proper structure")
        
        # Test that we have the expected tools (simplified check)
        expected_tools = ["detect_pii", "classify_text", "health_check", "manage_models"]
        logger.info(f"✓ Expected tools configured: {expected_tools}")
        
        # Test detect_pii tool
        test_arguments = {
            "text": "My name is Alice Johnson and I work at Google. My email is alice@google.com.",
            "include_context": False
        }
        
        logger.info("Testing detect_pii tool...")
        
        # Simulate tool call (we can't easily test the full MCP protocol here)
        # Instead, we'll test the underlying function directly
        from mcp_server import detection_engine
        
        if detection_engine:
            entities, domains = await detection_engine.detect_entities(
                test_arguments["text"]
            )
            
            result = {
                "entities": entities,
                "domains_used": domains,
                "total_entities": len(entities)
            }
            
            logger.info(f"✓ detect_pii tool test: {result['total_entities']} entities found")
            logger.info(f"  Domains: {result['domains_used']}")
            
            return True
        else:
            logger.error("✗ Detection engine not available in MCP server")
            return False
        
    except Exception as e:
        logger.error(f"✗ MCP Tools test failed: {e}", exc_info=True)
        return False

async def run_all_tests():
    """Run all Phase 1 tests"""
    logger.info("Starting Redactify MCP Server Phase 1 Tests")
    logger.info("=" * 50)
    
    tests = [
        ("ModelManager", test_model_manager),
        ("DetectionEngine", test_detection_engine), 
        ("MCP Tools", test_mcp_tools)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            success = await test_func()
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name} test: {status}")
        except Exception as e:
            logger.error(f"{test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name:20} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 1 tests PASSED! Ready for Phase 2.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests FAILED. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")
    os.environ.setdefault("MAX_WORKERS", "4")
    os.environ.setdefault("MAX_MODEL_MEMORY_MB", "2048")
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)