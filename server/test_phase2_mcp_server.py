#!/usr/bin/env python3
"""
Test script for Redactify MCP Server Phase 2

This script tests the Phase 2 implementation including:
- All NER model wrappers
- Anonymization engine
- New MCP tools (anonymize_text, verify_entities, process_batch)
- Enhanced functionality
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
logger = logging.getLogger("Phase2MCPServerTest")

async def test_all_model_wrappers():
    """Test all NER model wrappers"""
    logger.info("=== Testing All Model Wrappers ===")
    
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        logger.info("✓ ModelManager created successfully")
        
        # Test text for all domains
        test_texts = {
            "general": "Hello, my name is John Doe and I work at Microsoft Corporation.",
            "medical": "Patient John Smith (MRN: 12345) was admitted to General Hospital on 01/15/2024. Dr. Sarah Johnson prescribed medication for his condition.",
            "technical": "API key: sk-1234567890abcdef. Server IP: 192.168.1.100. Database connection string: mongodb://user:pass@localhost:27017/db",
            "legal": "Case No. 2024-CV-001234 in the Superior Court of California. Attorney John Legal (Bar No. 123456) represents the plaintiff.",
            "financial": "Account number: 1234567890. Routing number: 021000021. Credit card: 4111-1111-1111-1111. SWIFT: CHASUS33.",
            "pii_specialized": "SSN: 123-45-6789. Email: john@example.com. Phone: (555) 123-4567. Driver's License: D1234567."
        }
        
        model_results = {}
        
        for model_name, test_text in test_texts.items():
            logger.info(f"Testing {model_name} model...")
            
            try:
                model_wrapper = await manager.get_model(model_name)
                if model_wrapper:
                    entities = model_wrapper.predict(test_text)
                    model_results[model_name] = {
                        "status": "success",
                        "entities_found": len(entities),
                        "entity_types": list(set(e.get('entity_group', 'UNKNOWN') for e in entities)),
                        "model_info": model_wrapper.get_info()
                    }
                    logger.info(f"✓ {model_name} model: {len(entities)} entities found")
                    
                    # Show sample entities
                    for entity in entities[:3]:  # Show first 3
                        start, end = entity.get('start', 0), entity.get('end', 0)
                        text_span = test_text[start:end] if start < len(test_text) and end <= len(test_text) else "INVALID"
                        logger.info(f"  - {entity.get('entity_group', 'UNKNOWN')}: '{text_span}' (confidence: {entity.get('score', 0):.3f})")
                else:
                    model_results[model_name] = {
                        "status": "failed",
                        "error": "Model wrapper not loaded"
                    }
                    logger.error(f"✗ {model_name} model failed to load")
                    
            except Exception as e:
                model_results[model_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {model_name} model error: {e}")
        
        # Summary
        successful_models = sum(1 for r in model_results.values() if r["status"] == "success")
        total_models = len(model_results)
        
        logger.info(f"Model wrapper testing completed: {successful_models}/{total_models} models successful")
        
        return successful_models == total_models, model_results
        
    except Exception as e:
        logger.error(f"✗ Model wrapper testing failed: {e}", exc_info=True)
        return False, {}

async def test_anonymization_engine():
    """Test anonymization engine functionality"""
    logger.info("=== Testing Anonymization Engine ===")
    
    try:
        from anonymization_engine import get_anonymization_engine
        from detection_engine import get_detection_engine
        
        anon_engine = get_anonymization_engine()
        detection_engine = get_detection_engine()
        
        logger.info("✓ Anonymization engine created successfully")
        
        # Test text with various PII types
        test_text = """
        Hello, my name is Dr. Alice Johnson and I work at General Hospital.
        My email is alice.johnson@hospital.com and my phone is (555) 123-4567.
        Patient record: MRN-789012. SSN: 123-45-6789.
        Credit card: 4111-1111-1111-1111. Account: 9876543210.
        """
        
        # First detect entities
        logger.info("Detecting entities for anonymization test...")
        entities, domains = await detection_engine.detect_entities(test_text)
        logger.info(f"✓ Detected {len(entities)} entities for anonymization")
        
        # Test different anonymization strategies
        strategies = ["pseudonymize", "mask", "redact", "custom"]
        anonymization_results = {}
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy...")
            
            try:
                result = anon_engine.anonymize_text(
                    text=test_text,
                    entities=entities,
                    strategy=strategy,
                    preserve_format=True
                )
                
                anonymization_results[strategy] = {
                    "status": "success",
                    "entities_processed": len(result["entities_processed"]),
                    "original_length": result["processing_metadata"]["original_length"],
                    "anonymized_length": result["processing_metadata"]["anonymized_length"],
                    "sample_anonymized": result["anonymized_text"][:100] + "..." if len(result["anonymized_text"]) > 100 else result["anonymized_text"]
                }
                
                logger.info(f"✓ {strategy} strategy: {len(result['entities_processed'])} entities processed")
                
            except Exception as e:
                anonymization_results[strategy] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {strategy} strategy failed: {e}")
        
        # Test batch anonymization
        logger.info("Testing batch anonymization...")
        batch_texts = [
            "John Doe's email is john@example.com",
            "Jane Smith's phone is (555) 987-6543",
            "Account number: 1234567890"
        ]
        
        batch_entities = []
        for text in batch_texts:
            text_entities, _ = await detection_engine.detect_entities(text)
            batch_entities.append(text_entities)
        
        try:
            batch_results = anon_engine.batch_anonymize(
                texts=batch_texts,
                entities_list=batch_entities,
                strategy="pseudonymize"
            )
            
            logger.info(f"✓ Batch anonymization: {len(batch_results)} texts processed")
            anonymization_results["batch"] = {
                "status": "success",
                "texts_processed": len(batch_results)
            }
            
        except Exception as e:
            logger.error(f"✗ Batch anonymization failed: {e}")
            anonymization_results["batch"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check engine stats
        stats = anon_engine.get_stats()
        logger.info(f"✓ Anonymization engine stats: {stats['supported_strategies']}")
        
        successful_strategies = sum(1 for r in anonymization_results.values() if r["status"] == "success")
        total_strategies = len(anonymization_results)
        
        logger.info(f"Anonymization testing completed: {successful_strategies}/{total_strategies} strategies successful")
        
        return successful_strategies >= 3, anonymization_results  # At least 3 strategies should work
        
    except Exception as e:
        logger.error(f"✗ Anonymization engine testing failed: {e}", exc_info=True)
        return False, {}

async def test_new_mcp_tools():
    """Test new MCP tools (anonymize_text, verify_entities, process_batch)"""
    logger.info("=== Testing New MCP Tools ===")
    
    try:
        from mcp_server import setup_server, detection_engine, anonymization_engine
        
        server = setup_server()
        logger.info("✓ MCP Server setup successfully")
        
        # Test data
        test_text = "My name is John Doe, email: john@example.com, phone: (555) 123-4567"
        
        # Test anonymize_text tool
        logger.info("Testing anonymize_text tool...")
        try:
            # Get engines directly since globals might not be set
            from detection_engine import get_detection_engine
            from anonymization_engine import get_anonymization_engine
            
            det_engine = get_detection_engine()
            anon_engine = get_anonymization_engine()
            
            if det_engine and anon_engine:
                entities, domains = await det_engine.detect_entities(test_text)
                
                anon_result = anon_engine.anonymize_text(
                    text=test_text,
                    entities=entities,
                    strategy="pseudonymize"
                )
                
                logger.info(f"✓ anonymize_text tool: {len(anon_result['entities_processed'])} entities anonymized")
                anonymize_success = True
            else:
                logger.error("✗ anonymize_text tool: engines not available")
                anonymize_success = False
                
        except Exception as e:
            logger.error(f"✗ anonymize_text tool failed: {e}")
            anonymize_success = False
        
        # Test verify_entities tool
        logger.info("Testing verify_entities functionality...")
        try:
            det_engine = get_detection_engine()
            if det_engine:
                entities, _ = await det_engine.detect_entities(test_text)
                
                # Simulate verification
                verified_count = 0
                for entity in entities:
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    if 0 <= start < len(test_text) and start < end <= len(test_text):
                        verified_count += 1
                
                logger.info(f"✓ verify_entities functionality: {verified_count}/{len(entities)} entities verified")
                verify_success = verified_count > 0
            else:
                logger.error("✗ verify_entities functionality: detection engine not available")
                verify_success = False
                
        except Exception as e:
            logger.error(f"✗ verify_entities functionality failed: {e}")
            verify_success = False
        
        # Test batch processing functionality
        logger.info("Testing batch processing functionality...")
        try:
            batch_texts = [
                "Alice Smith works at Tech Corp",
                "Bob Johnson's email is bob@company.com",
                "Carol Davis lives in New York"
            ]
            
            batch_results = []
            det_engine = get_detection_engine()
            for text in batch_texts:
                if det_engine:
                    entities, domains = await det_engine.detect_entities(text)
                    batch_results.append({
                        "entities": entities,
                        "domains": domains,
                        "entity_count": len(entities)
                    })
            
            total_entities = sum(r["entity_count"] for r in batch_results)
            logger.info(f"✓ batch processing functionality: {len(batch_results)} texts processed, {total_entities} total entities")
            batch_success = len(batch_results) == len(batch_texts)
            
        except Exception as e:
            logger.error(f"✗ batch processing functionality failed: {e}")
            batch_success = False
        
        # Summary
        tools_tested = ["anonymize_text", "verify_entities", "batch_processing"]
        tools_success = [anonymize_success, verify_success, batch_success]
        successful_tools = sum(tools_success)
        
        logger.info(f"New MCP tools testing completed: {successful_tools}/{len(tools_tested)} tools successful")
        
        return successful_tools >= 2, {  # At least 2 tools should work
            "anonymize_text": anonymize_success,
            "verify_entities": verify_success,
            "batch_processing": batch_success
        }
        
    except Exception as e:
        logger.error(f"✗ New MCP tools testing failed: {e}", exc_info=True)
        return False, {}

async def test_enhanced_detection():
    """Test enhanced detection with all models"""
    logger.info("=== Testing Enhanced Detection ===")
    
    try:
        from detection_engine import get_detection_engine
        
        engine = get_detection_engine()
        logger.info("✓ Enhanced detection engine loaded")
        
        # Comprehensive test text covering all domains
        comprehensive_text = """
        Medical Report: Patient Dr. Sarah Johnson (MRN: MED-789012) was treated at General Hospital.
        Contact: sarah.johnson@hospital.com, phone (555) 123-4567.
        
        Technical Details: API key sk-1234567890abcdef, server IP 192.168.1.100.
        Database: mongodb://admin:secret@db.company.com:27017/prod
        
        Legal Case: Case No. 2024-CV-001234, Attorney John Legal (Bar: 123456).
        Court: Superior Court of California, Judge Maria Rodriguez presiding.
        
        Financial Info: Account 9876543210, Routing 021000021.
        Credit Card: 4111-1111-1111-1111, SWIFT: CHASUS33.
        
        Personal Data: SSN 123-45-6789, Driver's License D1234567.
        Address: 123 Main St, Anytown, CA 90210.
        """
        
        logger.info("Running comprehensive entity detection...")
        # Test with multiple domains since we don't have OpenAI classification
        test_domains = ["general", "medical", "technical", "financial", "pii_specialized"]
        entities, domains = await engine.detect_entities(comprehensive_text, domains=test_domains)
        
        # Analyze results
        entity_types = {}
        detectors_used = set()
        
        for entity in entities:
            entity_type = entity.get('entity_group', 'UNKNOWN')
            detector = entity.get('detector', 'unknown')
            
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1
            detectors_used.add(detector)
        
        logger.info(f"✓ Enhanced detection completed:")
        logger.info(f"  - Total entities: {len(entities)}")
        logger.info(f"  - Entity types: {len(entity_types)}")
        logger.info(f"  - Domains detected: {domains}")
        logger.info(f"  - Detectors used: {len(detectors_used)}")
        
        # Show entity type breakdown
        for entity_type, count in sorted(entity_types.items()):
            logger.info(f"  - {entity_type}: {count} entities")
        
        # Verify we're getting entities from multiple detectors
        expected_detectors = ["presidio_internal", "regex_internal", "model_general"]
        found_expected = sum(1 for detector in expected_detectors if any(detector in d for d in detectors_used))
        
        success = (
            len(entities) >= 10 and  # Should find at least 10 entities
            len(entity_types) >= 5 and  # Should find at least 5 different types
            len(domains) >= 2 and  # Should detect multiple domains
            found_expected >= 2  # Should use multiple detector types
        )
        
        logger.info(f"Enhanced detection test: {'PASSED' if success else 'FAILED'}")
        
        return success, {
            "total_entities": len(entities),
            "entity_types": len(entity_types),
            "domains_detected": domains,
            "detectors_used": list(detectors_used),
            "entity_breakdown": entity_types
        }
        
    except Exception as e:
        logger.error(f"✗ Enhanced detection testing failed: {e}", exc_info=True)
        return False, {}

async def run_phase2_tests():
    """Run all Phase 2 tests"""
    logger.info("Starting Redactify MCP Server Phase 2 Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Model Wrappers", test_all_model_wrappers),
        ("Anonymization Engine", test_anonymization_engine),
        ("New MCP Tools", test_new_mcp_tools),
        ("Enhanced Detection", test_enhanced_detection)
    ]
    
    results = {}
    detailed_results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            success, details = await test_func()
            results[test_name] = success
            detailed_results[test_name] = details
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name} test: {status}")
        except Exception as e:
            logger.error(f"{test_name} test FAILED with exception: {e}")
            results[test_name] = False
            detailed_results[test_name] = {"error": str(e)}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name:25} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Detailed results summary
    logger.info("\nDetailed Results:")
    for test_name, details in detailed_results.items():
        if isinstance(details, dict) and "error" not in details:
            logger.info(f"{test_name}:")
            for key, value in details.items():
                if isinstance(value, (int, float, str, list)):
                    logger.info(f"  {key}: {value}")
    
    if passed == total:
        logger.info("\n🎉 All Phase 2 tests PASSED! MCP server conversion complete!")
        return True
    else:
        logger.error(f"\n❌ {total - passed} tests FAILED. Phase 2 needs attention.")
        return False

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")
    os.environ.setdefault("MAX_WORKERS", "4")
    os.environ.setdefault("MAX_MODEL_MEMORY_MB", "4096")  # Increased for Phase 2
    
    # Run Phase 2 tests
    success = asyncio.run(run_phase2_tests())
    sys.exit(0 if success else 1)