#!/usr/bin/env python3
"""
Complete System Test

This script tests the entire MCP system end-to-end.
"""

import requests
import json
import time

def test_complete_system():
    """Test the complete MCP system"""
    print("🧪 TESTING COMPLETE REDACTIFY MCP SYSTEM")
    print("=" * 50)
    
    # Test data
    test_text = "John Smith works at Acme Corp. His email is john@acme.com and phone is 555-123-4567."
    
    try:
        # Test main server endpoint
        print("📡 Testing main server integration...")
        response = requests.post(
            "http://localhost:8000/anonymize",
            json={
                "text": test_text,
                "full_redaction": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Main server integration successful!")
            print(f"📝 Original: {test_text}")
            print(f"🔒 Anonymized: {data.get('anonymized_text', 'N/A')}")
            print(f"📊 Entities found: {len(data.get('entities', []))}")
            print(f"⏱️  Processing time: {data.get('processing_time', 0):.3f}s")
            print(f"🏷️  Domains detected: {data.get('domains_detected', [])}")
            
            # Show detected entities
            entities = data.get('entities', [])
            if entities:
                print("\n🔍 Detected Entities:")
                for i, entity in enumerate(entities[:5], 1):
                    entity_type = entity.get('entity_group', 'UNKNOWN')
                    score = entity.get('score', 0)
                    detector = entity.get('detector', 'unknown')
                    print(f"  {i}. {entity_type} (confidence: {score:.3f}, detector: {detector})")
            
            print("\n🎉 COMPLETE SYSTEM TEST PASSED!")
            return True
            
        else:
            print(f"❌ Main server error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("🔴 Main server not running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print("\n✅ ALL TESTS PASSED - MCP SYSTEM IS WORKING!")
    else:
        print("\n❌ TESTS FAILED - CHECK SYSTEM STATUS")