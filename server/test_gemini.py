#!/usr/bin/env python3
"""
Simple test script to verify Gemini integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

async def test_gemini():
    """Test Gemini classification"""
    try:
        from detection_engine import get_detection_engine
        
        engine = get_detection_engine()
        
        test_text = """
        Patient John Smith was admitted to General Hospital on January 15th, 2024.
        His medical record number is MRN-123456. The doctor prescribed medication
        for his heart condition. Contact: john@email.com, phone: (555) 123-4567.
        """
        
        print("Testing Gemini text classification...")
        classifications = await engine._classify_text(test_text)
        print(f"Classifications: {classifications}")
        
        if "medical" in classifications:
            print("✅ Gemini successfully classified medical text!")
        else:
            print("⚠️ Gemini classification working but didn't detect medical content")
            
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gemini())
    print(f"Gemini test: {'PASSED' if success else 'FAILED'}")