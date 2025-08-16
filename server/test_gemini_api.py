#!/usr/bin/env python3
"""
Test Gemini API integration
"""

import os
import logging
from dotenv import load_dotenv
from model_manager import get_model_manager

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestGemini")

def test_gemini_api():
    """Test Gemini API integration"""
    logger.info("=== Testing Gemini API Integration ===")
    
    # Check if API key is set
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        logger.info(f"✓ Gemini API Key: SET ({len(api_key)} characters)")
        logger.info(f"  Key starts with: {api_key[:10]}...")
    else:
        logger.warning("✗ Gemini API Key: NOT SET")
        logger.info("  Set GEMINI_API_KEY environment variable to enable Gemini")
        return False
    
    # Test model manager initialization
    try:
        logger.info("Testing ModelManager initialization...")
        manager = get_model_manager()
        logger.info("✓ ModelManager initialized")
        
        # Test Gemini model
        gemini_model = manager.get_gemini_model()
        
        if gemini_model:
            logger.info("✓ Gemini model initialized successfully")
            logger.info(f"  Model type: {type(gemini_model)}")
            
            # Test a simple generation
            logger.info("Testing Gemini text generation...")
            try:
                test_prompt = "Classify this text into domains: John Smith works at Microsoft"
                response = gemini_model.generate_content(test_prompt)
                logger.info("✓ Gemini generation test successful")
                logger.info(f"  Response: {response.text[:100]}...")
                return True
                
            except Exception as e:
                logger.error(f"✗ Gemini generation test failed: {e}")
                return False
        else:
            logger.error("✗ Gemini model not initialized")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error testing Gemini: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_gemini_api()
    
    if success:
        print("\n🎉 SUCCESS: Gemini API is working correctly!")
        print("The system can use Gemini for advanced text classification.")
    else:
        print("\n⚠️  Gemini API not available or not working")
        print("The system will work without Gemini, using fallback classification.")
        print("\nTo enable Gemini:")
        print("1. Get a Gemini API key from Google AI Studio")
        print("2. Set environment variable: GEMINI_API_KEY=your_key_here")
        print("3. Restart the server")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    result = main()
    sys.exit(result)