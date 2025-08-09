#!/usr/bin/env python3
"""
Simple test to verify MCP servers can be started and are accessible
"""

import subprocess
import time
import requests
import os
import sys

def test_single_server():
    """Test starting a single MCP server"""
    print("=== Testing Single MCP Server Startup ===")
    
    # Set environment variable
    env = os.environ.copy()
    env['A2A_GENERAL_PORT'] = '3001'
    
    print("1. Starting General NER server on port 3001...")
    process = subprocess.Popen(
        [sys.executable, 'a2a_ner_general/general_ner_agent.py'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("2. Waiting 10 seconds for server to initialize...")
    time.sleep(10)
    
    try:
        # Test health endpoint
        print("3. Testing health endpoint...")
        response = requests.get('http://localhost:3001/health', timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed: {health_data['status']}")
            print(f"  Agent ID: {health_data['agent_id']}")
            print(f"  Model loaded: {health_data['model_loaded']}")
        else:
            print(f"✗ Health check failed: HTTP {response.status_code}")
            return False
        
        # Test MCP endpoint
        print("4. Testing MCP endpoint...")
        mcp_request = {
            'jsonrpc': '2.0',
            'method': 'predict',
            'params': {
                'inputs': 'John Smith works at Microsoft Corporation',
                'parameters': {}
            },
            'id': 'test-123'
        }
        
        response = requests.post(
            'http://localhost:3001/mcp',
            json=mcp_request,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and 'entities' in result['result']:
                entities = result['result']['entities']
                print(f"✓ MCP endpoint working: Found {len(entities)} entities")
                for entity in entities[:3]:  # Show first 3 entities
                    print(f"  - {entity.get('label', 'UNKNOWN')}: {entity.get('word', 'N/A')}")
            else:
                print("✗ MCP endpoint returned unexpected format")
                return False
        else:
            print(f"✗ MCP endpoint failed: HTTP {response.status_code}")
            return False
        
        print("✓ All tests passed!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    finally:
        # Clean up
        print("5. Stopping server...")
        try:
            process.terminate()
            process.wait(timeout=5)
            print("✓ Server stopped cleanly")
        except subprocess.TimeoutExpired:
            process.kill()
            print("✓ Server force killed")

def main():
    """Run the startup test"""
    print("Redactify MCP Server Startup Test")
    print("=" * 40)
    
    success = test_single_server()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 SUCCESS: MCP server startup is working correctly!")
        print("\nNext steps:")
        print("1. Run: python start_mcp_servers.py")
        print("2. Wait for all servers to start")
        print("3. Run: python server.py")
        return 0
    else:
        print("❌ FAILED: MCP server startup has issues")
        print("\nTroubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Verify Python environment is correct")
        print("3. Check for port conflicts")
        return 1

if __name__ == "__main__":
    sys.exit(main())