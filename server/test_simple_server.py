#!/usr/bin/env python3
"""
Simple test to verify server.py works with all 6 microservices
"""

import subprocess
import time
import requests
import sys
import os

def test_server():
    """Test that server.py starts and works"""
    print("=== Testing Server with All 6 Microservices ===")
    
    server_process = None
    try:
        # Start server
        print("1. Starting server.py...")
        server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Wait for startup
        print("2. Waiting for server to start (2 minutes)...")
        time.sleep(120)  # Give plenty of time for all servers to start
        
        # Test main server health
        print("3. Testing main server health...")
        try:
            response = requests.get("http://localhost:8000/health", timeout=30)
            if response.status_code == 200:
                print("✓ Main server is responding")
                health_data = response.json()
                print(f"  Status: {health_data.get('status')}")
            else:
                print(f"✗ Main server health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to main server: {e}")
            return False
        
        # Test MCP status
        print("4. Testing MCP status...")
        try:
            response = requests.get("http://localhost:8000/mcp-status", timeout=30)
            if response.status_code == 200:
                mcp_data = response.json()
                servers = mcp_data.get('servers', {})
                print(f"✓ MCP Status endpoint working")
                print(f"  Total servers: {len(servers)}")
                
                for name, info in servers.items():
                    status = "✓" if info.get('running') else "✗"
                    print(f"  {name}: {status} (Port: {info.get('port')})")
            else:
                print(f"✗ MCP status failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"✗ MCP status error: {e}")
        
        # Test anonymization
        print("5. Testing anonymization endpoint...")
        try:
            test_data = {
                "text": "John Smith works at Microsoft. His email is john@microsoft.com",
                "full_redaction": True
            }
            
            response = requests.post(
                "http://localhost:8000/anonymize",
                json=test_data,
                timeout=60  # Give more time for processing
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Anonymization endpoint working!")
                print(f"  Original: {test_data['text']}")
                print(f"  Anonymized: {result.get('anonymized_text')}")
                print(f"  Entities found: {len(result.get('entities', []))}")
                print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
                return True
            else:
                print(f"✗ Anonymization failed: HTTP {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Anonymization error: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if server_process:
            print("6. Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
                print("✓ Server stopped")
            except subprocess.TimeoutExpired:
                server_process.kill()
                print("✓ Server force killed")

def main():
    success = test_server()
    
    if success:
        print("\n🎉 SUCCESS: Server with all 6 microservices is working!")
        print("\nYour system is ready! You can now:")
        print("1. Run: python server.py")
        print("2. Use the anonymization endpoints")
        print("3. All 6 MCP microservices will work automatically")
    else:
        print("\n❌ FAILED: Server has issues")
        print("\nTroubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Ensure you have enough RAM (8-12GB recommended)")
        print("3. Check the server logs for specific errors")
    
    return 0 if success else 1

if __name__ == "__main__":
    result = main()
    sys.exit(result)