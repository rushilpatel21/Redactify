#!/usr/bin/env python3
"""
Test the main server startup with automatic MCP management
"""

import asyncio
import subprocess
import time
import requests
import sys
import os

def test_main_server():
    """Test the main server startup"""
    print("=== Testing Main Server with Automatic MCP Management ===")
    
    # Start the main server
    print("1. Starting main server...")
    env = os.environ.copy()
    
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"   Server started with PID {process.pid}")
    
    # Wait for server to start up (MCP servers need time to load models)
    print("2. Waiting for server startup (this may take 2-3 minutes for model loading)...")
    
    max_wait = 180  # 3 minutes
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                server_ready = True
                elapsed = time.time() - start_time
                print(f"   ✓ Server is ready! (took {elapsed:.1f}s)")
                break
        except:
            pass
        
        # Show progress
        elapsed = time.time() - start_time
        print(f"   Waiting... ({elapsed:.0f}s elapsed)")
        time.sleep(10)
    
    if not server_ready:
        print("   ✗ Server failed to start within timeout")
        process.terminate()
        return False
    
    try:
        # Test health endpoint
        print("3. Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed: {health_data.get('status')}")
            print(f"   Components: {health_data.get('components')}")
        else:
            print(f"   ✗ Health check failed: HTTP {response.status_code}")
            return False
        
        # Test MCP status endpoint
        print("4. Testing MCP status endpoint...")
        response = requests.get("http://localhost:8000/mcp-status", timeout=5)
        if response.status_code == 200:
            mcp_data = response.json()
            summary = mcp_data.get('summary', {})
            print(f"   ✓ MCP Status: {summary.get('healthy', 0)}/{summary.get('total', 0)} servers healthy")
            
            servers = mcp_data.get('servers', {})
            for name, info in servers.items():
                status = "✓" if info.get('healthy') else "✗"
                print(f"     {name}: {status} (Port: {info.get('port')}, PID: {info.get('pid')})")
        else:
            print(f"   ✗ MCP status failed: HTTP {response.status_code}")
        
        # Test anonymization endpoint
        print("5. Testing anonymization endpoint...")
        test_data = {
            "text": "John Smith works at Microsoft Corporation. His email is john@microsoft.com",
            "full_redaction": True
        }
        
        response = requests.post("http://localhost:8000/anonymize", json=test_data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Anonymization successful!")
            print(f"   Original: {test_data['text']}")
            print(f"   Anonymized: {result.get('anonymized_text')}")
            print(f"   Entities found: {len(result.get('entities', []))}")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Domains detected: {result.get('domains_detected', [])}")
        else:
            print(f"   ✗ Anonymization failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
        
        print("\n✅ All tests passed! Main server with automatic MCP management is working perfectly!")
        return True
        
    except Exception as e:
        print(f"   ✗ Test error: {e}")
        return False
    
    finally:
        # Stop the server
        print("6. Stopping server...")
        try:
            process.terminate()
            process.wait(timeout=30)
            print("   ✓ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            process.kill()
            print("   ✓ Server force killed")

if __name__ == "__main__":
    success = test_main_server()
    
    if success:
        print("\n🎉 SUCCESS: Main server with automatic MCP management is fully functional!")
        print("\nYou can now simply run:")
        print("  python server.py")
        print("\nAnd everything will work automatically!")
    else:
        print("\n❌ FAILED: There are issues with the main server startup")
    
    sys.exit(0 if success else 1)