#!/usr/bin/env python3
"""
Test lightweight MCP startup with fewer servers
"""

import asyncio
import logging
import time
from auto_mcp_manager import MCPServerProcess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LightweightTest")

async def test_lightweight_startup():
    """Test starting just 2 servers to verify the system works"""
    logger.info("=== Testing Lightweight MCP Startup ===")
    
    # Create just 2 servers for testing
    servers = {
        "general": MCPServerProcess("general", "a2a_ner_general/general_ner_agent.py", 3001, "A2A_GENERAL_PORT"),
        "medical": MCPServerProcess("medical", "a2a_ner_medical/medical_ner_agent.py", 3002, "A2A_MEDICAL_PORT")
    }
    
    try:
        # Start servers with delays
        logger.info("Starting servers...")
        for name, server in servers.items():
            logger.info(f"Starting {name}...")
            success = server.start()
            if success:
                logger.info(f"✓ {name} started with PID {server.process.pid}")
            else:
                logger.error(f"✗ Failed to start {name}")
                return False
            
            # Wait between starts
            await asyncio.sleep(5)
        
        # Wait for servers to initialize
        logger.info("Waiting for servers to initialize...")
        await asyncio.sleep(20)
        
        # Check health
        logger.info("Checking server health...")
        all_healthy = True
        for name, server in servers.items():
            is_healthy = await server.check_health()
            status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            logger.info(f"  {name}: {status}")
            if not is_healthy:
                all_healthy = False
        
        if all_healthy:
            logger.info("✓ All servers are healthy!")
            
            # Test a simple request
            logger.info("Testing MCP request...")
            import requests
            
            try:
                response = requests.post(
                    "http://localhost:3001/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "predict",
                        "params": {
                            "inputs": "John Smith works at Microsoft",
                            "parameters": {}
                        },
                        "id": "test-123"
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    entities = result.get("result", {}).get("entities", [])
                    logger.info(f"✓ MCP request successful: Found {len(entities)} entities")
                else:
                    logger.error(f"✗ MCP request failed: HTTP {response.status_code}")
                    all_healthy = False
                    
            except Exception as e:
                logger.error(f"✗ MCP request error: {e}")
                all_healthy = False
        
        # Cleanup
        logger.info("Cleaning up...")
        for name, server in servers.items():
            if server.is_running:
                server.stop()
                logger.info(f"✓ Stopped {name}")
        
        return all_healthy
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_lightweight_startup()
    
    if success:
        print("\n🎉 SUCCESS: Lightweight MCP startup is working!")
        print("The issue might be resource contention with all 6 servers.")
        print("Consider:")
        print("1. Increasing system RAM")
        print("2. Starting servers with longer delays")
        print("3. Using fewer servers initially")
    else:
        print("\n❌ FAILED: Even lightweight startup has issues")
        print("Check individual server logs for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)