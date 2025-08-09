#!/usr/bin/env python3
"""
Test the automatic MCP server startup functionality
"""

import asyncio
import logging
import requests
import time
from auto_mcp_manager import get_auto_mcp_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoStartupTest")

async def test_auto_startup():
    """Test automatic MCP server startup"""
    logger.info("=== Testing Automatic MCP Server Startup ===")
    
    manager = get_auto_mcp_manager()
    
    try:
        # Test starting all servers
        logger.info("Starting all MCP servers...")
        success = await manager.start_all_servers(timeout=120.0)
        
        if success:
            logger.info("✓ All servers started successfully!")
            
            # Test health of each server
            logger.info("Testing server health...")
            health_results = await manager.check_all_health()
            
            for name, is_healthy in health_results.items():
                status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
                logger.info(f"  {name}: {status}")
            
            # Get detailed status
            status = manager.get_server_status()
            logger.info("\nDetailed Status:")
            for name, info in status.items():
                logger.info(f"  {name}: PID={info['pid']}, Port={info['port']}, Uptime={info['uptime']:.1f}s")
            
            # Test a sample request to one server
            logger.info("\nTesting sample request to General NER...")
            try:
                response = requests.get("http://localhost:3001/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✓ Sample request successful")
                else:
                    logger.error(f"✗ Sample request failed: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"✗ Sample request failed: {e}")
            
        else:
            logger.error("✗ Failed to start servers")
            return False
        
        # Test shutdown
        logger.info("\nTesting shutdown...")
        await manager.shutdown_all_servers()
        logger.info("✓ Shutdown completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

async def main():
    """Run the test"""
    success = await test_auto_startup()
    
    if success:
        print("\n🎉 SUCCESS: Automatic MCP startup is working!")
        print("\nYou can now run 'python server.py' and all MCP servers will start automatically!")
    else:
        print("\n❌ FAILED: Automatic MCP startup has issues")
        print("\nCheck the logs above for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)