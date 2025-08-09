#!/usr/bin/env python3
"""
Test script to verify MCP servers can start correctly
"""

import os
import sys
import subprocess
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCPStartupTest")

# Server configurations
MCP_SERVERS = [
    {
        "name": "General NER",
        "script": "a2a_ner_general/general_ner_agent.py",
        "port": 3001,
        "env_var": "A2A_GENERAL_PORT"
    },
    {
        "name": "Medical NER", 
        "script": "a2a_ner_medical/medical_ner_agent.py",
        "port": 3002,
        "env_var": "A2A_MEDICAL_PORT"
    },
    {
        "name": "Technical NER",
        "script": "a2a_ner_technical/technical_ner_agent.py", 
        "port": 3003,
        "env_var": "A2A_TECHNICAL_PORT"
    },
    {
        "name": "Legal NER",
        "script": "a2a_ner_legal/legal_ner_agent.py",
        "port": 3004, 
        "env_var": "A2A_LEGAL_PORT"
    },
    {
        "name": "Financial NER",
        "script": "a2a_ner_financial/financial_ner_agent.py",
        "port": 3005,
        "env_var": "A2A_FINANCIAL_PORT"
    },
    {
        "name": "PII Specialized",
        "script": "a2a_ner_pii_specialized/pii_specialized_ner_agent.py",
        "port": 3006,
        "env_var": "A2A_PII_SPECIALIZED_PORT"
    }
]

def test_server_startup(server_config):
    """Test if a server can start and respond to health checks"""
    name = server_config["name"]
    script = server_config["script"]
    port = server_config["port"]
    env_var = server_config["env_var"]
    
    logger.info(f"Testing {name} startup...")
    
    # Check if script exists
    script_path = Path(script)
    if not script_path.exists():
        logger.error(f"✗ Script not found: {script}")
        return False
    
    # Set environment variable
    env = os.environ.copy()
    env[env_var] = str(port)
    
    process = None
    try:
        # Start the server
        logger.info(f"  Starting {name} on port {port}...")
        process = subprocess.Popen(
            [sys.executable, script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        logger.info(f"  Waiting for {name} to initialize...")
        time.sleep(10)  # Give time for model loading
        
        # Test health endpoint
        health_url = f"http://localhost:{port}/health"
        logger.info(f"  Testing health endpoint: {health_url}")
        
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✓ {name} is healthy: {health_data.get('status', 'unknown')}")
            return True
        else:
            logger.error(f"✗ {name} health check failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ {name} connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {name} startup failed: {e}")
        return False
    finally:
        # Clean up process
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"  Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.info(f"  Force killed {name}")
            except Exception as e:
                logger.error(f"  Error stopping {name}: {e}")

def main():
    """Test all MCP servers"""
    logger.info("=== MCP Server Startup Test ===")
    
    results = {}
    
    for server_config in MCP_SERVERS:
        success = test_server_startup(server_config)
        results[server_config["name"]] = success
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    logger.info("\n=== Test Results ===")
    successful = 0
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {name}: {status}")
        if success:
            successful += 1
    
    logger.info(f"\nSummary: {successful}/{len(results)} servers tested successfully")
    
    if successful == len(results):
        logger.info("🎉 All MCP servers can start correctly!")
        return 0
    else:
        logger.error("❌ Some MCP servers failed to start")
        return 1

if __name__ == "__main__":
    sys.exit(main())