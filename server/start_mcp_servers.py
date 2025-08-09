#!/usr/bin/env python3
"""
Start all MCP servers for Redactify

This script starts all the MCP (Model Context Protocol) servers on their designated ports:
- General NER: Port 3001
- Medical NER: Port 3002  
- Technical NER: Port 3003
- Legal NER: Port 3004
- Financial NER: Port 3005
- PII Specialized: Port 3006
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCPServerStarter")

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

def start_mcp_server(server_config):
    """Start a single MCP server"""
    name = server_config["name"]
    script = server_config["script"]
    port = server_config["port"]
    env_var = server_config["env_var"]
    
    logger.info(f"Starting {name} on port {port}...")
    
    # Set environment variable for the port
    env = os.environ.copy()
    env[env_var] = str(port)
    
    # Check if script exists
    script_path = Path(script)
    if not script_path.exists():
        logger.error(f"Script not found: {script}")
        return None
    
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"✓ {name} started with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"✗ Failed to start {name}: {e}")
        return None

def main():
    """Start all MCP servers"""
    logger.info("=== Starting Redactify MCP Servers ===")
    
    processes = []
    
    for server_config in MCP_SERVERS:
        process = start_mcp_server(server_config)
        if process:
            processes.append((server_config["name"], process))
        
        # Small delay between starts
        time.sleep(1)
    
    if not processes:
        logger.error("No servers started successfully!")
        return 1
    
    logger.info(f"✓ Started {len(processes)} MCP servers")
    logger.info("=== MCP Servers Running ===")
    
    for name, process in processes:
        logger.info(f"  {name}: PID {process.pid}")
    
    logger.info("\nPress Ctrl+C to stop all servers...")
    
    try:
        # Wait for all processes
        while True:
            time.sleep(1)
            
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    logger.warning(f"{name} has stopped (exit code: {process.returncode})")
                    
    except KeyboardInterrupt:
        logger.info("\nShutting down MCP servers...")
        
        # Terminate all processes
        for name, process in processes:
            try:
                process.terminate()
                logger.info(f"✓ Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        # Wait for processes to terminate
        for name, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}...")
                process.kill()
        
        logger.info("All MCP servers stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())