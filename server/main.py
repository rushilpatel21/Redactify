#!/usr/bin/env python3
"""
Main entry point for Redactify MCP Server

This script starts the MCP server using the standard stdio transport.
It can be used directly or called by MCP clients.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the server directory to Python path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from mcp_server import main

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)  # Log to stderr to avoid interfering with MCP stdio
        ]
    )
    
    logger = logging.getLogger("RedactifyMain")
    logger.info("Starting Redactify MCP Server...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)