#!/usr/bin/env python3
"""
Development Helper Script for Redactify MCP System

This script provides easy development workflows:
- Start individual MCP servers for testing
- Start the complete system in development mode
- Run quick tests and health checks
- Monitor system status in real-time

Usage:
    python dev_start.py                     # Start complete system
    python dev_start.py --server general    # Start single server
    python dev_start.py --test              # Run tests
    python dev_start.py --monitor           # Monitor system
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from start_all import start_system, show_status, run_tests
from mcp_server_manager import MCPServerManager, MCPServerInfo, create_server_info
from mcp_client import MCPClient, MCPServerConfig

# Setup development logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

async def start_single_server(server_name: str):
    """Start a single MCP server for testing"""
    print(f"Starting single MCP server: {server_name}")
    
    # Server configurations
    server_configs = {
        "general": create_server_info("general", "a2a_ner_general/general_ner_agent.py", 3001),
        "medical": create_server_info("medical", "a2a_ner_medical/medical_ner_agent.py", 3002),
        "technical": create_server_info("technical", "a2a_ner_technical/technical_ner_agent.py", 3003),
        "legal": create_server_info("legal", "a2a_ner_legal/legal_ner_agent.py", 3004),
        "financial": create_server_info("financial", "a2a_ner_financial/financial_ner_agent.py", 3005),
        "pii_specialized": create_server_info("pii_specialized", "a2a_ner_pii_specialized/pii_specialized_ner_agent.py", 3006),
    }
    
    if server_name not in server_configs:
        print(f"Unknown server: {server_name}")
        print(f"Available servers: {list(server_configs.keys())}")
        return
    
    server_info = server_configs[server_name]
    
    try:
        # Create manager with single server
        manager = MCPServerManager()
        manager.add_server(server_info)
        
        async with manager:
            # Start the server
            success = await manager.start_server(server_name)
            
            if success:
                print(f"✓ Server {server_name} started successfully on port {server_info.port}")
                print("Press Ctrl+C to stop...")
                
                # Wait for shutdown
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print(f"\nStopping {server_name}...")
            else:
                print(f"✗ Failed to start server {server_name}")
                
    except Exception as e:
        print(f"Error: {e}")

async def test_single_server(server_name: str, port: int):
    """Test a single MCP server"""
    print(f"Testing MCP server: {server_name} on port {port}")
    
    try:
        config = MCPServerConfig(server_name, port=port)
        
        async with MCPClient(config) as client:
            # Test health check
            print("Testing health check...")
            health = await client.health_check()
            print(f"✓ Health check passed: {health}")
            
            # Test prediction
            print("Testing prediction...")
            test_text = "John Smith works at Acme Corp. His email is john@acme.com."
            result = await client.predict(test_text)
            print(f"✓ Prediction successful: {len(result.get('entities', []))} entities found")
            
            # Print entities
            for entity in result.get('entities', []):
                print(f"  - {entity.get('entity_group', 'UNKNOWN')}: {entity.get('word', 'N/A')} (score: {entity.get('score', 0):.3f})")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")

async def monitor_system():
    """Monitor system status in real-time"""
    print("Monitoring Redactify MCP System...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print("\n" + "="*60)
            print(f"System Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            await show_status()
            
            print("\nWaiting 10 seconds for next check...")
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

async def run_quick_test():
    """Run a quick test of the system"""
    print("Running quick system test...")
    
    # Test individual servers
    servers = [
        ("general", 3001),
        ("medical", 3002),
        ("technical", 3003),
        ("legal", 3004),
        ("financial", 3005),
        ("pii_specialized", 3006)
    ]
    
    print("\nTesting individual MCP servers:")
    print("-" * 40)
    
    for name, port in servers:
        try:
            await test_single_server(name, port)
            print(f"✓ {name} server test passed")
        except Exception as e:
            print(f"✗ {name} server test failed: {e}")
        print()
    
    print("Quick test completed!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Development helper for Redactify MCP System"
    )
    
    parser.add_argument(
        '--server',
        type=str,
        help='Start single MCP server (general, medical, technical, legal, financial, pii_specialized)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick system tests'
    )
    
    parser.add_argument(
        '--test-server',
        type=str,
        help='Test specific server'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Monitor system status in real-time'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current system status'
    )
    
    args = parser.parse_args()
    
    try:
        if args.server:
            asyncio.run(start_single_server(args.server))
        elif args.test:
            asyncio.run(run_quick_test())
        elif args.test_server:
            # Default ports for testing
            port_map = {
                "general": 3001,
                "medical": 3002,
                "technical": 3003,
                "legal": 3004,
                "financial": 3005,
                "pii_specialized": 3006
            }
            port = port_map.get(args.test_server, 3001)
            asyncio.run(test_single_server(args.test_server, port))
        elif args.monitor:
            asyncio.run(monitor_system())
        elif args.status:
            asyncio.run(show_status())
        else:
            # Start complete system in development mode
            print("Starting complete system in development mode...")
            asyncio.run(start_system(development_mode=True, wait_for_input=True))
            
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()