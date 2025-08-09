#!/usr/bin/env python3
"""
Redactify MCP System Startup Script

This script provides a unified entry point for starting the entire Redactify
MCP system with proper orchestration and monitoring.

Usage:
    python start_all.py                    # Production mode
    python start_all.py --dev              # Development mode
    python start_all.py --config config.json  # Custom configuration
    python start_all.py --status           # Show system status
    python start_all.py --stop             # Stop running system

Features:
- Intelligent startup sequencing
- Development and production modes
- Configuration management
- System status monitoring
- Graceful shutdown handling
- Error recovery and reporting
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from process_orchestrator import ProcessOrchestrator, OrchestrationConfig, create_orchestrator
from mcp_server_manager import MCPServerManager

# Configure logging
def setup_logging(level: str = "INFO", development_mode: bool = False):
    """Setup logging configuration"""
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if not development_mode else
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('redactify_mcp.log')
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    REDACTIFY MCP SYSTEM                      ║
║                                                              ║
║  Advanced PII Detection & Anonymization                     ║
║  Distributed MCP Architecture                               ║
║  Version 2.0                                                ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

async def start_system(
    config_file: Optional[str] = None,
    development_mode: bool = False,
    wait_for_input: bool = False
):
    """Start the complete MCP system"""
    print_banner()
    
    # Setup logging
    log_level = "DEBUG" if development_mode else "INFO"
    setup_logging(log_level, development_mode)
    
    logger = logging.getLogger("StartupScript")
    
    try:
        # Create orchestrator
        logger.info("Initializing Process Orchestrator...")
        config = OrchestrationConfig(
            config_file=config_file,
            development_mode=development_mode,
            log_level=log_level,
            auto_restart_on_failure=True
        )
        
        orchestrator = ProcessOrchestrator(config)
        
        # Start system
        async with orchestrator:
            logger.info("System started successfully!")
            
            if development_mode:
                print("\n" + "="*60)
                print("DEVELOPMENT MODE ACTIVE")
                print("- Auto-restart enabled")
                print("- Detailed logging enabled")
                print("- Press Ctrl+C to stop")
                print("="*60 + "\n")
            
            # Wait for shutdown signal or user input
            if wait_for_input:
                print("Press Enter to stop the system...")
                await asyncio.get_event_loop().run_in_executor(None, input)
            else:
                # Wait for shutdown signal
                await orchestrator.shutdown_event.wait()
        
        logger.info("System shutdown completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"System startup failed: {e}", exc_info=True)
        sys.exit(1)

async def show_status():
    """Show system status"""
    print("Checking system status...")
    
    # Try to connect to running system
    # This is a simplified status check - in production you might want
    # to implement a proper status endpoint or shared state mechanism
    
    try:
        # Check if MCP servers are running by attempting connections
        from mcp_client import MCPClient, MCPServerConfig
        
        servers = [
            ("general", 3001),
            ("medical", 3002),
            ("technical", 3003),
            ("legal", 3004),
            ("financial", 3005),
            ("pii_specialized", 3006)
        ]
        
        print("\nMCP SERVER STATUS:")
        print("-" * 50)
        
        for name, port in servers:
            try:
                config = MCPServerConfig(name, port=port)
                async with MCPClient(config) as client:
                    health = await client.health_check()
                    print(f"✓ {name:<20} RUNNING  (Port: {port})")
            except Exception:
                print(f"✗ {name:<20} STOPPED  (Port: {port})")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"Error checking status: {e}")

async def stop_system():
    """Stop running system"""
    print("Stopping Redactify MCP System...")
    
    # This is a simplified stop mechanism
    # In production, you might want to implement proper IPC or signal handling
    
    try:
        # Send shutdown signals to known processes
        import psutil
        
        # Find Python processes running MCP servers
        stopped_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' or proc.info['name'] == 'python.exe':
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'ner_agent.py' in cmdline or 'start_all.py' in cmdline:
                        print(f"Stopping process {proc.info['pid']}: {cmdline}")
                        proc.terminate()
                        stopped_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if stopped_count > 0:
            print(f"Sent stop signals to {stopped_count} processes")
            print("Waiting for graceful shutdown...")
            await asyncio.sleep(5)
        else:
            print("No running MCP processes found")
            
    except Exception as e:
        print(f"Error stopping system: {e}")

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "startup_timeout": 120.0,
        "shutdown_timeout": 60.0,
        "health_check_interval": 30.0,
        "development_mode": False,
        "auto_restart_on_failure": True,
        "log_level": "INFO",
        "servers": [
            {
                "name": "general",
                "script_path": "a2a_ner_general/general_ner_agent.py",
                "port": 3001,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            },
            {
                "name": "medical",
                "script_path": "a2a_ner_medical/medical_ner_agent.py",
                "port": 3002,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            },
            {
                "name": "technical",
                "script_path": "a2a_ner_technical/technical_ner_agent.py",
                "port": 3003,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            },
            {
                "name": "legal",
                "script_path": "a2a_ner_legal/legal_ner_agent.py",
                "port": 3004,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            },
            {
                "name": "financial",
                "script_path": "a2a_ner_financial/financial_ner_agent.py",
                "port": 3005,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            },
            {
                "name": "pii_specialized",
                "script_path": "a2a_ner_pii_specialized/pii_specialized_ner_agent.py",
                "port": 3006,
                "enabled": True,
                "auto_restart": True,
                "max_restarts": 5
            }
        ]
    }
    
    config_file = "mcp_system_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sample configuration created: {config_file}")

async def run_tests():
    """Run basic system tests"""
    print("Running system tests...")
    
    try:
        # Start system in test mode
        config = OrchestrationConfig(
            development_mode=True,
            startup_timeout=60.0,
            wait_for_health=True
        )
        
        orchestrator = ProcessOrchestrator(config)
        
        print("Starting system for testing...")
        await orchestrator.startup()
        
        # Run basic tests
        print("Running health checks...")
        status = orchestrator.get_system_status()
        
        if status["state"] == "running":
            print("✓ System is running")
        else:
            print(f"✗ System state: {status['state']}")
        
        print(f"✓ {status['running_servers']}/{status['total_servers']} servers running")
        
        # Test client connections
        if orchestrator.client_manager:
            print("Testing client connections...")
            health_results = await orchestrator.client_manager.health_check_all()
            
            for server_name, result in health_results.items():
                status_icon = "✓" if result["status"] == "healthy" else "✗"
                print(f"{status_icon} {server_name}: {result['status']}")
        
        print("Shutting down test system...")
        await orchestrator.shutdown()
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Tests failed: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Redactify MCP System Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_all.py                    # Start in production mode
  python start_all.py --dev              # Start in development mode
  python start_all.py --config my.json   # Use custom configuration
  python start_all.py --status           # Show system status
  python start_all.py --stop             # Stop running system
  python start_all.py --test             # Run system tests
        """
    )
    
    parser.add_argument(
        '--dev', '--development',
        action='store_true',
        help='Run in development mode with detailed logging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop running system and exit'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run system tests'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create sample configuration file'
    )
    
    parser.add_argument(
        '--wait-for-input',
        action='store_true',
        help='Wait for user input before stopping (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return
    
    if args.status:
        asyncio.run(show_status())
        return
    
    if args.stop:
        asyncio.run(stop_system())
        return
    
    if args.test:
        asyncio.run(run_tests())
        return
    
    # Start the system
    try:
        asyncio.run(start_system(
            config_file=args.config,
            development_mode=args.dev,
            wait_for_input=args.wait_for_input
        ))
    except KeyboardInterrupt:
        print("\nShutdown completed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()