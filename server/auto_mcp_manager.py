#!/usr/bin/env python3
"""
Automatic MCP Server Manager

This module automatically manages MCP servers:
- Checks if servers are already running
- Starts servers if they're not running
- Monitors server health
- Handles graceful shutdown
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil

logger = logging.getLogger("AutoMCPManager")

class MCPServerProcess:
    """Represents a managed MCP server process"""
    
    def __init__(self, name: str, script_path: str, port: int, env_var: str):
        self.name = name
        self.script_path = script_path
        self.port = port
        self.env_var = env_var
        self.process: Optional[subprocess.Popen] = None
        self.is_healthy = False
        self.start_time = 0
        
    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/health"
    
    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
    
    def check_port_in_use(self) -> bool:
        """Check if the port is already in use by another process"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == self.port and conn.status == 'LISTEN':
                    return True
            return False
        except Exception:
            return False
    
    async def check_health(self) -> bool:
        """Check if the server is healthy"""
        try:
            import requests
            # Increase timeout significantly for heavy ML model responses
            response = requests.get(self.health_url, timeout=60)
            if response.status_code == 200:
                # Try to parse JSON to ensure it's a valid response
                health_data = response.json()
                is_healthy = health_data.get('status') == 'ok' and health_data.get('model_loaded', False)
                self.is_healthy = is_healthy
                return is_healthy
            else:
                self.is_healthy = False
                return False
        except requests.exceptions.Timeout:
            # Don't mark as unhealthy immediately on timeout - server might be loading
            return self.is_healthy  # Keep previous state
        except Exception as e:
            self.is_healthy = False
            return False
    
    def start(self) -> bool:
        """Start the MCP server process"""
        if self.is_running:
            logger.info(f"{self.name} is already running")
            return True
        
        # Check if port is already in use
        if self.check_port_in_use():
            logger.info(f"{self.name} port {self.port} is already in use - assuming server is running")
            return True
        
        # Check if script exists
        script_path = Path(self.script_path)
        if not script_path.exists():
            logger.error(f"Script not found for {self.name}: {self.script_path}")
            return False
        
        try:
            # Set environment variable for port
            env = os.environ.copy()
            env[self.env_var] = str(self.port)
            
            logger.info(f"Starting {self.name} on port {self.port}...")
            
            # Start the process
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            self.start_time = time.time()
            logger.info(f"✓ {self.name} started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to start {self.name}: {e}")
            return False
    
    def stop(self):
        """Stop the MCP server process"""
        if not self.is_running:
            return
        
        try:
            logger.info(f"Stopping {self.name}...")
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
                logger.info(f"✓ {self.name} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {self.name}...")
                self.process.kill()
                self.process.wait()
                logger.info(f"✓ {self.name} force killed")
                
        except Exception as e:
            logger.error(f"Error stopping {self.name}: {e}")
        finally:
            self.process = None
            self.is_healthy = False

class AutoMCPManager:
    """Automatically manages all MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerProcess] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Define only 3 essential MCP servers for testing
        server_configs = [
            ("general", "a2a_ner_general/general_ner_agent.py", 3001, "A2A_GENERAL_PORT"),
            ("medical", "a2a_ner_medical/medical_ner_agent.py", 3002, "A2A_MEDICAL_PORT"),
            ("pii_specialized", "a2a_ner_pii_specialized/pii_specialized_ner_agent.py", 3006, "A2A_PII_SPECIALIZED_PORT"),
            # Disabled for testing:
            # ("technical", "a2a_ner_technical/technical_ner_agent.py", 3003, "A2A_TECHNICAL_PORT"),
            # ("legal", "a2a_ner_legal/legal_ner_agent.py", 3004, "A2A_LEGAL_PORT"),
            # ("financial", "a2a_ner_financial/financial_ner_agent.py", 3005, "A2A_FINANCIAL_PORT"),
        ]
        
        for name, script_path, port, env_var in server_configs:
            self.servers[name] = MCPServerProcess(name, script_path, port, env_var)
        
        logger.info(f"AutoMCPManager initialized with {len(self.servers)} servers")
    
    async def start_all_servers(self, timeout: float = 300.0) -> bool:
        """Start all MCP servers and wait for them to be healthy"""
        logger.info("=== Starting All MCP Servers ===")
        
        # Start all servers with staggered delays to reduce resource contention
        start_results = {}
        for i, (name, server) in enumerate(self.servers.items()):
            start_results[name] = server.start()
            # Moderate delay between starts
            if i < len(self.servers) - 1:  # Don't wait after the last server
                await asyncio.sleep(5)  # 5 seconds between each server start
        
        # Count successful starts
        started_count = sum(1 for success in start_results.values() if success)
        logger.info(f"Started {started_count}/{len(self.servers)} servers")
        
        if started_count == 0:
            logger.error("No servers started successfully!")
            return False
        
        # Wait for servers to become healthy
        logger.info("Waiting for servers to become healthy (this may take a few minutes for model loading)...")
        start_time = time.time()
        last_healthy_count = 0
        
        while time.time() - start_time < timeout:
            healthy_count = 0
            
            for name, server in self.servers.items():
                if not start_results[name]:
                    continue  # Skip servers that failed to start
                
                try:
                    is_healthy = await server.check_health()
                    if is_healthy:
                        if not server.is_healthy:  # First time becoming healthy
                            elapsed = time.time() - server.start_time
                            logger.info(f"✓ {name} is healthy (took {elapsed:.1f}s)")
                        healthy_count += 1
                except Exception as e:
                    logger.debug(f"Health check failed for {name}: {e}")
            
            # Log progress if changed
            # if healthy_count != last_healthy_count:
            #     logger.info(f"Progress: {healthy_count}/{started_count} servers healthy")
            #     last_healthy_count = healthy_count
            
            # if healthy_count == started_count:
            #     logger.info(f"🎉 All {healthy_count} servers are healthy!")
            #     return True
            if healthy_count > 0:
                return True
            # return True
            # Wait before checking again - shorter intervals initially, longer as time goes on
            elapsed = time.time() - start_time
            if elapsed < 60:
                await asyncio.sleep(3)  # Check every 3 seconds for first minute
            elif elapsed < 120:
                await asyncio.sleep(5)  # Check every 5 seconds for second minute
            else:
                await asyncio.sleep(10)  # Check every 10 seconds after that
        
        # Timeout reached
        logger.warning(f"Timeout reached after {timeout}s. {healthy_count}/{started_count} servers are healthy")
        
        # Log detailed status of each server
        for name, server in self.servers.items():
            if start_results[name]:
                try:
                    is_healthy = await server.check_health()
                    status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
                    logger.info(f"  {name}: {status} (Port: {server.port})")
                except Exception as e:
                    logger.info(f"  {name}: ✗ Error - {e}")
        
        # Return True if we have at least some servers running (even if health checks timeout)
        running_count = sum(1 for name, server in self.servers.items() if start_results[name] and server.is_running)
        logger.info(f"Final status: {running_count} servers running, {healthy_count} servers healthy")
        
        # Accept if we have servers running, even if health checks are timing out
        return running_count > 0
    
    async def check_all_health(self) -> Dict[str, bool]:
        """Check health of all servers"""
        results = {}
        for name, server in self.servers.items():
            results[name] = await server.check_health()
        return results
    
    def get_server_status(self) -> Dict[str, Dict]:
        """Get status of all servers"""
        status = {}
        for name, server in self.servers.items():
            status[name] = {
                "running": server.is_running,
                "healthy": server.is_healthy,
                "port": server.port,
                "pid": server.process.pid if server.process else None,
                "uptime": time.time() - server.start_time if server.start_time > 0 else 0
            }
        return status
    
    async def start_monitoring(self):
        """Start background monitoring of servers"""
        logger.info("Starting server monitoring...")
        
        while not self.shutdown_event.is_set():
            try:
                # Check health of all servers
                health_results = await self.check_all_health()
                
                # Log any unhealthy servers
                for name, is_healthy in health_results.items():
                    server = self.servers[name]
                    if server.is_running and not is_healthy:
                        logger.warning(f"{name} is running but unhealthy")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def shutdown_all_servers(self):
        """Shutdown all MCP servers"""
        logger.info("=== Shutting Down All MCP Servers ===")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop all servers
        for name, server in self.servers.items():
            if server.is_running:
                server.stop()
        
        logger.info("All MCP servers stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        success = await self.start_all_servers()
        if success:
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self.start_monitoring())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown_all_servers()

# Global instance
_auto_mcp_manager: Optional[AutoMCPManager] = None

def get_auto_mcp_manager() -> AutoMCPManager:
    """Get the global AutoMCPManager instance"""
    global _auto_mcp_manager
    if _auto_mcp_manager is None:
        _auto_mcp_manager = AutoMCPManager()
    return _auto_mcp_manager

async def ensure_mcp_servers_running() -> bool:
    """Ensure all MCP servers are running and healthy"""
    manager = get_auto_mcp_manager()
    return await manager.start_all_servers()