"""
MCP Server Manager for Redactify

This module provides comprehensive management of MCP server processes:
- Process lifecycle management (start/stop/restart)
- Health monitoring and auto-recovery
- Port management and conflict resolution
- Graceful shutdown handling
- Process monitoring and logging
- Resource usage tracking
- Development mode support

Features:
- Async process management
- Automatic restart on failure
- Process health monitoring
- Resource usage tracking
- Graceful shutdown coordination
- Development-friendly logging
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
import psutil
import aiofiles
from contextlib import asynccontextmanager

from mcp_client import MCPClientManager, MCPServerConfig, MCPClient

logger = logging.getLogger("MCPServerManager")

class ProcessState(Enum):
    """MCP server process states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"

@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    name: str
    script_path: str
    port: int
    host: str = "localhost"
    enabled: bool = True
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay: float = 5.0
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    env_vars: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

@dataclass
class ProcessInfo:
    """Runtime information about a process"""
    pid: Optional[int] = None
    state: ProcessState = ProcessState.STOPPED
    start_time: Optional[float] = None
    restart_count: int = 0
    last_restart_time: Optional[float] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    error_message: Optional[str] = None
    health_status: Optional[Dict[str, Any]] = None
    last_health_check: Optional[float] = None

class MCPServerManager:
    """
    Comprehensive MCP server process manager with advanced features:
    - Process lifecycle management
    - Health monitoring and auto-recovery
    - Resource usage tracking
    - Graceful shutdown handling
    - Development mode support
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        health_check_interval: float = 30.0,
        enable_auto_restart: bool = True
    ):
        self.base_dir = base_dir or Path(__file__).parent
        self.health_check_interval = health_check_interval
        self.enable_auto_restart = enable_auto_restart
        
        # Server configurations and runtime info
        self.servers: Dict[str, MCPServerInfo] = {}
        self.processes: Dict[str, ProcessInfo] = {}
        self.subprocesses: Dict[str, asyncio.subprocess.Process] = {}
        
        # Management state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.client_manager: Optional[MCPClientManager] = None
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        
        # Port management
        self.used_ports: Set[int] = set()
        self.port_range = range(3001, 3100)  # Available port range
        
        logger.info(f"Initialized MCP Server Manager (base_dir: {self.base_dir})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_manager()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_manager()
    
    def add_server(self, server_info: MCPServerInfo):
        """Add an MCP server configuration"""
        # Validate script path
        script_path = Path(server_info.script_path)
        if not script_path.is_absolute():
            script_path = self.base_dir / script_path
        
        if not script_path.exists():
            raise FileNotFoundError(f"MCP server script not found: {script_path}")
        
        # Check port conflicts
        if server_info.port in self.used_ports:
            raise ValueError(f"Port {server_info.port} is already in use")
        
        # Update script path and register
        server_info.script_path = str(script_path)
        self.servers[server_info.name] = server_info
        self.processes[server_info.name] = ProcessInfo()
        self.used_ports.add(server_info.port)
        
        logger.info(f"Added MCP server: {server_info.name} (port {server_info.port})")
    
    def remove_server(self, server_name: str):
        """Remove an MCP server configuration"""
        if server_name in self.servers:
            port = self.servers[server_name].port
            self.used_ports.discard(port)
            del self.servers[server_name]
            del self.processes[server_name]
            logger.info(f"Removed MCP server: {server_name}")
    
    async def start_manager(self):
        """Start the MCP server manager"""
        if self.is_running:
            logger.warning("MCP Server Manager is already running")
            return
        
        logger.info("Starting MCP Server Manager...")
        self.is_running = True
        self.shutdown_event.clear()
        
        # Initialize client manager
        self.client_manager = MCPClientManager()
        await self.client_manager.__aenter__()
        
        # Add clients for all servers
        for server_info in self.servers.values():
            config = MCPServerConfig(
                name=server_info.name,
                host=server_info.host,
                port=server_info.port,
                enabled=server_info.enabled
            )
            self.client_manager.add_server(config)
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._process_monitor())
        self._health_task = asyncio.create_task(self._health_monitor())
        
        logger.info("MCP Server Manager started successfully")
    
    async def stop_manager(self):
        """Stop the MCP server manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping MCP Server Manager...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop all servers
        await self.stop_all_servers()
        
        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        # Close client manager
        if self.client_manager:
            await self.client_manager.__aexit__(None, None, None)
        
        logger.info("MCP Server Manager stopped")
    
    async def start_server(self, server_name: str) -> bool:
        """
        Start a specific MCP server
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        server_info = self.servers[server_name]
        process_info = self.processes[server_name]
        
        if process_info.state in [ProcessState.RUNNING, ProcessState.STARTING]:
            logger.warning(f"Server {server_name} is already running/starting")
            return True
        
        if not server_info.enabled:
            logger.info(f"Server {server_name} is disabled, skipping start")
            return False
        
        logger.info(f"Starting MCP server: {server_name}")
        process_info.state = ProcessState.STARTING
        process_info.error_message = None
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(server_info.env_vars)
            
            # Set the correct environment variable for each server type
            port_env_map = {
                "general": "A2A_GENERAL_PORT",
                "medical": "A2A_MEDICAL_PORT", 
                "technical": "A2A_TECHNICAL_PORT",
                "legal": "A2A_LEGAL_PORT",
                "financial": "A2A_FINANCIAL_PORT",
                "pii_specialized": "A2A_PII_SPECIALIZED_PORT"
            }
            
            port_env_var = port_env_map.get(server_name, "MCP_SERVER_PORT")
            env[port_env_var] = str(server_info.port)
            env["MCP_SERVER_HOST"] = server_info.host
            
            # Start process
            process = await asyncio.create_subprocess_exec(
                sys.executable, server_info.script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            self.subprocesses[server_name] = process
            process_info.pid = process.pid
            process_info.start_time = time.time()
            
            # Wait for startup with timeout
            startup_success = await self._wait_for_startup(
                server_name, server_info.startup_timeout
            )
            
            if startup_success:
                process_info.state = ProcessState.RUNNING
                logger.info(f"MCP server {server_name} started successfully (PID: {process.pid})")
                return True
            else:
                process_info.state = ProcessState.FAILED
                process_info.error_message = "Startup timeout"
                await self._cleanup_process(server_name)
                logger.error(f"MCP server {server_name} failed to start (timeout)")
                return False
                
        except Exception as e:
            process_info.state = ProcessState.FAILED
            process_info.error_message = str(e)
            logger.error(f"Failed to start MCP server {server_name}: {e}", exc_info=True)
            return False
    
    async def stop_server(self, server_name: str, force: bool = False) -> bool:
        """
        Stop a specific MCP server
        
        Args:
            server_name: Name of the server to stop
            force: If True, use SIGKILL instead of SIGTERM
            
        Returns:
            True if server stopped successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        process_info = self.processes[server_name]
        
        if process_info.state == ProcessState.STOPPED:
            logger.info(f"Server {server_name} is already stopped")
            return True
        
        logger.info(f"Stopping MCP server: {server_name}")
        process_info.state = ProcessState.STOPPING
        
        try:
            process = self.subprocesses.get(server_name)
            if not process:
                process_info.state = ProcessState.STOPPED
                return True
            
            # Send termination signal
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Wait for graceful shutdown
            server_info = self.servers[server_name]
            try:
                await asyncio.wait_for(
                    process.wait(), 
                    timeout=server_info.shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Server {server_name} didn't stop gracefully, forcing...")
                process.kill()
                await process.wait()
            
            await self._cleanup_process(server_name)
            process_info.state = ProcessState.STOPPED
            logger.info(f"MCP server {server_name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server {server_name}: {e}", exc_info=True)
            return False
    
    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server"""
        logger.info(f"Restarting MCP server: {server_name}")
        
        process_info = self.processes[server_name]
        process_info.state = ProcessState.RESTARTING
        process_info.restart_count += 1
        process_info.last_restart_time = time.time()
        
        # Stop then start
        await self.stop_server(server_name)
        
        # Wait for restart delay
        server_info = self.servers[server_name]
        await asyncio.sleep(server_info.restart_delay)
        
        return await self.start_server(server_name)
    
    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all enabled MCP servers"""
        logger.info("Starting all MCP servers...")
        
        results = {}
        start_tasks = []
        
        for server_name, server_info in self.servers.items():
            if server_info.enabled:
                task = asyncio.create_task(self.start_server(server_name))
                start_tasks.append((server_name, task))
        
        # Wait for all servers to start
        for server_name, task in start_tasks:
            try:
                results[server_name] = await task
            except Exception as e:
                logger.error(f"Error starting {server_name}: {e}")
                results[server_name] = False
        
        successful_starts = sum(results.values())
        total_servers = len(results)
        
        logger.info(f"Started {successful_starts}/{total_servers} MCP servers")
        return results
    
    async def stop_all_servers(self) -> Dict[str, bool]:
        """Stop all MCP servers"""
        logger.info("Stopping all MCP servers...")
        
        results = {}
        stop_tasks = []
        
        for server_name in self.servers.keys():
            task = asyncio.create_task(self.stop_server(server_name))
            stop_tasks.append((server_name, task))
        
        # Wait for all servers to stop
        for server_name, task in stop_tasks:
            try:
                results[server_name] = await task
            except Exception as e:
                logger.error(f"Error stopping {server_name}: {e}")
                results[server_name] = False
        
        logger.info("All MCP servers stopped")
        return results
    
    async def _wait_for_startup(self, server_name: str, timeout: float) -> bool:
        """Wait for server to become healthy after startup"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if self.client_manager:
                    client = self.client_manager.get_client(server_name)
                    await client.health_check()
                    return True
            except Exception:
                pass  # Server not ready yet
            
            await asyncio.sleep(1.0)
        
        return False
    
    async def _cleanup_process(self, server_name: str):
        """Clean up process resources"""
        if server_name in self.subprocesses:
            del self.subprocesses[server_name]
        
        process_info = self.processes[server_name]
        process_info.pid = None
        process_info.cpu_percent = 0.0
        process_info.memory_mb = 0.0
    
    async def _process_monitor(self):
        """Background task to monitor process health and resources"""
        while self.is_running:
            try:
                for server_name, process_info in self.processes.items():
                    if process_info.state == ProcessState.RUNNING and process_info.pid:
                        try:
                            # Get process info
                            proc = psutil.Process(process_info.pid)
                            process_info.cpu_percent = proc.cpu_percent()
                            process_info.memory_mb = proc.memory_info().rss / 1024 / 1024
                            
                            # Check if process is still alive
                            if not proc.is_running():
                                logger.warning(f"Process {server_name} (PID {process_info.pid}) died")
                                process_info.state = ProcessState.FAILED
                                await self._handle_process_failure(server_name)
                        
                        except psutil.NoSuchProcess:
                            logger.warning(f"Process {server_name} (PID {process_info.pid}) no longer exists")
                            process_info.state = ProcessState.FAILED
                            await self._handle_process_failure(server_name)
                        
                        except Exception as e:
                            logger.error(f"Error monitoring process {server_name}: {e}")
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process monitor: {e}")
                await asyncio.sleep(5.0)
    
    async def _health_monitor(self):
        """Background task for health monitoring"""
        while self.is_running:
            try:
                if self.client_manager:
                    health_results = await self.client_manager.health_check_all()
                    
                    for server_name, health_info in health_results.items():
                        process_info = self.processes.get(server_name)
                        if process_info:
                            process_info.health_status = health_info
                            process_info.last_health_check = time.time()
                            
                            # Handle unhealthy servers
                            if health_info["status"] == "unhealthy":
                                logger.warning(f"Server {server_name} is unhealthy")
                                if process_info.state == ProcessState.RUNNING:
                                    await self._handle_process_failure(server_name)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10.0)
    
    async def _handle_process_failure(self, server_name: str):
        """Handle process failure with auto-restart logic"""
        server_info = self.servers[server_name]
        process_info = self.processes[server_name]
        
        if not server_info.auto_restart or not self.enable_auto_restart:
            logger.info(f"Auto-restart disabled for {server_name}")
            return
        
        if process_info.restart_count >= server_info.max_restarts:
            logger.error(
                f"Server {server_name} exceeded max restarts "
                f"({server_info.max_restarts}), giving up"
            )
            return
        
        logger.info(f"Auto-restarting failed server: {server_name}")
        await self.restart_server(server_name)
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific server"""
        if server_name not in self.servers:
            return None
        
        server_info = self.servers[server_name]
        process_info = self.processes[server_name]
        
        return {
            "name": server_name,
            "enabled": server_info.enabled,
            "host": server_info.host,
            "port": server_info.port,
            "state": process_info.state.value,
            "pid": process_info.pid,
            "start_time": process_info.start_time,
            "restart_count": process_info.restart_count,
            "last_restart_time": process_info.last_restart_time,
            "cpu_percent": process_info.cpu_percent,
            "memory_mb": process_info.memory_mb,
            "error_message": process_info.error_message,
            "health_status": process_info.health_status,
            "last_health_check": process_info.last_health_check,
            "uptime": (
                time.time() - process_info.start_time 
                if process_info.start_time else 0
            )
        }
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all servers"""
        return {
            name: self.get_server_status(name)
            for name in self.servers.keys()
        }
    
    def get_running_servers(self) -> List[str]:
        """Get list of running server names"""
        return [
            name for name, process_info in self.processes.items()
            if process_info.state == ProcessState.RUNNING
        ]
    
    def get_failed_servers(self) -> List[str]:
        """Get list of failed server names"""
        return [
            name for name, process_info in self.processes.items()
            if process_info.state == ProcessState.FAILED
        ]
    
    async def save_status_report(self, filepath: str):
        """Save detailed status report to file"""
        report = {
            "timestamp": time.time(),
            "manager_status": {
                "is_running": self.is_running,
                "total_servers": len(self.servers),
                "running_servers": len(self.get_running_servers()),
                "failed_servers": len(self.get_failed_servers())
            },
            "servers": self.get_all_status()
        }
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(report, indent=2, default=str))
        
        logger.info(f"Status report saved to: {filepath}")

# Convenience functions

def create_server_info(
    name: str,
    script_path: str,
    port: int,
    **kwargs
) -> MCPServerInfo:
    """Create MCPServerInfo with defaults"""
    return MCPServerInfo(
        name=name,
        script_path=script_path,
        port=port,
        **kwargs
    )

async def create_standard_manager() -> MCPServerManager:
    """Create manager with standard Redactify MCP servers"""
    manager = MCPServerManager()
    
    # Standard server configurations with proper environment variables
    servers = [
        create_server_info("general", "a2a_ner_general/general_ner_agent.py", 3001, 
                          env_vars={"A2A_GENERAL_PORT": "3001"}),
        create_server_info("medical", "a2a_ner_medical/medical_ner_agent.py", 3002,
                          env_vars={"A2A_MEDICAL_PORT": "3002"}),
        create_server_info("technical", "a2a_ner_technical/technical_ner_agent.py", 3003,
                          env_vars={"A2A_TECHNICAL_PORT": "3003"}),
        create_server_info("legal", "a2a_ner_legal/legal_ner_agent.py", 3004,
                          env_vars={"A2A_LEGAL_PORT": "3004"}),
        create_server_info("financial", "a2a_ner_financial/financial_ner_agent.py", 3005,
                          env_vars={"A2A_FINANCIAL_PORT": "3005"}),
        create_server_info("pii_specialized", "a2a_ner_pii_specialized/pii_specialized_ner_agent.py", 3006,
                          env_vars={"A2A_PII_SPECIALIZED_PORT": "3006"}),
    ]
    
    for server_info in servers:
        try:
            manager.add_server(server_info)
        except FileNotFoundError as e:
            logger.warning(f"Skipping server {server_info.name}: {e}")
    
    return manager