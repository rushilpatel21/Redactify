"""
Process Orchestrator for Redactify MCP Architecture

This module provides high-level orchestration of the entire MCP ecosystem:
- Coordinated startup/shutdown sequences
- Dependency management between services
- System-wide health monitoring
- Graceful error handling and recovery
- Development and production mode support
- Configuration management
- Logging coordination

Features:
- Intelligent startup sequencing
- Dependency-aware shutdown
- System health monitoring
- Configuration validation
- Development mode helpers
- Production deployment support
- Comprehensive error handling
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import yaml

from mcp_server_manager import MCPServerManager, MCPServerInfo, create_standard_manager
from mcp_client import MCPClientManager

logger = logging.getLogger("ProcessOrchestrator")

class SystemState(Enum):
    """Overall system states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    DEGRADED = "degraded"  # Some services failed but system still functional

@dataclass
class OrchestrationConfig:
    """Configuration for the process orchestrator"""
    # Startup configuration
    startup_timeout: float = 120.0
    startup_check_interval: float = 2.0
    wait_for_health: bool = True
    
    # Shutdown configuration
    shutdown_timeout: float = 60.0
    force_kill_timeout: float = 10.0
    
    # Health monitoring
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    max_consecutive_failures: int = 3
    
    # Development mode
    development_mode: bool = False
    auto_restart_on_failure: bool = True
    log_level: str = "INFO"
    
    # Configuration files
    config_file: Optional[str] = None
    servers_config_file: Optional[str] = None

class ProcessOrchestrator:
    """
    High-level orchestrator for the entire MCP system.
    
    Responsibilities:
    - Coordinate startup/shutdown of all services
    - Monitor system health and handle failures
    - Manage configuration and dependencies
    - Provide unified interface for system management
    - Handle graceful degradation scenarios
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        self.state = SystemState.STOPPED
        self.start_time: Optional[float] = None
        
        # Core components
        self.server_manager: Optional[MCPServerManager] = None
        self.client_manager: Optional[MCPClientManager] = None
        
        # System monitoring
        self.health_failures: Dict[str, int] = {}
        self.last_health_check: Optional[float] = None
        self.system_metrics: Dict[str, Any] = {}
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self._shutdown_handlers: List[Callable] = []
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Process Orchestrator initialized")
    
    def _load_configuration(self):
        """Load configuration from files"""
        if self.config.config_file:
            try:
                config_path = Path(self.config.config_file)
                if config_path.exists():
                    with open(config_path) as f:
                        if config_path.suffix.lower() in ['.yaml', '.yml']:
                            config_data = yaml.safe_load(f)
                        else:
                            config_data = json.load(f)
                    
                    # Update config with loaded values
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.startup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
    
    async def startup(self):
        """
        Orchestrated startup sequence for the entire system
        
        Sequence:
        1. Validate configuration
        2. Initialize server manager
        3. Start MCP servers
        4. Initialize client manager
        5. Verify system health
        6. Start monitoring tasks
        """
        if self.state != SystemState.STOPPED:
            logger.warning(f"System is not stopped (current state: {self.state.value})")
            return
        
        logger.info("=== Starting Redactify MCP System ===")
        self.state = SystemState.STARTING
        self.start_time = time.time()
        
        try:
            # Step 1: Validate configuration
            logger.info("Step 1: Validating configuration...")
            await self._validate_configuration()
            
            # Step 2: Initialize server manager
            logger.info("Step 2: Initializing MCP server manager...")
            self.server_manager = await self._create_server_manager()
            await self.server_manager.start_manager()
            
            # Step 3: Start MCP servers
            logger.info("Step 3: Starting MCP servers...")
            start_results = await self.server_manager.start_all_servers()
            
            # Check if any critical servers failed
            failed_servers = [name for name, success in start_results.items() if not success]
            if failed_servers:
                logger.warning(f"Failed to start servers: {failed_servers}")
                
                # Determine if system can continue
                if len(failed_servers) == len(start_results):
                    raise RuntimeError("All MCP servers failed to start")
                else:
                    logger.warning("Some servers failed, continuing with degraded functionality")
                    self.state = SystemState.DEGRADED
            
            # Step 4: Wait for servers to be healthy
            if self.config.wait_for_health:
                logger.info("Step 4: Waiting for servers to become healthy...")
                await self._wait_for_system_health()
            
            # Step 5: Initialize client manager
            logger.info("Step 5: Initializing MCP client manager...")
            self.client_manager = self.server_manager.client_manager
            
            # Step 6: Start monitoring tasks
            logger.info("Step 6: Starting system monitoring...")
            await self._start_monitoring()
            
            # Step 7: Setup signal handlers
            self._setup_signal_handlers()
            
            if self.state != SystemState.DEGRADED:
                self.state = SystemState.RUNNING
            
            startup_time = time.time() - self.start_time
            logger.info(f"=== System startup complete in {startup_time:.2f}s ===")
            logger.info(f"System state: {self.state.value}")
            
            # Print system status
            await self._print_system_status()
            
        except Exception as e:
            logger.error(f"System startup failed: {e}", exc_info=True)
            self.state = SystemState.FAILED
            await self._cleanup_on_failure()
            raise
    
    async def shutdown(self, force: bool = False):
        """
        Orchestrated shutdown sequence for the entire system
        
        Sequence:
        1. Signal shutdown intent
        2. Stop monitoring tasks
        3. Stop MCP servers gracefully
        4. Cleanup resources
        5. Final status report
        """
        if self.state == SystemState.STOPPED:
            logger.info("System is already stopped")
            return
        
        logger.info("=== Shutting down Redactify MCP System ===")
        self.state = SystemState.STOPPING
        self.shutdown_event.set()
        
        try:
            # Step 1: Stop monitoring tasks
            logger.info("Step 1: Stopping monitoring tasks...")
            await self._stop_monitoring()
            
            # Step 2: Run shutdown handlers
            logger.info("Step 2: Running shutdown handlers...")
            for handler in self._shutdown_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Error in shutdown handler: {e}")
            
            # Step 3: Stop MCP servers
            if self.server_manager:
                logger.info("Step 3: Stopping MCP servers...")
                await self.server_manager.stop_all_servers()
                await self.server_manager.stop_manager()
            
            # Step 4: Cleanup client manager
            if self.client_manager and self.client_manager != self.server_manager.client_manager:
                logger.info("Step 4: Cleaning up client manager...")
                await self.client_manager.__aexit__(None, None, None)
            
            self.state = SystemState.STOPPED
            shutdown_time = time.time() - (self.start_time or time.time())
            logger.info(f"=== System shutdown complete (uptime: {shutdown_time:.2f}s) ===")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            if force:
                logger.warning("Force shutdown requested, terminating...")
                os._exit(1)
    
    async def restart(self):
        """Restart the entire system"""
        logger.info("Restarting system...")
        await self.shutdown()
        await asyncio.sleep(2.0)  # Brief pause
        await self.startup()
    
    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server"""
        if not self.server_manager:
            logger.error("Server manager not initialized")
            return False
        
        return await self.server_manager.restart_server(server_name)
    
    async def _create_server_manager(self) -> MCPServerManager:
        """Create and configure the MCP server manager"""
        if self.config.servers_config_file:
            # Load custom server configuration
            manager = MCPServerManager(
                health_check_interval=self.config.health_check_interval,
                enable_auto_restart=self.config.auto_restart_on_failure
            )
            
            # Load server configurations from file
            config_path = Path(self.config.servers_config_file)
            if config_path.exists():
                with open(config_path) as f:
                    servers_config = json.load(f)
                
                for server_config in servers_config.get("servers", []):
                    server_info = MCPServerInfo(**server_config)
                    manager.add_server(server_info)
            else:
                logger.warning(f"Servers config file not found: {config_path}")
                return await create_standard_manager()
        else:
            # Use standard configuration
            manager = await create_standard_manager()
        
        return manager
    
    async def _validate_configuration(self):
        """Validate system configuration"""
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ is required")
        
        # Check required directories
        base_dir = Path(__file__).parent
        required_dirs = [
            "a2a_ner_general",
            "a2a_ner_medical", 
            "a2a_ner_technical",
            "a2a_ner_legal",
            "a2a_ner_financial",
            "a2a_ner_pii_specialized"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not (base_dir / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            logger.warning(f"Missing MCP server directories: {missing_dirs}")
        
        # Validate configuration values
        if self.config.startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        
        if self.config.shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")
        
        logger.info("Configuration validation passed")
    
    async def _wait_for_system_health(self):
        """Wait for the system to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < self.config.startup_timeout:
            try:
                if self.server_manager and self.server_manager.client_manager:
                    health_results = await self.server_manager.client_manager.health_check_all()
                    
                    healthy_servers = [
                        name for name, result in health_results.items()
                        if result["status"] == "healthy"
                    ]
                    
                    total_servers = len(health_results)
                    if len(healthy_servers) == total_servers:
                        logger.info("All servers are healthy")
                        return
                    elif len(healthy_servers) > 0:
                        logger.info(f"{len(healthy_servers)}/{total_servers} servers healthy")
                    else:
                        logger.warning("No servers are healthy yet")
                
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
            
            await asyncio.sleep(self.config.startup_check_interval)
        
        # Timeout reached
        logger.warning("System health check timeout reached")
        if self.server_manager:
            running_servers = self.server_manager.get_running_servers()
            if running_servers:
                logger.info(f"Continuing with {len(running_servers)} running servers")
                self.state = SystemState.DEGRADED
            else:
                raise RuntimeError("No servers are running after startup timeout")
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        self._metrics_task = asyncio.create_task(self._metrics_collector())
        
        logger.info("System monitoring started")
    
    async def _stop_monitoring(self):
        """Stop background monitoring tasks"""
        tasks = [self._health_monitor_task, self._metrics_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("System monitoring stopped")
    
    async def _health_monitor(self):
        """Background health monitoring task"""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10.0)
    
    async def _perform_health_check(self):
        """Perform comprehensive system health check"""
        if not self.server_manager or not self.server_manager.client_manager:
            return
        
        try:
            health_results = await asyncio.wait_for(
                self.server_manager.client_manager.health_check_all(),
                timeout=self.config.health_check_timeout
            )
            
            self.last_health_check = time.time()
            
            # Process health results
            healthy_count = 0
            for server_name, result in health_results.items():
                if result["status"] == "healthy":
                    healthy_count += 1
                    self.health_failures[server_name] = 0
                else:
                    self.health_failures[server_name] = self.health_failures.get(server_name, 0) + 1
                    
                    # Handle consecutive failures
                    if self.health_failures[server_name] >= self.config.max_consecutive_failures:
                        logger.error(f"Server {server_name} has failed {self.health_failures[server_name]} consecutive health checks")
                        
                        if self.config.auto_restart_on_failure:
                            logger.info(f"Attempting to restart {server_name}")
                            await self.server_manager.restart_server(server_name)
            
            # Update system state based on health
            total_servers = len(health_results)
            if healthy_count == total_servers:
                if self.state == SystemState.DEGRADED:
                    self.state = SystemState.RUNNING
                    logger.info("System recovered to full health")
            elif healthy_count > 0:
                if self.state == SystemState.RUNNING:
                    self.state = SystemState.DEGRADED
                    logger.warning("System degraded - some servers unhealthy")
            else:
                logger.error("All servers are unhealthy")
                self.state = SystemState.FAILED
            
        except asyncio.TimeoutError:
            logger.warning("Health check timed out")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _metrics_collector(self):
        """Background metrics collection task"""
        while not self.shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60.0)  # Collect metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30.0)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        if not self.server_manager:
            return
        
        # Collect server statistics
        server_stats = {}
        if self.server_manager.client_manager:
            server_stats = self.server_manager.client_manager.get_all_stats()
        
        # Collect process information
        process_stats = self.server_manager.get_all_status()
        
        # System-wide metrics
        self.system_metrics = {
            "timestamp": time.time(),
            "uptime": time.time() - (self.start_time or time.time()),
            "state": self.state.value,
            "server_count": len(self.server_manager.servers),
            "running_servers": len(self.server_manager.get_running_servers()),
            "failed_servers": len(self.server_manager.get_failed_servers()),
            "server_stats": server_stats,
            "process_stats": process_stats,
            "last_health_check": self.last_health_check
        }
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != "win32":  # Unix-like systems
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
    
    async def _cleanup_on_failure(self):
        """Cleanup resources when startup fails"""
        try:
            if self.server_manager:
                await self.server_manager.stop_manager()
            
            await self._stop_monitoring()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _print_system_status(self):
        """Print comprehensive system status"""
        if not self.server_manager:
            return
        
        print("\n" + "="*60)
        print("REDACTIFY MCP SYSTEM STATUS")
        print("="*60)
        print(f"System State: {self.state.value.upper()}")
        print(f"Uptime: {time.time() - (self.start_time or time.time()):.1f}s")
        print(f"Total Servers: {len(self.server_manager.servers)}")
        print(f"Running Servers: {len(self.server_manager.get_running_servers())}")
        print(f"Failed Servers: {len(self.server_manager.get_failed_servers())}")
        
        print("\nSERVER STATUS:")
        print("-" * 60)
        for name, status in self.server_manager.get_all_status().items():
            state_icon = "✓" if status["state"] == "running" else "✗"
            print(f"{state_icon} {name:<20} {status['state']:<10} Port: {status['port']}")
        
        print("="*60 + "\n")
    
    def add_shutdown_handler(self, handler: Callable):
        """Add a custom shutdown handler"""
        self._shutdown_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "state": self.state.value,
            "uptime": time.time() - (self.start_time or time.time()) if self.start_time else 0,
            "last_health_check": self.last_health_check,
            "metrics": self.system_metrics
        }
        
        if self.server_manager:
            status.update({
                "total_servers": len(self.server_manager.servers),
                "running_servers": len(self.server_manager.get_running_servers()),
                "failed_servers": len(self.server_manager.get_failed_servers()),
                "server_status": self.server_manager.get_all_status()
            })
        
        return status
    
    async def save_system_report(self, filepath: str):
        """Save comprehensive system report"""
        report = {
            "timestamp": time.time(),
            "system_status": self.get_system_status(),
            "configuration": {
                "startup_timeout": self.config.startup_timeout,
                "shutdown_timeout": self.config.shutdown_timeout,
                "health_check_interval": self.config.health_check_interval,
                "development_mode": self.config.development_mode,
                "auto_restart_on_failure": self.config.auto_restart_on_failure
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"System report saved to: {filepath}")

# Convenience functions

async def create_orchestrator(
    config_file: Optional[str] = None,
    development_mode: bool = False
) -> ProcessOrchestrator:
    """Create process orchestrator with configuration"""
    config = OrchestrationConfig(
        config_file=config_file,
        development_mode=development_mode,
        log_level="DEBUG" if development_mode else "INFO"
    )
    
    return ProcessOrchestrator(config)

async def quick_start(development_mode: bool = False) -> ProcessOrchestrator:
    """Quick start with default configuration"""
    orchestrator = await create_orchestrator(development_mode=development_mode)
    await orchestrator.startup()
    return orchestrator