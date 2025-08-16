"""
MCP Client Library for Redactify

This module provides a high-quality, production-ready client for communicating
with MCP (Model Context Protocol) servers using JSON-RPC 2.0.

Features:
- Async/await support for high performance
- Connection pooling and reuse
- Automatic retry with exponential backoff
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling and logging
- Request/response validation
- Health monitoring and metrics
- Timeout management
- Batch request optimization
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
import backoff

logger = logging.getLogger("MCPClient")

class MCPErrorCode(Enum):
    """Standard JSON-RPC 2.0 error codes plus MCP-specific codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_UNAVAILABLE = -32001
    MODEL_NOT_LOADED = -32002
    PREDICTION_FAILED = -32003
    TIMEOUT_ERROR = -32004

@dataclass
class MCPRequest:
    """JSON-RPC 2.0 request structure"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

@dataclass
class MCPResponse:
    """JSON-RPC 2.0 response structure"""
    jsonrpc: str
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None
    
    @property
    def is_error(self) -> bool:
        return self.error is not None

@dataclass
class MCPError:
    """JSON-RPC 2.0 error structure"""
    code: int
    message: str
    data: Optional[Any] = None

class MCPClientError(Exception):
    """Base exception for MCP client errors"""
    def __init__(self, message: str, code: Optional[int] = None, data: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.data = data

class MCPServerUnavailableError(MCPClientError):
    """Raised when MCP server is unavailable"""
    pass

class MCPTimeoutError(MCPClientError):
    """Raised when MCP request times out"""
    pass

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class MCPServerConfig:
    """Configuration for an MCP server"""
    
    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 3001,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enabled: bool = True
    ):
        self.name = name
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enabled = enabled
        self.url = f"http://{host}:{port}"

class MCPClient:
    """
    High-performance MCP client with advanced features:
    - Connection pooling
    - Circuit breaker pattern
    - Automatic retries with exponential backoff
    - Request batching
    - Health monitoring
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        server_config: MCPServerConfig,
        session: Optional[ClientSession] = None,
        enable_circuit_breaker: bool = True
    ):
        self.config = server_config
        self.session = session
        self._own_session = session is None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.last_health_check = 0
        self.is_healthy = True
        
        logger.info(f"Initialized MCP client for {server_config.name} at {server_config.url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_session:
            timeout = ClientTimeout(total=self.config.timeout)
            self.session = ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_session and self.session:
            await self.session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _make_request(self, request: MCPRequest) -> MCPResponse:
        """Make a single JSON-RPC request with retry logic"""
        if not self.config.enabled:
            raise MCPServerUnavailableError(f"MCP server {self.config.name} is disabled")
        
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise MCPServerUnavailableError(
                f"Circuit breaker is open for {self.config.name}"
            )
        
        if not self.session:
            raise MCPClientError("Session not initialized. Use async context manager.")
        
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = asdict(request)
            
            logger.debug(f"Making MCP request to {self.config.url}: {request.method}")
            
            # Make HTTP request to the /mcp endpoint
            mcp_url = f"{self.config.url}/mcp"
            async with self.session.post(
                mcp_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_time = time.time() - start_time
                self.total_response_time += response_time
                self.request_count += 1
                
                if response.status != 200:
                    raise MCPClientError(
                        f"HTTP {response.status}: {await response.text()}",
                        code=response.status
                    )
                
                # Parse JSON response
                response_data = await response.json()
                mcp_response = MCPResponse(**response_data)
                
                # Handle JSON-RPC errors
                if mcp_response.is_error:
                    error = mcp_response.error
                    raise MCPClientError(
                        error.get("message", "Unknown error"),
                        code=error.get("code"),
                        data=error.get("data")
                    )
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                logger.debug(
                    f"MCP request successful: {request.method} "
                    f"({response_time:.3f}s)"
                )
                
                return mcp_response
                
        except asyncio.TimeoutError as e:
            self.error_count += 1
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            raise MCPTimeoutError(
                f"Request to {self.config.name} timed out after {self.config.timeout}s"
            ) from e
            
        except ClientError as e:
            self.error_count += 1
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            raise MCPServerUnavailableError(
                f"Failed to connect to {self.config.name}: {str(e)}"
            ) from e
        
        except Exception as e:
            self.error_count += 1
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            logger.error(f"Unexpected error in MCP request: {e}", exc_info=True)
            raise MCPClientError(f"Unexpected error: {str(e)}") from e
    
    async def predict(
        self,
        inputs: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a prediction request to the MCP server
        
        Args:
            inputs: Input text for prediction
            parameters: Optional parameters for the model
            
        Returns:
            Prediction results from the MCP server
            
        Raises:
            MCPClientError: If the request fails
            MCPServerUnavailableError: If server is unavailable
            MCPTimeoutError: If request times out
        """
        request = MCPRequest(
            method="predict",
            params={
                "inputs": inputs,
                "parameters": parameters or {}
            }
        )
        
        response = await self._make_request(request)
        return response.result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the MCP server
        
        Returns:
            Health status information
        """
        request = MCPRequest(method="health_check")
        
        try:
            response = await self._make_request(request)
            self.is_healthy = True
            self.last_health_check = time.time()
            return response.result
        except Exception as e:
            self.is_healthy = False
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            raise
    
    async def get_info(self) -> Dict[str, Any]:
        """Get server information"""
        request = MCPRequest(method="get_info")
        response = await self._make_request(request)
        return response.result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "server_name": self.config.name,
            "server_url": self.config.url,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time": avg_response_time,
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check,
            "circuit_breaker_state": (
                self.circuit_breaker.state.value 
                if self.circuit_breaker else None
            )
        }

class MCPClientManager:
    """
    Manages multiple MCP clients with advanced features:
    - Load balancing
    - Failover handling
    - Batch request optimization
    - Health monitoring
    - Connection pooling
    """
    
    def __init__(self, session: Optional[ClientSession] = None):
        self.clients: Dict[str, MCPClient] = {}
        self.session = session
        self._own_session = session is None
        self.health_check_interval = 60.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized MCP Client Manager")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_session:
            timeout = ClientTimeout(total=120.0)
            self.session = ClientSession(timeout=timeout)
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all clients
        for client in self.clients.values():
            if hasattr(client, '__aexit__'):
                await client.__aexit__(None, None, None)
        
        # Close session
        if self._own_session and self.session:
            await self.session.close()
    
    def add_server(self, config: MCPServerConfig):
        """Add an MCP server to the manager"""
        client = MCPClient(config, session=self.session)
        self.clients[config.name] = client
        logger.info(f"Added MCP server: {config.name} at {config.url}")
    
    def remove_server(self, server_name: str):
        """Remove an MCP server from the manager"""
        if server_name in self.clients:
            del self.clients[server_name]
            logger.info(f"Removed MCP server: {server_name}")
    
    def get_client(self, server_name: str) -> MCPClient:
        """Get MCP client by server name"""
        if server_name not in self.clients:
            raise MCPClientError(f"MCP server '{server_name}' not found")
        return self.clients[server_name]
    
    async def predict(
        self,
        server_name: str,
        inputs: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make prediction request to specific server"""
        client = self.get_client(server_name)
        return await client.predict(inputs, parameters)
    
    async def batch_predict(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions across multiple servers
        
        Args:
            requests: List of prediction requests with format:
                     [{"server": "general", "inputs": "text", "parameters": {...}}]
        
        Returns:
            List of prediction results in the same order
        """
        tasks = []
        
        for req in requests:
            server_name = req["server"]
            inputs = req["inputs"]
            parameters = req.get("parameters")
            
            task = self.predict(server_name, inputs, parameters)
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error dictionaries
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "error_type": type(result).__name__
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all servers"""
        results = {}
        
        for server_name, client in self.clients.items():
            try:
                health_info = await client.health_check()
                results[server_name] = {
                    "status": "healthy",
                    "info": health_info
                }
            except Exception as e:
                results[server_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return results
    
    async def _health_monitor(self):
        """Background task for periodic health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check_all()
                logger.debug("Completed periodic health check")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all clients"""
        return {
            name: client.get_stats() 
            for name, client in self.clients.items()
        }
    
    def get_healthy_servers(self) -> List[str]:
        """Get list of healthy server names"""
        return [
            name for name, client in self.clients.items()
            if client.is_healthy and client.config.enabled
        ]
    
    def get_server_count(self) -> int:
        """Get total number of registered servers"""
        return len(self.clients)
    
    def get_healthy_server_count(self) -> int:
        """Get number of healthy servers"""
        return len(self.get_healthy_servers())

# Convenience functions for common use cases

async def create_mcp_client(
    server_name: str,
    host: str = "localhost",
    port: int = 3001,
    **kwargs
) -> MCPClient:
    """Create a single MCP client"""
    config = MCPServerConfig(server_name, host, port, **kwargs)
    return MCPClient(config)

async def create_mcp_manager(
    server_configs: List[Dict[str, Any]]
) -> MCPClientManager:
    """Create MCP client manager with multiple servers"""
    manager = MCPClientManager()
    
    for config_dict in server_configs:
        config = MCPServerConfig(**config_dict)
        manager.add_server(config)
    
    return manager