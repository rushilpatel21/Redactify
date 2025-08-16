# MCP (Model Context Protocol) Implementation in Redactify

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           REDACTIFY MCP ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────────┐
                                    │   React Client  │
                                    │  (Port 5173)    │
                                    └─────────┬───────┘
                                              │ HTTP REST API
                                              │ POST /anonymize
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │            MAIN FASTAPI SERVER                      │
                    │              (Port 8000)                            │
                    │                                                     │
                    │  ┌─────────────────┐    ┌─────────────────────────┐ │
                    │  │  REST API       │    │   MCP CLIENT MANAGER    │ │
                    │  │  Gateway        │    │                         │ │
                    │  │                 │    │  ┌─────────────────────┐│ │
                    │  │ /anonymize      │◄──►│  │   MCPClient Pool    ││ │
                    │  │ /health         │    │  │                     ││ │
                    │  │ /mcp-status     │    │  │  - Connection Mgmt  ││ │
                    │  └─────────────────┘    │  │  - Load Balancing   ││ │
                    │                         │  │  - Health Checks    ││ │
                    │                         │  └─────────────────────┘│ │
                    │                         └─────────────────────────┘ │
                    └─────────────────────────┬───────────────────────────┘
                                              │ JSON-RPC 2.0 over HTTP
                                              │ MCP Protocol
                                              ▼
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                        MCP MICROSERVICES LAYER                          │
        └─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  General NER    │  │  Medical NER    │  │ Technical NER   │  │  Legal NER      │
│  MCP Server     │  │  MCP Server     │  │  MCP Server     │  │  MCP Server     │
│  (Port 3001)    │  │  (Port 3002)    │  │  (Port 3003)    │  │  (Port 3004)    │
│                 │  │                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │FastMCP      │ │  │ │FastMCP      │ │  │ │FastMCP      │ │  │ │FastMCP      │ │
│ │Framework    │ │  │ │Framework    │ │  │ │Framework    │ │  │ │Framework    │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│                 │  │                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │BERT-CoNLL   │ │  │ │RoBERTa-i2b2 │ │  │ │Tech-Domain  │ │  │ │Legal-Domain │ │
│ │NER Model    │ │  │ │Medical Model│ │  │ │NER Model    │ │  │ │NER Model    │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐
│ Financial NER   │  │PII Specialized  │
│ MCP Server      │  │MCP Server       │
│ (Port 3005)     │  │ (Port 3006)     │
│                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │FastMCP      │ │  │ │FastMCP      │ │
│ │Framework    │ │  │ │Framework    │ │
│ └─────────────┘ │  │ └─────────────┘ │
│                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │Finance-     │ │  │ │PII-Focused  │ │
│ │Domain Model │ │  │ │NER Model    │ │
│ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘

                    ┌─────────────────────────────────────┐
                    │         MCP PROTOCOL FLOW           │
                    └─────────────────────────────────────┘

Client Request:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ {                                                                               │
│   "jsonrpc": "2.0",                                                             │
│   "method": "predict",                                                          │
│   "params": {                                                                   │
│     "inputs": "John Smith works at Acme Corp",                                  │
│     "parameters": {"confidence_threshold": 0.8}                                 │
│   },                                                                            │
│   "id": "req-123"                                                               │
│ }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘

Server Response:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ {                                                                               │
│   "jsonrpc": "2.0",                                                             │
│   "result": {                                                                   │
│     "entities": [                                                               │
│       {                                                                         │
│         "entity_group": "PERSON",                                               │
│         "score": 0.9998,                                                        │
│         "word": "John Smith",                                                   │
│         "start": 0,                                                             │
│         "end": 10,                                                              │
│         "detector": "a2a_ner_general"                                           │
│       }                                                                         │
│     ]                                                                           │
│   },                                                                            │
│   "id": "req-123"                                                               │
│ }                                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## What is MCP (Model Context Protocol)?

**Model Context Protocol (MCP)** is an open standard that enables AI applications to securely connect with external data sources and tools. It provides a standardized way for AI systems to:

- **Access external resources** through a consistent interface
- **Execute tools and functions** in a controlled environment  
- **Maintain security boundaries** between AI and external systems
- **Enable modular architectures** with distributed services

### Key MCP Concepts

1. **JSON-RPC 2.0 Based**: Uses standard request/response messaging
2. **Tool-Oriented**: Exposes functionality as callable tools/functions
3. **Secure by Design**: Controlled access with proper authentication
4. **Language Agnostic**: Works with any programming language
5. **Standardized Protocol**: Consistent API across implementations

## How Redactify Implements MCP

### 1. Distributed Microservices Architecture

Redactify uses MCP to create a **distributed AI system** where:

- **Main Server**: Acts as MCP client and REST API gateway
- **Specialized Servers**: Each domain-specific NER model runs as an independent MCP server
- **Protocol Communication**: All inter-service communication uses MCP JSON-RPC

### 2. MCP Server Implementation

Each specialized NER service is a **true MCP server**:

```python
# server/a2a_ner_general/general_ner_agent.py
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    name="GeneralNERAgent", 
    version="1.0.0",
    description="General-purpose Named Entity Recognition agent"
)

# Expose tools via MCP decorators
@mcp.tool()
async def predict(inputs: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect named entities in text using BERT-based NER model."""
    # Load and run NER pipeline
    results = ner_pipeline(inputs)
    
    # Process and return entities
    return {
        "entities": [
            {
                "entity_group": entity["entity_group"],
                "score": float(entity["score"]),
                "word": entity["word"],
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "detector": "a2a_ner_general"
            }
            for entity in results
        ]
    }

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check the health of the NER agent service."""
    return {
        "status": "ok",
        "agent_id": "a2a_ner_general",
        "model_loaded": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
```

### 3. JSON-RPC 2.0 Endpoint

Each MCP server exposes a standard JSON-RPC endpoint:

```python
@app.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    """MCP JSON-RPC endpoint that processes requests and forwards to appropriate tools"""
    data = await request.json()
    
    # Validate JSON-RPC 2.0 format
    if "jsonrpc" in data and "method" in data:
        method = data["method"]
        params = data.get("params", {})
        json_rpc_id = data.get("id")
        
        # Route to appropriate tool
        if method == "predict":
            result = await predict(inputs=params["inputs"], parameters=params.get("parameters"))
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "result": result,
                "id": json_rpc_id
            }), media_type="application/json")
        
        elif method == "health_check":
            result = await health_check()
            return Response(content=json.dumps({
                "jsonrpc": "2.0",
                "result": result,
                "id": json_rpc_id
            }), media_type="application/json")
```

### 4. MCP Client Implementation

The main server acts as an **MCP client** to communicate with specialized servers:

```python
# server/server_app/mcp_client.py
class MCPClient:
    """Client for communicating with MCP servers via JSON-RPC 2.0"""
    
    async def predict(self, inputs: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a prediction request to the MCP server"""
        request = MCPRequest(
            method="predict",
            params={
                "inputs": inputs,
                "parameters": parameters or {}
            }
        )
        
        response = await self._make_request(request)
        return response.result
    
    async def _make_request(self, request: MCPRequest) -> MCPResponse:
        """Send JSON-RPC request to MCP server"""
        payload = {
            "jsonrpc": "2.0",
            "method": request.method,
            "params": request.params,
            "id": str(uuid.uuid4())
        }
        
        async with self.session.post(
            f"{self.config.url}/mcp",
            json=payload,
            timeout=self.config.timeout
        ) as response:
            data = await response.json()
            return MCPResponse.from_dict(data)
```

## MCP Server Specifications

### Server Configurations

| Server | Port | Model | Domain | Purpose |
|--------|------|-------|---------|---------|
| General NER | 3001 | `dbmdz/bert-large-cased-finetuned-conll03-english` | General | Standard NER (PERSON, ORG, LOC) |
| Medical NER | 3002 | `obi/deid_roberta_i2b2` | Healthcare | Medical entities, PHI |
| Technical NER | 3003 | Custom tech model | Technology | API keys, URLs, IPs |
| Legal NER | 3004 | Custom legal model | Legal | Legal entities, case numbers |
| Financial NER | 3005 | Custom finance model | Finance | Account numbers, routing |
| PII Specialized | 3006 | Custom PII model | Privacy | Specialized PII detection |

### MCP Tools Available

Each MCP server exposes these standardized tools:

#### 1. `predict` Tool
```json
{
  "name": "predict",
  "description": "Detect named entities in text",
  "parameters": {
    "inputs": {
      "type": "string",
      "description": "Input text to analyze"
    },
    "parameters": {
      "type": "object",
      "description": "Optional parameters for model inference",
      "properties": {
        "confidence_threshold": {"type": "number", "default": 0.5},
        "max_entities": {"type": "integer", "default": 100}
      }
    }
  }
}
```

#### 2. `health_check` Tool
```json
{
  "name": "health_check", 
  "description": "Check the health status of the MCP server",
  "parameters": {}
}
```

## MCP Protocol Flow

### 1. Request Processing Flow

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client    │    │   Main Server   │    │   MCP Server    │
│             │    │  (MCP Client)   │    │                 │
└──────┬──────┘    └─────────┬───────┘    └─────────┬───────┘
       │                     │                      │
       │ POST /anonymize     │                      │
       ├────────────────────►│                      │
       │                     │                      │
       │                     │ JSON-RPC predict     │
       │                     ├─────────────────────►│
       │                     │                      │
       │                     │                      │ ┌─────────────┐
       │                     │                      │ │   Run NER   │
       │                     │                      │ │   Pipeline  │
       │                     │                      │ └─────────────┘
       │                     │                      │
       │                     │ JSON-RPC response    │
       │                     │◄─────────────────────┤
       │                     │                      │
       │ Aggregated results  │                      │
       │◄────────────────────┤                      │
       │                     │                      │
```

### 2. Parallel Processing

The main server sends requests to **multiple MCP servers simultaneously**:

```python
# Concurrent requests to all MCP servers
async def detect_entities_parallel(text: str) -> List[Dict]:
    tasks = []
    
    # Create tasks for each MCP server
    for server_name, client in mcp_client_manager.clients.items():
        if client.config.enabled:
            task = client.predict(inputs=text)
            tasks.append((server_name, task))
    
    # Execute all requests concurrently
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    # Aggregate results from all servers
    all_entities = []
    for (server_name, _), result in zip(tasks, results):
        if isinstance(result, dict) and "entities" in result:
            all_entities.extend(result["entities"])
    
    return all_entities
```

## Environment Configuration

### MCP Server Ports
```bash
# MCP Server Ports (auto-configured)
A2A_GENERAL_PORT=3001
A2A_MEDICAL_PORT=3002  
A2A_TECHNICAL_PORT=3003
A2A_LEGAL_PORT=3004
A2A_FINANCIAL_PORT=3005
A2A_PII_SPECIALIZED_PORT=3006
```

### MCP Client Configuration
```bash
# MCP Configuration
MAX_WORKERS=8
MAX_MODEL_MEMORY_MB=4096
CONFIDENCE_THRESHOLD=0.5

# Connection settings
MCP_TIMEOUT=120.0
MCP_MAX_RETRIES=3
MCP_RETRY_DELAY=1.0
```

## Running the MCP System

### 1. Start All MCP Servers
```bash
# Main server automatically starts all MCP servers
python server.py

# Or start individually for debugging
python a2a_ner_general/general_ner_agent.py    # Port 3001
python a2a_ner_medical/medical_ner_agent.py    # Port 3002
python a2a_ner_technical/technical_ner_agent.py # Port 3003
# ... etc
```

### 2. Health Check Endpoints

**Individual MCP Server Health:**
```bash
curl http://localhost:3001/health  # General NER
curl http://localhost:3002/health  # Medical NER
# ... etc
```

**Main Server MCP Status:**
```bash
curl http://localhost:8000/mcp-status
```

**Response:**
```json
{
  "mcp_servers": {
    "a2a_ner_general": {
      "status": "healthy",
      "url": "http://localhost:3001",
      "last_check": "2024-01-15T10:30:00Z"
    },
    "a2a_ner_medical": {
      "status": "healthy", 
      "url": "http://localhost:3002",
      "last_check": "2024-01-15T10:30:00Z"
    }
  },
  "total_servers": 6,
  "healthy_servers": 6
}
```

## MCP Protocol Benefits

### 1. **Modularity**
- Each domain model runs as independent service
- Easy to add/remove specialized models
- Clear separation of concerns

### 2. **Scalability** 
- Individual services can be scaled based on demand
- Load balancing across multiple instances
- Resource isolation per service

### 3. **Fault Tolerance**
- If one MCP server fails, others continue working
- Graceful degradation of functionality
- Health monitoring and automatic recovery

### 4. **Standardization**
- All services follow same MCP protocol
- Consistent API across different models
- Easy integration with external tools

### 5. **Security**
- Controlled access through MCP protocol
- Service isolation and sandboxing
- Standardized authentication mechanisms

## Development and Testing

### Testing MCP Servers

```bash
# Test individual MCP server
curl -X POST http://localhost:3001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "predict",
    "params": {
      "inputs": "John Smith works at Acme Corp"
    },
    "id": "test-123"
  }'
```

### MCP Client Testing

```python
# Test MCP client connection
async def test_mcp_client():
    client = MCPClient(MCPServerConfig("test", "localhost", 3001))
    
    result = await client.predict("Test text with John Smith")
    print(f"Entities found: {len(result['entities'])}")
    
    health = await client.health_check()
    print(f"Server status: {health['status']}")
```

## Conclusion

Redactify's MCP implementation demonstrates a **production-ready distributed AI system** that:

- Uses **true MCP servers** for specialized NER models
- Implements **JSON-RPC 2.0 protocol** correctly
- Provides **fault-tolerant microservices architecture**
- Enables **parallel processing** across multiple AI models
- Maintains **standardized interfaces** for easy integration

This architecture allows Redactify to efficiently process PII detection across multiple specialized domains while maintaining the flexibility to add new models and scale individual services as needed.