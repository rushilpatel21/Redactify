# Redactify MCP Server Startup Guide

This guide will help you start all the MCP (Model Context Protocol) servers for Redactify.

## Quick Start

### Step 1: Start MCP Servers

Choose one of the following methods to start all MCP servers:

#### Option A: PowerShell Script (Recommended for Windows)
```powershell
.\start_mcp_servers.ps1
```

#### Option B: Python Script (Cross-platform)
```bash
python start_mcp_servers.py
```

#### Option C: Batch File (Windows CMD)
```cmd
start_mcp_servers.bat
```

### Step 2: Verify Servers Are Running

The following servers should be running on these ports:

| Server | Port | Health Check URL |
|--------|------|------------------|
| General NER | 3001 | http://localhost:3001/health |
| Medical NER | 3002 | http://localhost:3002/health |
| Technical NER | 3003 | http://localhost:3003/health |
| Legal NER | 3004 | http://localhost:3004/health |
| Financial NER | 3005 | http://localhost:3005/health |
| PII Specialized | 3006 | http://localhost:3006/health |

You can test each server by visiting its health check URL in your browser or using curl:

```bash
curl http://localhost:3001/health
```

### Step 3: Start Main Server

Once all MCP servers are running, start the main Redactify server:

```bash
python server.py
```

The main server will be available at: http://localhost:8000

## Manual Startup (For Debugging)

If you need to start servers individually for debugging:

```bash
# Terminal 1 - General NER
$env:A2A_GENERAL_PORT=3001; python a2a_ner_general/general_ner_agent.py

# Terminal 2 - Medical NER  
$env:A2A_MEDICAL_PORT=3002; python a2a_ner_medical/medical_ner_agent.py

# Terminal 3 - Technical NER
$env:A2A_TECHNICAL_PORT=3003; python a2a_ner_technical/technical_ner_agent.py

# Terminal 4 - Legal NER
$env:A2A_LEGAL_PORT=3004; python a2a_ner_legal/legal_ner_agent.py

# Terminal 5 - Financial NER
$env:A2A_FINANCIAL_PORT=3005; python a2a_ner_financial/financial_ner_agent.py

# Terminal 6 - PII Specialized
$env:A2A_PII_SPECIALIZED_PORT=3006; python a2a_ner_pii_specialized/pii_specialized_ner_agent.py
```

## Troubleshooting

### Port Already in Use
If you get "port already in use" errors, check what's running on the ports:

```powershell
# Windows
netstat -ano | findstr :3001

# Kill process by PID
taskkill /PID <PID> /F
```

### Model Loading Issues
- Each server needs to download and load ML models on first run
- This can take 5-10 minutes per server initially
- Subsequent starts will be faster as models are cached

### Memory Issues
- Each server loads a separate ML model (~1-2GB RAM each)
- You may need 8-12GB RAM to run all servers simultaneously
- Consider running only the servers you need for testing

### Connection Refused
- Make sure all MCP servers are fully started before starting the main server
- Check the server logs for any error messages
- Verify the correct ports are being used

## Testing

To test if everything is working:

```bash
# Test MCP servers startup
python test_mcp_startup.py

# Test complete system
python test_complete_system.py
```

## Performance Tips

1. **Start servers in order**: General → Medical → Technical → Legal → Financial → PII
2. **Wait between starts**: Give each server 30-60 seconds to fully load
3. **Monitor memory usage**: Each server uses 1-2GB RAM
4. **Use SSD storage**: Faster model loading from SSD vs HDD

## Production Deployment

For production, consider:
- Running servers on separate machines
- Using process managers like PM2 or systemd
- Setting up load balancers
- Implementing health monitoring
- Using container orchestration (Docker/Kubernetes)