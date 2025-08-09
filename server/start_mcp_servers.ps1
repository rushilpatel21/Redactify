#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start all MCP servers for Redactify

.DESCRIPTION
    This script starts all the MCP (Model Context Protocol) servers on their designated ports:
    - General NER: Port 3001
    - Medical NER: Port 3002  
    - Technical NER: Port 3003
    - Legal NER: Port 3004
    - Financial NER: Port 3005
    - PII Specialized: Port 3006
#>

Write-Host "=== Starting Redactify MCP Servers ===" -ForegroundColor Green

# Server configurations
$servers = @(
    @{ Name = "General NER"; Script = "a2a_ner_general/general_ner_agent.py"; Port = 3001; EnvVar = "A2A_GENERAL_PORT" },
    @{ Name = "Medical NER"; Script = "a2a_ner_medical/medical_ner_agent.py"; Port = 3002; EnvVar = "A2A_MEDICAL_PORT" },
    @{ Name = "Technical NER"; Script = "a2a_ner_technical/technical_ner_agent.py"; Port = 3003; EnvVar = "A2A_TECHNICAL_PORT" },
    @{ Name = "Legal NER"; Script = "a2a_ner_legal/legal_ner_agent.py"; Port = 3004; EnvVar = "A2A_LEGAL_PORT" },
    @{ Name = "Financial NER"; Script = "a2a_ner_financial/financial_ner_agent.py"; Port = 3005; EnvVar = "A2A_FINANCIAL_PORT" },
    @{ Name = "PII Specialized"; Script = "a2a_ner_pii_specialized/pii_specialized_ner_agent.py"; Port = 3006; EnvVar = "A2A_PII_SPECIALIZED_PORT" }
)

$processes = @()

foreach ($server in $servers) {
    Write-Host "Starting $($server.Name) on port $($server.Port)..." -ForegroundColor Yellow
    
    # Check if script exists
    if (-not (Test-Path $server.Script)) {
        Write-Host "✗ Script not found: $($server.Script)" -ForegroundColor Red
        continue
    }
    
    try {
        # Set environment variable
        Set-Item -Path "env:$($server.EnvVar)" -Value $server.Port.ToString()
        
        # Start the process
        $process = Start-Process -FilePath "python" -ArgumentList $server.Script -PassThru -WindowStyle Normal
        
        if ($process) {
            $processes += @{ Name = $server.Name; Process = $process; Port = $server.Port }
            Write-Host "✓ $($server.Name) started with PID $($process.Id)" -ForegroundColor Green
        }
        else {
            Write-Host "✗ Failed to start $($server.Name)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "✗ Error starting $($server.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Small delay between starts
    Start-Sleep -Seconds 2
}

if ($processes.Count -eq 0) {
    Write-Host "No servers started successfully!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== MCP Servers Running ===" -ForegroundColor Green
foreach ($proc in $processes) {
    Write-Host "  $($proc.Name): PID $($proc.Process.Id) on port $($proc.Port)" -ForegroundColor Cyan
}

Write-Host "`nPress Ctrl+C to stop all servers..." -ForegroundColor Yellow

try {
    # Wait for user interrupt
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if any process has exited
        foreach ($proc in $processes) {
            if ($proc.Process.HasExited) {
                Write-Host "$($proc.Name) has stopped (exit code: $($proc.Process.ExitCode))" -ForegroundColor Red
            }
        }
    }
}
catch {
    Write-Host "`nShutting down MCP servers..." -ForegroundColor Yellow
    
    # Stop all processes
    foreach ($proc in $processes) {
        try {
            if (-not $proc.Process.HasExited) {
                $proc.Process.Kill()
                Write-Host "✓ Stopped $($proc.Name)" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Error stopping $($proc.Name): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host "All MCP servers stopped" -ForegroundColor Green
}