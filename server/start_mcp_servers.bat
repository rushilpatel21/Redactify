@echo off
echo === Starting Redactify MCP Servers ===

REM Set environment variables for ports
set A2A_GENERAL_PORT=3001
set A2A_MEDICAL_PORT=3002
set A2A_TECHNICAL_PORT=3003
set A2A_LEGAL_PORT=3004
set A2A_FINANCIAL_PORT=3005
set A2A_PII_SPECIALIZED_PORT=3006

echo Starting General NER on port %A2A_GENERAL_PORT%...
start "General NER" python a2a_ner_general/general_ner_agent.py

timeout /t 2 /nobreak >nul

echo Starting Medical NER on port %A2A_MEDICAL_PORT%...
start "Medical NER" python a2a_ner_medical/medical_ner_agent.py

timeout /t 2 /nobreak >nul

echo Starting Technical NER on port %A2A_TECHNICAL_PORT%...
start "Technical NER" python a2a_ner_technical/technical_ner_agent.py

timeout /t 2 /nobreak >nul

echo Starting Legal NER on port %A2A_LEGAL_PORT%...
start "Legal NER" python a2a_ner_legal/legal_ner_agent.py

timeout /t 2 /nobreak >nul

echo Starting Financial NER on port %A2A_FINANCIAL_PORT%...
start "Financial NER" python a2a_ner_financial/financial_ner_agent.py

timeout /t 2 /nobreak >nul

echo Starting PII Specialized on port %A2A_PII_SPECIALIZED_PORT%...
start "PII Specialized" python a2a_ner_pii_specialized/pii_specialized_ner_agent.py

echo.
echo === All MCP Servers Started ===
echo Check the individual windows for each server's status
echo Press any key to exit...
pause >nul