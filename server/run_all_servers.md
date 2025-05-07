Open separate terminal instances (like Command Prompt or PowerShell) in the `Redactify` directory and run one command in each:

1.  **Run MCP Classifier (Port 8001):**  
    ```bash
    python server/mcp_classifier/classifier_server.py
    ```

2.  **Run General NER Agent (Port 8002):**  
    ```bash
    python server/a2a_ner_general/general_ner_agent.py
    ```

3.  **Run Medical NER Agent (Port 8003):**  
    ```bash
    python server/a2a_ner_medical/medical_ner_agent.py
    ```

4.  **Run Technical NER Agent (Port 8004):**  
    ```bash
    python server/a2a_ner_technical/technical_ner_agent.py
    ```

5.  **Run PII Specialized NER Agent (Port 8005):**  
    ```bash
    python server/a2a_ner_pii_specialized/pii_specialized_ner_agent.py
    ```

6.  **Run Legal NER Agent (Port 8006):**  
    ```bash
    python server/a2a_ner_legal/legal_ner_agent.py
    ```

7.  **Run Financial NER Agent (Port 8007):**  
    ```bash
    python server/a2a_ner_financial/financial_ner_agent.py
    ```

8.  **Run Dispatcher (Port 8000):**  
    ```bash
    python server/server.py
    ```

Alternatively, you can use the `start` command in a single Command Prompt window (run from `Redactify`) to launch each process in its own new window:

```bash
start "MCP Classifier" python server/mcp_classifier/classifier_server.py
start "General NER" python server/a2a_ner_general/general_ner_agent.py
start "Medical NER" python server/a2a_ner_medical/medical_ner_agent.py
start "Technical NER" python server/a2a_ner_technical/technical_ner_agent.py
start "PII Specialized NER" python server/a2a_ner_pii_specialized/pii_specialized_ner_agent.py
start "Legal NER" python server/a2a_ner_legal/legal_ner_agent.py
start "Financial NER" python server/a2a_ner_financial/financial_ner_agent.py
start "Dispatcher Server" python server/server.py
```

Keep all these terminal windows open while you are using the application.