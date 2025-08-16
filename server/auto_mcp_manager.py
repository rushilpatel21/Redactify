import asyncio
import subprocess
import sys
import os
import logging

logger = logging.getLogger("AutoMCPManager")

class AutoMCPManager:
    """
    Manages automatic startup, health-check, monitoring, and shutdown of MCP agent microservices.
    """
    def __init__(self):
        self.procs = {}
        # Define agent scripts and their default ports
        self.agent_scripts = {
            "general": ("a2a_ner_general/general_ner_agent.py", 3001),
            "medical": ("a2a_ner_medical/medical_ner_agent.py", 3002),
            "technical": ("a2a_ner_technical/technical_ner_agent.py", 3003),
            "legal": ("a2a_ner_legal/legal_ner_agent.py", 3004),
            "financial": ("a2a_ner_financial/financial_ner_agent.py", 3005),
            "pii_specialized": ("a2a_ner_pii_specialized/pii_specialized_ner_agent.py", 3006),
            "classifier": ("mcp_classifier/classifier_server.py", 3007),
        }

    async def start_all_servers(self, timeout: float = 300.0) -> bool:
        script_dir = os.path.dirname(__file__)
        for name, (rel_path, port) in self.agent_scripts.items():
            script_path = os.path.join(script_dir, rel_path)
            try:
                proc = subprocess.Popen([sys.executable, script_path], cwd=script_dir)
                self.procs[name] = proc
                logger.info(f"Started MCP agent '{name}' on port {port} (pid={proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start MCP agent '{name}': {e}")
        # Give processes time to initialize
        await asyncio.sleep(5)
        # Return True if at least one process is running
        return any(p.poll() is None for p in self.procs.values())

    def get_server_status(self) -> dict:
        status = {}
        for name, (rel_path, port) in self.agent_scripts.items():
            proc = self.procs.get(name)
            status[name] = {
                "running": proc.poll() is None if proc else False,
                "port": port
            }
        return status

    async def shutdown_all_servers(self):
        for name, proc in list(self.procs.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
                logger.info(f"Terminated MCP agent '{name}' (pid={proc.pid})")
            except Exception:
                proc.kill()
            self.procs.pop(name, None)

    async def check_all_health(self) -> dict:
        health = {}
        for name, proc in self.procs.items():
            health[name] = (proc.poll() is None)
        return health

    def start_monitoring(self):
        # Optionally implement continuous health monitoring here
        # For now, no-op
        async def noop():
            while True:
                await asyncio.sleep(60)
        return noop()


def get_auto_mcp_manager() -> AutoMCPManager:
    return AutoMCPManager()
