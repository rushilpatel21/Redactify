import subprocess
import os
import time
import sys
import signal
import requests  # Import requests
from dotenv import load_dotenv  # Import dotenv

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(BASE_DIR, "server")

# Load environment variables from .env file located in the server directory
dotenv_path = os.path.join(SERVER_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Define the services to run: (dir_name, script_name, title, health_check_url_env_var, default_port)
SERVICES = [
    ("mcp_classifier", "classifier_server.py", "MCP Classifier", "MCP_CLASSIFIER_URL", 8001),
    ("a2a_ner_general", "general_ner_agent.py", "General NER", "A2A_GENERAL_URL", 8002),
    ("a2a_ner_medical", "medical_ner_agent.py", "Medical NER", "A2A_MEDICAL_URL", 8003),
    ("a2a_ner_technical", "technical_ner_agent.py", "Technical NER", "A2A_TECHNICAL_URL", 8004),
    ("a2a_ner_pii_specialized", "pii_specialized_ner_agent.py", "PII Specialized NER", "A2A_PII_SPECIALIZED_URL", 8005),
]

DISPATCHER_SCRIPT = "server.py"
DISPATCHER_TITLE = "Dispatcher"
PYTHON_EXE = sys.executable

# Health Check Configuration
HEALTH_CHECK_TIMEOUT = 120  # Max seconds to wait for all agents
HEALTH_CHECK_INTERVAL = 3  # Seconds between checks
HEALTH_CHECK_REQUEST_TIMEOUT = 2  # Seconds timeout for each individual health check request
POST_HEALTH_CHECK_DELAY = 10  # Extra delay after all agents are healthy

# --- Global list to hold process objects ---
processes = []
# --- Dictionary to store agent health check URLs ---
agent_health_urls = {}

# --- Termination Handler ---
def terminate_processes(signum, frame):
    """Signal handler to terminate all child processes."""
    print("\nTermination signal received. Stopping services...")
    # Terminate in reverse order of startup
    for proc, title in reversed(processes):
        try:
            print(f"  > Terminating {title} (PID: {proc.pid})...")
            proc.terminate()
        except Exception as e:
            print(f"  [WARN] Could not terminate {title} (PID: {proc.pid}): {e}")
    print("All termination signals sent.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, terminate_processes)
signal.signal(signal.SIGTERM, terminate_processes)

# --- Health Check Function ---
def check_agent_health(url: str, title: str) -> bool:
    """Checks the /health endpoint of a single agent."""
    health_url = f"{url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=HEALTH_CHECK_REQUEST_TIMEOUT)
        # Check for 2xx status codes for success
        if 200 <= response.status_code < 300:
            print(f"  > {title} ({health_url}) is healthy (Status: {response.status_code}).")
            return True
        else:
            print(f"  > {title} ({health_url}) health check failed (Status: {response.status_code}). Retrying...")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  > {title} ({health_url}) not reachable yet. Retrying...")
        return False
    except requests.exceptions.Timeout:
        print(f"  > {title} ({health_url}) health check timed out. Retrying...")
        return False
    except Exception as e:
        print(f"  > Error checking {title} ({health_url}) health: {e}. Retrying...")
        return False

# --- Launch Services ---
print("Starting Redactify Services...")

creationflags = 0
if sys.platform == "win32":
    creationflags = subprocess.CREATE_NEW_CONSOLE

# Start Agent Services
for service_dir, script_name, title, url_env_var, default_port in SERVICES:
    print(f"Starting {title}...")
    service_path = os.path.join(SERVER_DIR, service_dir)
    script_path = os.path.join(service_path, script_name)
    cmd = [PYTHON_EXE, script_path]

    # Determine agent URL for health check
    agent_base_url = os.environ.get(url_env_var)
    if not agent_base_url:
        agent_base_url = f"http://localhost:{default_port}"
        print(f"  [INFO] Env var {url_env_var} not set, using default URL: {agent_base_url}")
    agent_health_urls[title] = agent_base_url  # Store for health check

    try:
        # Ensure the script path exists before attempting to run
        if not os.path.exists(script_path):
            print(f"  [ERROR] Script not found: {script_path}")
            raise FileNotFoundError(f"Script not found: {script_path}")

        proc = subprocess.Popen(cmd, cwd=service_path, creationflags=creationflags)
        processes.append((proc, title))
        print(f"  > {title} launched (PID: {proc.pid})")
    except FileNotFoundError:
        print(f"  [ERROR] Could not find script or directory for {title}: {script_path}")
        print("  > Attempting to terminate already launched processes...")
        terminate_processes(None, None)
    except Exception as e:
        print(f"  [ERROR] Failed to launch {title}: {e}")
        print("  > Attempting to terminate already launched processes...")
        terminate_processes(None, None)

# --- Wait for Agents with Health Checks ---
print(f"\nWaiting up to {HEALTH_CHECK_TIMEOUT} seconds for agents to become healthy...")
start_wait_time = time.time()
healthy_agents = set()
agents_to_check = set(agent_health_urls.keys())

all_agents_healthy = False
while time.time() - start_wait_time < HEALTH_CHECK_TIMEOUT:
    agents_still_pending = agents_to_check - healthy_agents
    if not agents_still_pending:
        print("\nAll agents reported healthy via /health endpoint.")
        all_agents_healthy = True
        break

    print(f"\nChecking health for pending agents ({len(agents_still_pending)} remaining)...")
    for title in list(agents_still_pending):  # Iterate over a copy
        agent_url = agent_health_urls[title]
        if check_agent_health(agent_url, title):
            healthy_agents.add(title)

    # Wait before next check cycle only if there are still pending agents
    if agents_to_check - healthy_agents:
        print(f"Waiting {HEALTH_CHECK_INTERVAL}s before next health check cycle...")
        time.sleep(HEALTH_CHECK_INTERVAL)

if not all_agents_healthy:
    print(f"\n[ERROR] Timeout: Not all agents became healthy within {HEALTH_CHECK_TIMEOUT} seconds.")
    print(f"  > Missing agents: {', '.join(sorted(list(agents_to_check - healthy_agents)))}")
    print("  > Attempting to terminate launched processes...")
    terminate_processes(None, None)  # Trigger cleanup and exit

# --- Additional Delay ---
print(f"Adding an extra delay of {POST_HEALTH_CHECK_DELAY} seconds before starting dispatcher...")
time.sleep(POST_HEALTH_CHECK_DELAY)

# --- Start Dispatcher Service (only if all agents are healthy) ---
print(f"\nStarting {DISPATCHER_TITLE}...")
dispatcher_script_path = os.path.join(SERVER_DIR, DISPATCHER_SCRIPT)
cmd = [PYTHON_EXE, dispatcher_script_path]
try:
    if not os.path.exists(dispatcher_script_path):
        print(f"  [ERROR] Dispatcher script not found: {dispatcher_script_path}")
        raise FileNotFoundError(f"Dispatcher script not found: {dispatcher_script_path}")

    proc = subprocess.Popen(cmd, cwd=SERVER_DIR, creationflags=creationflags)
    processes.append((proc, DISPATCHER_TITLE))
    print(f"  > {DISPATCHER_TITLE} launched (PID: {proc.pid})")
except FileNotFoundError:
    print(f"  [ERROR] Could not find dispatcher script: {dispatcher_script_path}")
    terminate_processes(None, None)  # Clean up other processes
except Exception as e:
    print(f"  [ERROR] Failed to launch {DISPATCHER_TITLE}: {e}")
    terminate_processes(None, None)  # Clean up other processes

print("\nAll services launched successfully.")
print("Press Ctrl+C in this window to stop all services and exit.")

# Keep the launcher script running, waiting for signals or unexpected termination
try:
    while True:
        for i, (proc, title) in enumerate(processes):
            exit_code = proc.poll()
            if exit_code is not None:
                print(f"\n[WARN] Service '{title}' (PID: {proc.pid}) terminated unexpectedly with code {exit_code}.")
                processes.pop(i)  # Remove from list to avoid re-checking
                break  # Restart loop check since list modified
        time.sleep(5)  # Check every 5 seconds
except Exception as e:
    print(f"\nUnexpected error in launcher's main loop: {e}")
    terminate_processes(None, None)