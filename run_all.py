import subprocess
import os
import time
import sys

# --- Configuration ---
# Base directory where the 'server' folder is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(BASE_DIR, "server")

# Define the services to run: (directory_name, script_name, window_title)
SERVICES = [
    ("mcp_classifier", "classifier_server.py", "MCP Classifier"),
    ("a2a_ner_general", "general_ner_agent.py", "General NER"),
    ("a2a_ner_medical", "medical_ner_agent.py", "Medical NER"),
    ("a2a_ner_technical", "technical_ner_agent.py", "Technical NER"),
    ("a2a_ner_pii_specialized", "pii_specialized_ner_agent.py", "PII Specialized NER"),
]

DISPATCHER_SCRIPT = "server.py"
DISPATCHER_TITLE = "Dispatcher"
PYTHON_EXE = sys.executable # Use the same python interpreter that runs this script

# Delay before starting the dispatcher (seconds)
STARTUP_DELAY = 5

# --- Launch Services ---
print("Starting Redactify Services...")
processes = []

# Determine creation flags based on OS (for new console window on Windows)
creationflags = 0
if sys.platform == "win32":
    creationflags = subprocess.CREATE_NEW_CONSOLE

# Start Agent Services
for service_dir, script_name, title in SERVICES:
    print(f"Starting {title}...")
    service_path = os.path.join(SERVER_DIR, service_dir)
    script_path = os.path.join(service_path, script_name)
    cmd = [PYTHON_EXE, script_path]

    try:
        # Use Popen to start the process without waiting
        # cwd sets the working directory for the subprocess
        proc = subprocess.Popen(cmd, cwd=service_path, creationflags=creationflags)
        processes.append(proc)
        print(f"  > {title} launched (PID: {proc.pid})")
    except FileNotFoundError:
        print(f"  [ERROR] Could not find script or directory for {title}: {script_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to launch {title}: {e}")

# Wait for agents to initialize
print(f"\nWaiting {STARTUP_DELAY} seconds for agents to initialize...")
time.sleep(STARTUP_DELAY)

# Start Dispatcher Service
print(f"Starting {DISPATCHER_TITLE}...")
dispatcher_script_path = os.path.join(SERVER_DIR, DISPATCHER_SCRIPT)
cmd = [PYTHON_EXE, dispatcher_script_path]
try:
    proc = subprocess.Popen(cmd, cwd=SERVER_DIR, creationflags=creationflags)
    processes.append(proc)
    print(f"  > {DISPATCHER_TITLE} launched (PID: {proc.pid})")
except FileNotFoundError:
    print(f"  [ERROR] Could not find dispatcher script: {dispatcher_script_path}")
except Exception as e:
    print(f"  [ERROR] Failed to launch {DISPATCHER_TITLE}: {e}")


print("\nAll services launched in separate windows.")
print("Press Ctrl+C in this window to exit this launcher script (services will keep running).")
print("To stop the services, close their individual console windows.")

# Keep the launcher script running until interrupted,
# otherwise it might exit immediately.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nLauncher script interrupted. Remember to close service windows manually.")
    # Note: This does NOT automatically terminate the child processes started with Popen.
    # You would need more complex logic (e.g., storing PIDs and terminating them) for that.