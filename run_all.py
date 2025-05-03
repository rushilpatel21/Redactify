import subprocess
import os
import time
import sys
import signal # Import signal module

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
STARTUP_DELAY = 20

# --- Global list to hold process objects ---
processes = []

# --- Termination Handler ---
def terminate_processes(signum, frame):
    """Signal handler to terminate all child processes."""
    print("\nTermination signal received. Stopping services...")
    # Iterate in reverse order in case of dependencies (though unlikely here)
    for proc, title in reversed(processes):
        try:
            print(f"  > Terminating {title} (PID: {proc.pid})...")
            proc.terminate() # Send SIGTERM (or TerminateProcess on Windows)
        except Exception as e:
            print(f"  [WARN] Could not terminate {title} (PID: {proc.pid}): {e}")
    print("All termination signals sent.")
    sys.exit(0) # Exit the launcher script

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, terminate_processes)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, terminate_processes) # Handle termination signals (e.g., from OS)

# --- Launch Services ---
print("Starting Redactify Services...")

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
        proc = subprocess.Popen(cmd, cwd=service_path, creationflags=creationflags)
        processes.append((proc, title)) # Store tuple of (process_object, title)
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
    processes.append((proc, DISPATCHER_TITLE))
    print(f"  > {DISPATCHER_TITLE} launched (PID: {proc.pid})")
except FileNotFoundError:
    print(f"  [ERROR] Could not find dispatcher script: {dispatcher_script_path}")
except Exception as e:
    print(f"  [ERROR] Failed to launch {DISPATCHER_TITLE}: {e}")

print("\nAll services launched.")
print("Press Ctrl+C in this window to stop all services and exit.")

# Keep the launcher script running, waiting for signals
try:
    while True:
        # Check if any process exited unexpectedly (optional)
        for i, (proc, title) in enumerate(processes):
            if proc.poll() is not None: # poll() returns exit code if terminated, None otherwise
                print(f"\n[WARN] Service '{title}' (PID: {proc.pid}) terminated unexpectedly with code {proc.returncode}.")
        time.sleep(5) # Check every 5 seconds
except Exception as e:
    print(f"\nUnexpected error in launcher's main loop: {e}")
    terminate_processes(None, None)