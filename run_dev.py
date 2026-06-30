import os
import sys
import subprocess
import time
import signal

def run_dev():
    project_root = os.path.abspath(os.path.dirname(__file__))
    backend_dir = os.path.join(project_root, "backend")
    frontend_dir = os.path.join(project_root, "frontend")

    processes = []

    # Function to handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nStopping all services...")
        for p in processes:
            try:
                if os.name == 'nt':
                    # Windows taskkill command for clean tree termination
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
                else:
                    p.terminate()
            except Exception as e:
                print(f"Error terminating process {p.pid}: {e}")
        print("Services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("      STARTING ISRO CLEAN AIR INTELLIGENCE PLATFORM")
    print("=" * 60)
    print("Press Ctrl+C to terminate both servers.")
    print("=" * 60)

    # 1. Start FastAPI Backend
    print("\n[START] Launching FastAPI Backend on http://localhost:8000...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    try:
        p_back = subprocess.Popen(
            backend_cmd,
            cwd=backend_dir,
            stdout=None, # Keep output visible in terminal
            stderr=None
        )
        processes.append(p_back)
    except Exception as e:
        print(f"[ERROR] Failed to start FastAPI backend: {e}")
        sys.exit(1)

    time.sleep(2) # Wait for backend port binding

    # 2. Start Vite React Frontend
    print("\n[START] Launching React (Vite) Frontend on http://localhost:5173...")
    
    # Use npm.cmd on Windows, npm on Unix
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    
    try:
        p_front = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_dir,
            stdout=None,
            stderr=None
        )
        processes.append(p_front)
    except Exception as e:
        print(f"[ERROR] Failed to start React frontend: {e}")
        # Clean backend first
        p_back.terminate()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("      SERVICES RUNNING CONCURRENTLY")
    print("=" * 60)
    print("FastAPI Backend : http://localhost:8000")
    print("React Frontend  : http://localhost:5173")
    print("=" * 60)
    
    # Keep the main process alive
    try:
        while True:
            # Check if any process terminated unexpectedly
            for p in processes:
                if p.poll() is not None:
                    print(f"[WARNING] Process {p.pid} terminated with code {p.returncode}")
                    signal_handler(None, None)
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    run_dev()
