"""One command starts the local judge and Vite, then cleans up both on Ctrl-C."""
import signal
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from paperleet.export import export_catalog


def main():
    repository = Path(__file__).resolve().parents[1]
    if not (repository / "web/node_modules/.bin/vite").exists():
        raise SystemExit("Run `cd web && npm ci` first.")
    for port in (8765, 5173):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"Port {port} is already in use. Stop the existing dev process first.")
    export_catalog()
    processes = []
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "paperleet.cli", "serve"], cwd=repository, start_new_session=True))
        processes.append(subprocess.Popen(["npm", "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=repository / "web", start_new_session=True))
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        while all(process.poll() is None for process in processes):
            time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
