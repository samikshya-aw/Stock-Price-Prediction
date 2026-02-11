import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.request


def build_uvicorn_cmd(host, port, reload_enabled):
    cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", host, "--port", str(port)]
    if reload_enabled:
        cmd.append("--reload")
    return cmd


def build_streamlit_cmd():
    return [sys.executable, "-m", "streamlit", "run", "app.py"]


def relaunch_in_venv(root_path):
    venv_python = os.path.join(root_path, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        return False

    if os.path.normcase(sys.executable) == os.path.normcase(venv_python):
        return False

    print("Detected .venv. Relaunching with the project virtual environment...")
    args = [venv_python, os.path.join(root_path, "run.py")] + sys.argv[1:]
    raise SystemExit(subprocess.call(args))


def require_modules(modules):
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return True

    print("Missing modules: " + ", ".join(missing))
    print("Install with: pip install " + " ".join(missing))
    return False


def is_port_in_use(host, port):
    check_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((check_host, port)) == 0


def find_available_port(host, start_port, max_tries=10):
    for offset in range(max_tries):
        port = start_port + offset
        if not is_port_in_use(host, port):
            return port
    return start_port


def wait_for_api(api_url, timeout_seconds=20):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(api_url, timeout=1) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    relaunch_in_venv(root)

    if not require_modules(["uvicorn", "streamlit"]):
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="Run FastAPI and Streamlit together.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    args = parser.parse_args()

    port = find_available_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is in use. Using {port} instead.")

    api_url = f"http://{args.host}:{port}"
    api_cmd = build_uvicorn_cmd(args.host, port, args.reload)
    ui_cmd = build_streamlit_cmd()

    env = os.environ.copy()
    env["API_URL"] = api_url

    api_url_path = os.path.join(root, ".api_url")
    with open(api_url_path, "w", encoding="utf-8") as handle:
        handle.write(api_url)

    print(f"API URL: {api_url}")

    api_proc = subprocess.Popen(api_cmd)

    if not wait_for_api(api_url):
        print("API did not become ready in time. Streamlit will still start.")

    ui_proc = subprocess.Popen(ui_cmd, env=env)

    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (api_proc, ui_proc):
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
