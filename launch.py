import subprocess
import sys
import time
import webbrowser
import requests
from qdrant_client import QdrantClient

# === ⚙️ SETTINGS ===
DEFAULT_RETRIES = 20
DEFAULT_DELAY = 2


# === 🧾 LOGGING ===
def log(status: str, message: str, end="\n"):
    icons = {
        "info": "ℹ️ ",
        "success": "✅",
        "error": "❌",
        "action": "🔧",
        "waiting": "⏳",
        "build": "🚀",
    }
    print(f"\r{icons.get(status, '❔')} {message}", end=end, flush=True)


# === 🐳 DOCKER ===
def check_docker():
    log("action", "Checking Docker & Compose...", end="")
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(
            ["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL
        )
        log("success", "Docker and Compose available.")
    except subprocess.CalledProcessError:
        log("error", "Docker or Docker Compose not found.")
        sys.exit(1)


def run_compose(path: str, name: str):
    base_msg = f"{name} → launching"
    log("build", base_msg, end="")

    try:
        process = subprocess.Popen(
            ["docker", "compose", "up", "--build", "-d"],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        dots = ""
        while process.poll() is None:
            dots += "."
            print(f"\r🚀 {base_msg}{dots}", end="", flush=True)
            time.sleep(1)

        if process.returncode == 0:
            print("\r" + " " * (len(base_msg) + len(dots) + 3), end="\r")
            log("success", f"{name} is up and running.")
        else:
            log("error", f"{name} failed to launch (exit {process.returncode}).")
            sys.exit(1)
    except Exception as e:
        log("error", f"Exception during {name} launch: {e}")
        sys.exit(1)


# === ⏳ WAITERS ===
def wait_for_service(
    host: str, port: int, retries=DEFAULT_RETRIES, delay=DEFAULT_DELAY
):
    url = f"http://{host}:{port}"
    msg = f"Waiting for {url}"
    log("waiting", msg, end="")

    dots = ""
    for _ in range(retries):
        try:
            if requests.get(url, timeout=2).status_code < 500:
                print("\r" + " " * (len(msg) + len(dots) + 4), end="\r")
                log("success", f"{url} is ready.")
                return
        except:
            pass

        dots += "."
        print(f"\r⏳ {msg}{dots}", end="", flush=True)
        time.sleep(delay)

    log("error", f"Timeout waiting for {url}")
    sys.exit(1)


def wait_for_qdrant(retries=DEFAULT_RETRIES, delay=DEFAULT_DELAY):
    msg = "Waiting for Qdrant (port 6333)"
    log("waiting", msg, end="")

    dots = ""
    for _ in range(retries):
        try:
            QdrantClient(host="localhost", port=6333).get_collections()
            print("\r" + " " * (len(msg) + len(dots) + 4), end="\r")
            log("success", "Qdrant is ready.")
            return
        except:
            dots += "."
            print(f"\r⏳ {msg}{dots}", end="", flush=True)
            time.sleep(delay)

    log("error", "Qdrant did not become ready.")
    sys.exit(1)


# === 🌐 BROWSER ===
def open_browser(url="http://localhost:8000"):
    log("action", f"Opening browser at {url}")
    webbrowser.open(url)


# === 🚀 MAIN ===
def main():
    log("info", "=== 🚀 FastAPI Scraper Bootstrap ===")
    check_docker()
    run_compose("searxng", "SearXNG")
    run_compose(".", "Search Engine Backend")
    wait_for_qdrant()
    wait_for_service("localhost", 8000)
    open_browser()
    log("success", "🎉 All systems operational!")


if __name__ == "__main__":
    main()
