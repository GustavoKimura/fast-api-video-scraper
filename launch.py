import subprocess, sys, webbrowser, time, os


def check_docker():
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL, check=True)
        subprocess.run(
            ["docker", "compose", "version"], stdout=subprocess.DEVNULL, check=True
        )
    except subprocess.CalledProcessError:
        print("[✘] Docker or Compose not installed.")
        sys.exit(1)


def run_compose(project_dir, name=""):
    print(f"[*] Launching services in {project_dir}...")
    try:
        subprocess.run(
            ["docker", "compose", "up", "--build", "-d"],
            cwd=project_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[✓] {name or project_dir} started.")
    except subprocess.CalledProcessError as e:
        print(f"[✘] Failed to start {name or project_dir}: {e}")
        sys.exit(1)


def wait_for_backend(host="http://localhost:8000", timeout=30):
    import requests

    print("[*] Waiting for backend to be available...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(host, timeout=2)
            if r.status_code == 200:
                print(" ready!")
                return
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print("\n[✘] Backend not responding in time.")
    sys.exit(1)


def open_browser():
    print("[*] Opening UI in browser...")
    webbrowser.open("http://localhost:8000")


def main():
    check_docker()
    run_compose("searxng", name="SearXNG")
    run_compose(".", name="Main Search Engine")
    wait_for_backend()
    open_browser()
    print("[✓] All systems operational.")


if __name__ == "__main__":
    main()
