#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from venv import EnvBuilder


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def install_requirements(venv_dir: Path, requirements_file: Path):
    log("Creating .venv ...")
    EnvBuilder(with_pip=True).create(venv_dir)
    log("Created .venv!")

    log("Upgrading pip")
    python_bin = str(venv_dir / "bin" / "python")
    if os.name == "nt":
        python_bin = str(venv_dir / "Scripts" / "python.exe")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    if requirements_file.exists():
        log("Installing requirements")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        log("No requirements.txt found, skipping install")


def read_files_list(files_file: Path):
    if not files_file.exists():
        raise FileNotFoundError(f"Missing file list: {files_file}")

    items = []
    for line in files_file.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if ">>>" not in line:
            raise ValueError(f"Bad line (expected 'url >>> location'): {line}")

        url, location = [part.strip() for part in line.split(">>>", 1)]
        if not url or not location:
            raise ValueError(f"Bad line (empty url/location): {line}")
        items.append((url, location))
    return items


should_quit = False
def download_file(url: str, location: str):
    if should_quit:
        return

    out_path = Path(location)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log(f"Skipping existing file: '{location}' which was last changed at {datetime.fromtimestamp(out_path.lstat().st_mtime)}")
        return "skipped"

    log(f"Starting: {url} -> {location}")
    try:
        with urllib.request.urlopen(url, timeout = 15) as response, open(out_path, "wb") as f:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 256)
                if should_quit or not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = (downloaded / total) * 100
                    log(f"Downloading {location}: {downloaded}/{total} bytes ({pct:.1f}%)")
                else:
                    log(f"Downloading {location}: {downloaded} bytes")

        if should_quit:
            log(f"Quitting on {location}")
            raise Exception()

        log(f"Finished: {location}")
        return "downloaded"

    except Exception:
        out_path.unlink(missing_ok = True)
        raise


def cleanup(items):
    log("Cleaning up ...")

    venv_path = Path(".venv")
    if venv_path.exists():
        shutil.rmtree(Path(".venv"))
        log(f"Removed {venv_path}")

    for _, location in items:
        location_path = Path(location)
        if location_path.exists():
            Path.unlink(location_path, missing_ok=True)
            log(f"Removed {location_path}")

    log("Done cleaning!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host_url", nargs="?", help="Base host URL")
    parser.add_argument("files_file", nargs="?", default="files.txt", help="File list")
    parser.add_argument("--clean", action="store_true", help="Remove .venv and downloads")
    args = parser.parse_args()

    if args.clean:
        cleanup(read_files_list(Path(args.files_file)))
        return

    if not args.host_url:
        parser.print_help()
        sys.exit(1)

    host_url = args.host_url.rstrip("/")
    items = read_files_list(Path(args.files_file))
    log(f"Downloading {len(items)} files in parallel")
    raised_errors = False
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as pool:
        futures = []
        for url_part, location in items:
            futures.append(pool.submit(download_file, f"{host_url}/{url_part.lstrip('/')}", location))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log(f"Download failed: {e}")
                raised_errors = True

    if raised_errors:
        sys.exit(1)
    log("Downloads completed!")

    install_requirements(Path(".venv"), Path("requirements.txt"))
    log("All done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        should_quit = True
