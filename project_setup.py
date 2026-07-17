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


def venv_python(venv_dir: Path) -> str:
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def create_venv(venv_dir: Path):
    log("Creating .venv")
    EnvBuilder(with_pip=True).create(venv_dir)


def install_requirements(python_bin: str, requirements_file: Path):
    log("Upgrading pip")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])
    if requirements_file.exists():
        log("Installing requirements")
        subprocess.check_call(
            [python_bin, "-m", "pip", "install", "-r", str(requirements_file)]
        )
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


def download_file(url: str, location: str):
    out_path = Path(location)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log(f"Skipping existing file: {location}")
        return "skipped"

    log(f"Starting: {url} -> {location}")
    try:
        with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = (downloaded / total) * 100
                    log(
                        f"Downloading {location}: {downloaded}/{total} bytes ({pct:.1f}%)"
                    )
                else:
                    log(f"Downloading {location}: {downloaded} bytes")

        log(f"Finished: {location}")
        return "downloaded"
    except Exception:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
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
    parser.add_argument(
        "--clean", action="store_true", help="Remove .venv and downloads"
    )
    args = parser.parse_args()

    if args.clean:
        cleanup(read_files_list(Path(args.files_file)))
        return

    if not args.host_url:
        parser.print_help()
        parser.print_usage()
        sys.exit(1)
        # print(f"Usage: {sys.argv[0]} HOST_URL [files.txt] [--clean]", file=sys.stderr)

    items = read_files_list(Path(args.files_file))
    if not items:
        log("No files to download")
        return

    log(f"Downloading {len(items)} files in parallel")
    host_url = args.host_url.rstrip("/")
    errors = []
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
        futures = []
        for url_part, location in items:
            full_url = f"{host_url}/{url_part.lstrip('/')}"
            futures.append(pool.submit(download_file, full_url, location))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                errors.append(e)
                log(f"Download failed: {e}")

    venv_dir = Path(".venv")
    create_venv(venv_dir)
    install_requirements(venv_python(venv_dir), Path("requirements.txt"))

    if errors:
        sys.exit(1)

    log("All done!")


if __name__ == "__main__":
    main()
