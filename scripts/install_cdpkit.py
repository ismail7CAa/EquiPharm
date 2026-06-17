#!/usr/bin/env python
"""Install Linux CDPKit binaries under external/CDPKit for benchmark runs."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path


RELEASE_API_URL = "https://api.github.com/repos/molinfo-vienna/CDPKit/releases/latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install official Linux CDPKit binaries under external/CDPKit.")
    parser.add_argument("--prefix", type=Path, default=Path("external/CDPKit"))
    parser.add_argument("--asset-url", help="Override the detected CDPKit release asset URL.")
    parser.add_argument("--keep-download", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for each installer attempt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.prefix.mkdir(parents=True, exist_ok=True)
    asset_url = args.asset_url or detect_asset_url()
    download_path = args.prefix.parent / Path(asset_url).name
    print(f"Downloading {asset_url}")
    download(asset_url, download_path)
    try:
        install_asset(download_path, args.prefix, timeout=args.timeout)
    finally:
        if not args.keep_download and download_path.exists():
            download_path.unlink()

    normalize_install_layout(args.prefix)
    psdcreate = args.prefix / "Bin" / "psdcreate"
    if not psdcreate.exists():
        raise SystemExit(f"CDPKit install did not produce expected executable: {psdcreate}")
    print(f"CDPKit installed at: {args.prefix}")
    print(f"Use Bin path: {args.prefix / 'Bin'}")


def detect_asset_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux" or machine not in {"x86_64", "amd64"}:
        raise SystemExit(
            "This installer is intentionally Linux x86_64 only. On the Linux "
            "Jupyter server, run `python scripts/install_cdpkit.py`."
        )

    with urllib.request.urlopen(RELEASE_API_URL) as response:
        release = json.load(response)

    for asset in release["assets"]:
        if asset["name"].endswith("Linux-x86_64.sh"):
            return asset["browser_download_url"]
    raise SystemExit("Could not find a Linux x86_64 CDPKit release asset.")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing download: {destination}")
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def install_asset(asset_path: Path, prefix: Path, *, timeout: int) -> None:
    if asset_path.suffix == ".sh":
        install_linux_shell_asset(asset_path, prefix, timeout=timeout)
        return
    raise SystemExit(f"Unsupported CDPKit installer asset: {asset_path}")


def install_linux_shell_asset(asset_path: Path, prefix: Path, *, timeout: int) -> None:
    asset_path.chmod(asset_path.stat().st_mode | 0o755)
    commands = [
        [
            str(asset_path),
            "--root",
            str(prefix),
            "--accept-licenses",
            "--default-answer",
            "--confirm-command",
            "install",
        ],
        [str(asset_path), "--target", str(prefix), "--noexec"],
        [str(asset_path), "-b", "-p", str(prefix)],
        [str(asset_path), "--prefix", str(prefix), "--batch"],
    ]
    errors = []
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            errors.append(f"{' '.join(command)}\nTimed out after {timeout} seconds.")
            continue
        normalize_install_layout(prefix)
        if result.returncode == 0 and (prefix / "Bin" / "psdcreate").exists():
            return
        errors.append(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
    raise SystemExit(
        "Could not install the Linux CDPKit shell asset automatically. "
        "Try running the downloaded installer manually with a prefix under external/CDPKit.\n"
        + "\n\n".join(errors)
    )


def normalize_install_layout(prefix: Path) -> None:
    """Ensure prefix/Bin points at the directory containing psdcreate."""
    direct = prefix / "Bin" / "psdcreate"
    if direct.exists():
        return

    matches = sorted(prefix.rglob("psdcreate"))
    if not matches:
        return
    bin_dir = matches[0].parent
    target_bin = prefix / "Bin"
    if target_bin.exists() or target_bin.is_symlink():
        return
    try:
        target_bin.symlink_to(bin_dir.relative_to(prefix), target_is_directory=True)
    except ValueError:
        target_bin.symlink_to(bin_dir, target_is_directory=True)


if __name__ == "__main__":
    main()
