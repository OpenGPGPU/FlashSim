from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

CIRCT_RELEASE = "firtool-1.158.0"
CIRCT_BASE = f"https://github.com/llvm/circt/releases/download/{CIRCT_RELEASE}"
REQUIRED_BINS = ("firtool", "circt-opt")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def circt_dir() -> Path:
    override = os.environ.get("FLASHSIM_CIRCT")
    if override:
        return Path(override)
    return repo_root() / "third_party" / "circt-release"


def _asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "circt-full-shared-macos-arm64.tar.gz"
    if system == "Darwin":
        return "circt-full-shared-macos-x64.tar.gz"
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return "circt-full-shared-linux-x64.tar.gz"
    raise RuntimeError(f"no CIRCT release asset for {system} {machine}")


def find_bin(name: str) -> Path | None:
    env = os.environ.get(name.upper().replace("-", "_"))
    if env:
        path = Path(env)
        if path.exists():
            return path
    direct = circt_dir() / "bin" / name
    if direct.exists():
        return direct
    if circt_dir().exists():
        for candidate in circt_dir().rglob(name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _toolchain_ok() -> bool:
    marker = circt_dir() / ".flashsim-release"
    if not marker.exists() or marker.read_text().strip() != CIRCT_RELEASE:
        return False
    return all(find_bin(name) is not None for name in REQUIRED_BINS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        subprocess.check_call(
            [curl, "-fL", "--retry", "3", "--retry-delay", "2", "-o", str(dest), url]
        )
        return
    urllib.request.urlretrieve(url, dest)


def setup_circt(force: bool = False) -> Path:
    dest = circt_dir()
    if _toolchain_ok() and not force:
        print(f"CIRCT {CIRCT_RELEASE} already installed at {dest}")
        return dest

    asset = _asset_name()
    url = f"{CIRCT_BASE}/{asset}"
    sha_url = f"{url}.sha256"
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / asset
    sha_file = dest.parent / f"{asset}.sha256"

    print(f"downloading {sha_url}")
    _download(sha_url, sha_file)
    expected = sha_file.read_text().split()[0].strip()

    print(f"downloading {url}")
    _download(url, archive)
    actual = _sha256(archive)
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"CIRCT archive checksum mismatch: {actual} != {expected} (truncated download?)"
        )

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink(missing_ok=True)
    sha_file.unlink(missing_ok=True)

    if not all(find_bin(name) is not None for name in REQUIRED_BINS):
        raise RuntimeError(f"CIRCT extract at {dest} is missing {REQUIRED_BINS}")
    (dest / ".flashsim-release").write_text(CIRCT_RELEASE + "\n")

    print(f"firtool:       {find_bin('firtool')}")
    print(f"circt-opt:     {find_bin('circt-opt')}")
    print(f"arcilator:     {find_bin('arcilator')}")
    print(f"circt-verilog: {find_bin('circt-verilog')}")
    return dest
