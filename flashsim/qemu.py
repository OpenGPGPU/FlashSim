"""In-process QEMU/ARTI glue: compile GpuHostAxi and run the MMIO smoke."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from flashsim.arti_model import write_arti_model
from flashsim.compile import compile_file
from flashsim.toolchain import repo_root


def _need(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required tool not found: {name}")
    return path


def run_qemu_smoke(frontend: str = "circt") -> int:
    root = repo_root()
    rtl = root / "rtl" / "gpu" / "host-axi" / "GpuHostAxi.sv"
    work = root / "build" / "GpuHostAxi"
    compile_file(rtl, work / "dut.h", frontend=frontend)
    header, source, smoke = write_arti_model(work)
    cxx = _need("c++")
    out = work / "arti_smoke"
    proc = subprocess.run(
        [
            cxx,
            "-O3",
            "-std=c++17",
            "-Wno-parentheses-equality",
            "-fbracket-depth=4096",
            "-I",
            str(work),
            str(source),
            str(smoke),
            "-o",
            str(out),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    run = subprocess.run([str(out)], text=True, capture_output=True)
    sys.stdout.write(run.stdout)
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        print(f"FAIL  {header.name} via {out.name} rc={run.returncode}")
        return run.returncode or 1
    print(f"GO  ARTI MMIO API on FlashSim GpuHostAxi ({out})")
    return 0
