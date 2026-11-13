#!/usr/bin/env python3
"""Compare GC policies in isolated, real native-frontend compilations.

Run from the repository root with ``python -B scripts/bench_compile_gc.py``.
No model or measurement files are written. This is a synthetic Python
generation benchmark, not a GPU simulation or a CIRCT import benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def _worker(count: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flashsim.compile import compile_verilog

    ports = ["input clock", "input [31:0] a", "input [31:0] b"]
    ports.extend(f"output [31:0] y{i}" for i in range(count))
    lines = ["module ManyRegisters(" + ", ".join(ports) + ");"]
    for i in range(count):
        lines.extend([f"reg [31:0] q{i};", f"assign y{i} = q{i};"])
    lines.append("always @(posedge clock) begin")
    lines.extend(f"q{i} <= (a ^ 32'd{i + 1}) + b;" for i in range(count))
    lines.extend(["end", "endmodule"])
    source = "\n".join(lines)
    seconds = 0.0
    collections = 0
    started = 0.0

    def collection(phase: str, info: dict) -> None:
        nonlocal seconds, collections, started
        if phase == "start":
            started = time.perf_counter()
        else:
            seconds += time.perf_counter() - started
            collections += 1

    gc.collect()
    gc.callbacks.append(collection)
    begin = time.perf_counter()
    try:
        model, cpp = compile_verilog(source, frontend="native")
        elapsed = time.perf_counter() - begin
    finally:
        gc.callbacks.remove(collection)
    print(json.dumps({
        "policy": os.environ["FLASHSIM_GC_POLICY"], "registers": count,
        "seconds": elapsed, "gc_seconds": seconds, "collections": collections,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "cpp_bytes": len(cpp.encode()),
        "cpp_sha256": hashlib.sha256(cpp.encode()).hexdigest(),
        "assertions": len(model.assertions),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registers", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.registers <= 0 or args.repeat <= 0:
        parser.error("registers and repeat must be positive")
    if args.worker:
        _worker(args.registers)
        return
    digest = None
    for repeat in range(args.repeat):
        for policy in ["normal", "defer"]:
            env = os.environ.copy()
            env.update(FLASHSIM_GC_POLICY=policy, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([
                sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                "--registers", str(args.registers),
            ], env=env, check=True, capture_output=True, text=True)
            record = json.loads(result.stdout)
            if digest is None:
                digest = record["cpp_sha256"]
            if record["cpp_sha256"] != digest:
                raise RuntimeError("GC policy changed emitted C++")
            # Only the final numeric results are printed, never model data.
            record.pop("cpp_sha256")
            print(json.dumps({"repeat": repeat + 1, **record}), flush=True)


if __name__ == "__main__":
    main()
