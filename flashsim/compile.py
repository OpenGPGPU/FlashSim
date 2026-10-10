from __future__ import annotations

import os
from pathlib import Path

from flashsim.circt_frontend import (
    CirctError,
    circt_verilog_to_mlir,
    parse_hw_mlir,
    verilog_sources,
)
from flashsim.emit import emit_cpp, split_dut_methods
from flashsim.ir import Module
from flashsim.opt import optimize
from flashsim.parse import parse_verilog
from flashsim.toolchain import find_bin


def compile_verilog(
    src: str,
    frontend: str = "auto",
    path: Path | None = None,
    mlir_path: Path | None = None,
) -> tuple[Module, str]:
    kind = _resolve_frontend(frontend)
    if kind == "circt":
        if path is None:
            raise CirctError("CIRCT frontend needs a Verilog file path")
        if mlir_path is None or not (
            _mlir_fresh(mlir_path, path) and _mlir_matches(mlir_path, path)
        ):
            mlir = circt_verilog_to_mlir(path)
            if mlir_path is not None:
                mlir_path.parent.mkdir(parents=True, exist_ok=True)
                mlir_path.write_text(mlir)
                mlir_path.with_suffix(".mlir.top").write_text(path.stem)
        else:
            mlir = mlir_path.read_text()
        mod = parse_hw_mlir(mlir, top=path.stem)
    else:
        mod = parse_verilog(src)
    mod = optimize(mod)
    return mod, emit_cpp(mod)


def compile_file(
    path: Path,
    out_h: Path,
    frontend: str = "auto",
    *,
    split: int = 0,
) -> Module:
    """Compile Verilog to `out_h`.

    When `split > 0`, large DUT method bodies are written to `dut_0.cpp` …
    `dut_{split-1}.cpp` beside the header so each shard can be -O2'd in
    parallel (needed for GpuHostSystemAxi-sized tops).
    """
    out_h.parent.mkdir(parents=True, exist_ok=True)
    mlir_path = out_h.parent / "hw.mlir" if _resolve_frontend(frontend) == "circt" else None
    mod, cpp = compile_verilog(
        path.read_text(), frontend=frontend, path=path, mlir_path=mlir_path
    )
    # Drop stale shards from a prior split emit.
    for stale in out_h.parent.glob("dut_*.cpp"):
        stale.unlink()
    if split == 0:
        split = _auto_split_count(path, len(cpp))
    if split > 0:
        header, parts = split_dut_methods(cpp, n_shards=split)
        out_h.write_text(header)
        for name, body in parts:
            (out_h.parent / name).write_text(body)
    else:
        out_h.write_text(cpp)
    return mod


def _auto_split_count(path: Path, cpp_bytes: int) -> int:
    """Pick a bounded shard count for generated multi-CU GPU models.

    Small designs keep the historical monolithic output.  Once a generated
    GPU top crosses a few hundred megabytes, compiling one translation unit
    dominates the workflow and can exhaust the compiler's memory.  The
    threshold is based on emitted C++ size, so it follows elaboration size
    without importing or changing any GPU configuration.
    """
    if path.stem not in {"GpuSystem", "GpuHostSystemAxi"}:
        return 0
    # 4-CU GpuSystem/GpuHostSystemAxi emits roughly 550-600 MB.  Keep the
    # existing 16-way split for the older large snapshots and use 32 shards
    # once the multi-CU expansion crosses 400 MB.
    if cpp_bytes <= 200_000_000:
        return 0
    override = os.environ.get("FLASHSIM_DUT_SHARDS")
    if override is not None:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError("FLASHSIM_DUT_SHARDS must be an integer") from exc
        if value < 0:
            raise ValueError("FLASHSIM_DUT_SHARDS must be >= 0")
        return value
    return 32 if cpp_bytes > 400_000_000 else 16


def _mlir_fresh(mlir_path: Path, verilog: Path) -> bool:
    if not mlir_path.is_file():
        return False
    stamp = mlir_path.stat().st_mtime
    return all(src.stat().st_mtime <= stamp for src in verilog_sources(verilog))


def _resolve_frontend(frontend: str) -> str:
    if frontend == "auto":
        return "circt" if find_bin("circt-verilog") else "native"
    if frontend not in {"circt", "native"}:
        raise ValueError(f"unknown frontend {frontend}")
    return frontend


def _mlir_matches(mlir_path: Path, verilog: Path) -> bool:
    """Whether the cached MLIR was produced from *this* Verilog top.

    Freshness by mtime alone is not enough: the cache lives in the output
    directory, so compiling a second design into a directory that already
    holds a cache finds every source older than the cache and silently reuses
    the wrong design — the emitter then compiles the previous DUT. The top
    name is recorded in a sidecar rather than scanned for in the MLIR, because
    CIRCT emits submodules first and the top can sit megabytes into the file.
    A cache with no sidecar predates this check and is treated as stale, so
    the next run regenerates it once and every run after that reuses it.
    """
    try:
        return mlir_path.with_suffix(".mlir.top").read_text().strip() == verilog.stem
    except OSError:
        return False
