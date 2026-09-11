from __future__ import annotations

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
        if mlir_path is None or not _mlir_fresh(mlir_path, path):
            mlir = circt_verilog_to_mlir(path)
            if mlir_path is not None:
                mlir_path.parent.mkdir(parents=True, exist_ok=True)
                mlir_path.write_text(mlir)
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
    if split > 0:
        header, parts = split_dut_methods(cpp, n_shards=split)
        out_h.write_text(header)
        for name, body in parts:
            (out_h.parent / name).write_text(body)
    else:
        out_h.write_text(cpp)
    return mod


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
