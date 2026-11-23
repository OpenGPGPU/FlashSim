from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

from flashsim.circt_frontend import (
    CirctError,
    circt_import_parallelism,
    circt_verilog_to_mlir,
    parse_hw_mlir,
    verilog_sources,
    verilog_dependencies,
)
from flashsim.emit import emit_cpp, emit_split_cpp, split_dut_methods
from flashsim.data_regions import group_primitive_storage
from flashsim.gc_policy import compilation_gc_policy
from flashsim.phase_timing import phase
from flashsim.ir import Module
from flashsim.opt import _optimize_owned as optimize
from flashsim.parse import parse_verilog
from flashsim.toolchain import find_bin


@compilation_gc_policy
def compile_verilog(
    src: str,
    frontend: str = "auto",
    path: Path | None = None,
    mlir_path: Path | None = None,
    *,
    assertions: bool = False,
) -> tuple[Module, str]:
    mod = _compile_module(src, frontend, path, mlir_path, assertions=assertions)
    with phase('emit_cpp'):
        cpp = emit_cpp(mod)
    return mod, cpp


def prepare_circt_mlir(path: Path, mlir_path: Path, *, assertions: bool = False) -> None:
    """Import RTL into a reusable MLIR file without parsing or generating C++."""
    workers = circt_import_parallelism()
    # Keep both legacy values reusable. Older FlashSim versions reject a
    # marker containing a worker limit and regenerate their own cache.
    threading = {None: "0", 0: "1"}.get(workers, f"threads:{workers}")
    fingerprint = _assertion_sources(path) if assertions else None
    if (_mlir_matches(mlir_path, path, threading, assertions=assertions)
            and _mlir_fresh(mlir_path, path, fingerprint)):
        return

    mlir_path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=mlir_path.parent, prefix=mlir_path.name + '.',
                                suffix='.tmp', delete=False)
    temporary = Path(stream.name)
    try:
        with stream, phase('frontend.circt'):
            circt_verilog_to_mlir(path, assertions=assertions, output=stream)
        if not temporary.stat().st_size:
            raise CirctError("circt-verilog produced no MLIR")
        if assertions and fingerprint != _assertion_sources(path):
            raise CirctError("assertion inputs changed during CIRCT compilation")
        threading_marker = mlir_path.with_suffix(".mlir.circt-serial")
        threading_marker.unlink(missing_ok=True)
        temporary.replace(mlir_path)
        mlir_path.with_suffix(".mlir.top").write_text(path.stem)
        mlir_path.with_suffix(".mlir.assertions").write_text("1" if assertions else "0")
        if assertions:
            mlir_path.with_suffix(".mlir.assertion-sources.json").write_text(
                json.dumps(fingerprint, sort_keys=True))
        threading_marker.write_text(threading)
    finally:
        temporary.unlink(missing_ok=True)


def _compile_module(
    src: str, frontend: str, path: Path | None, mlir_path: Path | None, *,
    assertions: bool,
) -> Module:
    preflatten = os.environ.get('FLASHSIM_PREFLATTEN_WIRES', '0')
    if preflatten not in {'0', '1'}:
        raise ValueError('FLASHSIM_PREFLATTEN_WIRES must be 0 or 1')
    kind = _resolve_frontend(frontend)
    if kind == "circt":
        if path is None:
            raise CirctError("CIRCT frontend needs a Verilog file path")
        if mlir_path is None:
            fingerprint = _assertion_sources(path) if assertions else None
            with phase('frontend.circt'):
                mlir = circt_verilog_to_mlir(path, assertions=assertions)
            if assertions and fingerprint != _assertion_sources(path):
                raise CirctError("assertion inputs changed during CIRCT compilation")
        else:
            prepare_circt_mlir(path, mlir_path, assertions=assertions)
            mlir = mlir_path
        with phase('parse.hw_mlir'):
            if preflatten == '1':
                mod = parse_hw_mlir(mlir, top=path.stem, preflatten_wires=True)
            else:
                mod = parse_hw_mlir(mlir, top=path.stem)
        del mlir
    else:
        if assertions:
            raise CirctError("bound assertions require the CIRCT frontend")
        with phase('parse.verilog'):
            mod = parse_verilog(src)
    with phase('optimize'):
        owned = [mod]
        del mod
        mod = optimize(owned)
    return mod


@compilation_gc_policy
def compile_file(
    path: Path,
    out_h: Path,
    frontend: str = "auto",
    *,
    split: int = 0,
    assertions: bool = False,
) -> Module:
    """Compile Verilog to `out_h`.

    When `split > 0`, large DUT method bodies are written to `dut_0.cpp` …
    `dut_{split-1}.cpp` beside the header so each shard can be -O2'd in
    parallel (needed for GpuHostSystemAxi-sized tops).
    """
    try:
        region_fields = int(os.environ.get("FLASHSIM_DATA_REGION_FIELDS", "0"))
    except ValueError as exc:
        raise ValueError("FLASHSIM_DATA_REGION_FIELDS must be an integer >= 0") from exc
    if region_fields < 0:
        raise ValueError("FLASHSIM_DATA_REGION_FIELDS must be >= 0")
    out_h.parent.mkdir(parents=True, exist_ok=True)
    # A direct compilation can replace a benchmark's DUT. Do not leave its
    # old generation-options record attached to the replacement.
    (out_h.parent / ".flashsim_emit_options.json").unlink(missing_ok=True)
    mlir_path = out_h.parent / "hw.mlir" if _resolve_frontend(frontend) == "circt" else None
    if split > 0:
        mod = _compile_module(path.read_text(), frontend, path, mlir_path,
                              assertions=assertions)
        header, parts = emit_split_cpp(mod, split)
    else:
        mod, cpp = compile_verilog(
            path.read_text(), frontend=frontend, path=path, mlir_path=mlir_path,
            assertions=assertions,
        )
        if split == 0:
            split = _auto_split_count(path, len(cpp))
        if split > 0:
            with phase('split_dut_methods'):
                header, parts = split_dut_methods(cpp, n_shards=split)
            del cpp
    wanted: set[str] = set()
    if split > 0:
        _write_generated(out_h, group_primitive_storage(
            header, f"{mod.name}Dut", region_fields))
        for name, body in parts:
            wanted.add(name)
            _write_generated(out_h.parent / name, body)
    else:
        _write_generated(out_h, group_primitive_storage(
            cpp, f"{mod.name}Dut", region_fields))
    # Remove only obsolete shards. Keeping identical source/header mtimes
    # lets ccache's direct mode reuse a large model after regeneration.
    for stale in out_h.parent.glob("dut_*.cpp"):
        if stale.name not in wanted:
            stale.unlink()
    return mod


def _encoded_chunks(text: str) -> Iterator[bytes]:
    chars_per_chunk = 1024 * 1024
    for start in range(0, len(text), chars_per_chunk):
        yield text[start:start + chars_per_chunk].encode()


def _write_generated(path: Path, text: str) -> None:
    if path.is_file():
        with path.open('rb') as previous:
            if (all(previous.read(len(chunk)) == chunk for chunk in _encoded_chunks(text))
                    and not previous.read(1)):
                return
    with path.open('wb') as output:
        output.writelines(_encoded_chunks(text))


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


def _mlir_fresh(mlir_path: Path, verilog: Path, fingerprint: dict[str, str] | None) -> bool:
    if not mlir_path.is_file():
        return False
    if fingerprint is not None:
        try:
            recorded = json.loads(mlir_path.with_suffix(".mlir.assertion-sources.json").read_text())
            return recorded == fingerprint
        except (OSError, ValueError):
            return False
    stamp = mlir_path.stat().st_mtime
    return all(src.stat().st_mtime <= stamp for src in verilog_sources(verilog))


def _assertion_sources(verilog: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    verilog_dependencies(verilog, hashes=hashes)
    return hashes


def _resolve_frontend(frontend: str) -> str:
    if frontend == "auto":
        return "circt" if find_bin("circt-verilog") else "native"
    if frontend not in {"circt", "native"}:
        raise ValueError(f"unknown frontend {frontend}")
    return frontend


def _mlir_matches(mlir_path: Path, verilog: Path, threading: str, *,
                  assertions: bool = False) -> bool:
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
        marker = mlir_path.with_suffix(".mlir.assertions")
        cached_assertions = marker.is_file() and marker.read_text().strip() == "1"
        threading_marker = mlir_path.with_suffix(".mlir.circt-serial")
        cached_threading = threading_marker.read_text().strip()
        return (cached_assertions == assertions
                and cached_threading == threading
                and mlir_path.with_suffix(".mlir.top").read_text().strip() == verilog.stem)
    except OSError:
        return False
