from __future__ import annotations

from pathlib import Path

from flashsim.circt_frontend import CirctError, import_verilog_file
from flashsim.emit import emit_cpp
from flashsim.ir import Module
from flashsim.opt import optimize
from flashsim.parse import parse_verilog
from flashsim.toolchain import find_bin


def compile_verilog(
    src: str,
    frontend: str = "auto",
    path: Path | None = None,
) -> tuple[Module, str]:
    kind = _resolve_frontend(frontend)
    if kind == "circt":
        if path is None:
            raise CirctError("CIRCT frontend needs a Verilog file path")
        mod = import_verilog_file(path)
    else:
        mod = parse_verilog(src)
    mod = optimize(mod)
    return mod, emit_cpp(mod)


def compile_file(path: Path, out_h: Path, frontend: str = "auto") -> Module:
    mod, cpp = compile_verilog(path.read_text(), frontend=frontend, path=path)
    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_h.write_text(cpp)
    return mod


def _resolve_frontend(frontend: str) -> str:
    if frontend == "auto":
        return "circt" if find_bin("circt-verilog") else "native"
    if frontend not in {"circt", "native"}:
        raise ValueError(f"unknown frontend {frontend}")
    return frontend
