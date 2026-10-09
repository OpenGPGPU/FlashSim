"""A slice wider than a uint64_t must retain its upper bits."""

from __future__ import annotations

import subprocess

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Extract, Id, Module, NbAssign, Signal


def test_75_bit_slice_from_wide_register(tmp_path) -> None:
    mod = Module(
        name="WideSlice",
        signals={
            "clock": Signal("clock", 1, "input"),
            "src": Signal("src", 162, "input"),
            "result": Signal("result", 75, "reg"),
        },
        assigns=[],
        always=AlwaysFF("clock", [NbAssign("result", Extract(Id("src"), 0, 75))]),
        ports=["clock", "src"],
    )
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\n'
        'int main() {\n'
        '  WideSliceDut d;\n'
        '  d.src[0] = 0xffffafffffffffffULL;\n'
        '  d.src[1] = 0x7ffULL;\n'
        '  d.tick();\n'
        '  return d.result == (((unsigned __int128)0x7ff << 64) | '
        '0xffffafffffffffffULL) ? 0 : 1;\n'
        '}\n'
    )
    subprocess.run(
        ["c++", "-std=c++17", str(tmp_path / "check.cpp"), "-o", str(tmp_path / "check")],
        check=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True)
