"""Distinct valid entries in non-power-of-two memories must not alias."""

from __future__ import annotations

import subprocess

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Assign, Id, MemRead, MemWrite, Module, Signal


def test_non_power_of_two_memory_read_write(tmp_path) -> None:
    mod = Module(
        name="MemDepth",
        signals={
            "clock": Signal("clock", 1, "input"),
            "addr": Signal("addr", 3, "input"),
            "data": Signal("data", 8, "input"),
            "we": Signal("we", 1, "input"),
            "out": Signal("out", 8, "output"),
            "m": Signal("m", 8, "mem", depth=6),
        },
        assigns=[Assign("out", MemRead("m", Id("addr")))],
        always=AlwaysFF("clock", []),
        ports=["clock", "addr", "data", "we", "out"],
        mem_writes=[MemWrite("m", Id("addr"), Id("data"), Id("we"))],
    )
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\n'
        "int main() {\n"
        "  MemDepthDut d;\n"
        "  for (unsigned i = 0; i < 6; ++i) {\n"
        "    d.addr = i; d.data = i + 11; d.we = 1; d.tick();\n"
        "  }\n"
        "  d.we = 0;\n"
        "  for (unsigned i = 0; i < 6; ++i) {\n"
        "    d.addr = i; d.poke_inputs(); d.eval_out();\n"
        "    if (d.out != i + 11) return i + 1;\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    )
    subprocess.run(
        ["c++", "-std=c++17", str(tmp_path / "check.cpp"),
         "-o", str(tmp_path / "check")],
        check=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True)
