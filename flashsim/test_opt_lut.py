"""arcilator lookup tables for small functions."""

from __future__ import annotations

import subprocess

from flashsim.emit import emit_cpp
from flashsim.ir import (
    AlwaysFF,
    Assign,
    BinOp,
    Const,
    Id,
    Module,
    Signal,
)
from flashsim.opt import optimize


def _sigs() -> dict[str, Signal]:
    return {
        "clock": Signal("clock", 1, "input"),
        "x": Signal("x", 3, "input"),
        "a": Signal("a", 1, "input"),
        "b": Signal("b", 1, "input"),
        "c": Signal("c", 1, "input"),
        "f": Signal("f", 1, "wire"),
        "out": Signal("out", 1, "output"),
        "flag2": Signal("flag2", 1, "output"),
    }


def _rhs7():
    expr = BinOp("==", Id("x"), Const(0, 3))
    for v in (1, 2, 3, 4, 6, 7):
        expr = BinOp("|", expr, BinOp("==", Id("x"), Const(v, 3)))
    return expr


def test_single_leaf_decode_becomes_lut() -> None:
    rhs = _rhs7()
    mod = Module("Top", _sigs(), [Assign("f", rhs), Assign("out", Id("f")), Assign("flag2", Id("f"))], AlwaysFF("clock", []), ["clock"])
    out = optimize(mod)
    assigns = {a.lhs: a.rhs for a in out.assigns}
    assert "f" in assigns
    # LUT form: Extract(Const(table,8) >> x, 0, 1) with bits 0,1,2,3,4,6,7 set.
    assert assigns["f"] == __import__("flashsim.ir", fromlist=["Extract"]).Extract(BinOp(">>", Const(223, 8), Id("x")), 0, 1)


def test_lut_native_cpp_matches_truth_table(tmp_path) -> None:
    rhs = _rhs7()
    mod = Module("Top", _sigs(), [Assign("f", rhs), Assign("out", Id("f")), Assign("flag2", Id("f"))], AlwaysFF("clock", []), ["clock"])
    opt = optimize(mod)
    (tmp_path / "dut.h").write_text(emit_cpp(opt))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\n'
        "int main() {\n"
        "  TopDut d;\n"
        "  const int want[8] = {1,1,1,1,1,0,1,1};\n"
        "  for (int v = 0; v < 8; v++) { d.x = v; d.tick(); d.eval_out(); if (d.out != want[v]) return 1; }\n"
        "  return 0;\n"
        "}\n"
    )
    subprocess.run(["c++", "-std=c++17", str(tmp_path / "check.cpp"), "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)
