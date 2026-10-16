"""Custom eager groups must retain lazy updates outside the drained closure."""
import re
import shutil
import subprocess

import pytest

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, BinOp, Const, Id, Module, NbAssign, Signal


def test_later_unscheduled_producer_updates_registers(tmp_path, monkeypatch):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    monkeypatch.setattr(emit, "_LARGE_WRITES", 8)
    monkeypatch.setattr(emit, "_EAGER_KEY_RE", re.compile(r"(a|z)"))
    monkeypatch.setattr(emit, "cached_wires", lambda mod, assigns: set(assigns))
    signals = {name: Signal(name, width, kind) for name, width, kind in [
        ("clock", 1, "input"), ("a", 32, "input"), ("b", 32, "input"),
        ("out", 32, "output"), ("z_source_t0", 32, "wire"),
        ("a_target_t0", 32, "wire")
    ]}
    assigns = [
        Assign("z_source_t0", BinOp("+", Id("a"), Const(1, 32))),
        Assign("a_target_t0", BinOp("^", Id("z_source_t0"), Id("b"))),
        Assign("out", Id("r0")),
    ]
    # Trigger eager grouping while leaving the two actual groups independent.
    for i in range(130):
        signals[f"padding{i}"] = Signal(f"padding{i}", 32, "wire")
        assigns.append(Assign(f"padding{i}", Const(i, 32)))
    body = []
    for i in range(9):
        signals[f"r{i}"] = Signal(f"r{i}", 32, "reg")
        body.append(NbAssign(f"r{i}", BinOp("+", Id("a_target_t0"), Const(i, 32))))
    mod = Module("GroupingProof", signals, assigns, AlwaysFF("clock", body),
                 ["clock", "a", "b", "out"])
    (tmp_path / "dut.h").write_text(emit.emit_cpp(mod))
    source = tmp_path / "check.cpp"
    source.write_text('''#include "dut.h"
int main() {
  GroupingProofDut d;
  d.a=2; d.b=1; d.tick();
  if(d.r0!=2 || d.r8!=10)return 1;
  d.a=5; d.tick();
  if(d.r0!=7 || d.r8!=15)return 2;
  d.b=3; d.tick();
  return d.r0==5 && d.r8==13 ? 0 : 3;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(source), "-o", str(binary)],
                   check=True)
    subprocess.run([str(binary)], check=True)
