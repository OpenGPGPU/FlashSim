"""Deep priority muxes keep lazy demand and compile without nested scopes."""

import shutil
import subprocess
import sys

import pytest

import flashsim.emit as emitter
from flashsim.ir import BinOp, Const, Id, Signal, Ternary


def _compile_and_run(tmp_path, source):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    cpp = tmp_path / "check.cpp"
    cpp.write_text(source)
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def _reset_emitter_globals(monkeypatch):
    monkeypatch.setattr(emitter, "_EAGER_DONE", set())
    monkeypatch.setattr(emitter, "_WIRE_PART", {})


def test_deep_cached_priority_is_iterative_and_lazy(tmp_path, monkeypatch):
    _reset_emitter_globals(monkeypatch)
    # This depth exceeds Python's normal recursion limit. Repeated conditions
    # and a nested mux also check priority and which demand calls actually run.
    count = 1200
    inner = Const(2004, 32)
    for i in reversed(range(3)):
        inner = Ternary(Id(f"inner{i}"), Const(2001 + i, 32), inner)
    expr = Const(65535, 32)
    for i in reversed(range(count)):
        match = 2 if i == count - 1 else i
        arm = inner if i == 2 else Const(i + 1, 32)
        expr = Ternary(BinOp("==", Id("selected"), Const(match, 32)), arm, expr)
    sigs = {"selected": Signal("selected", 32, "input")}
    sigs.update({f"inner{i}": Signal(f"inner{i}", 1, "input") for i in range(3)})
    cached = set(sigs)
    lines = []
    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(600)
        emitter._emit_narrow_from_wide("result", expr, 32, cached, sigs, lines, 2, [0])
    finally:
        sys.setrecursionlimit(previous_limit)
    source = '''#include <cstdint>
int main() {
  uint32_t selected, result, selected_calls, inner_calls;
  uint8_t inner0=0, inner1=1, inner2=1;
  auto eval_selected = [&] { ++selected_calls; };
  auto eval_inner0 = [&] { ++inner_calls; };
  auto eval_inner1 = [&] { ++inner_calls; };
  auto eval_inner2 = [&] { ++inner_calls; };
  auto run = [&] {
''' + "\n".join(lines) + '''
  };
  selected=0; selected_calls=inner_calls=0; run();
  if(result!=1 || selected_calls!=1 || inner_calls!=0) return 1;
  selected=2; selected_calls=inner_calls=0; run();
  if(result!=2002 || selected_calls!=3 || inner_calls!=2) return 2;
  selected=1198; selected_calls=inner_calls=0; run();
  if(result!=1199 || selected_calls!=1199 || inner_calls!=0) return 3;
  selected=1199; selected_calls=inner_calls=0; run();
  if(result!=65535 || selected_calls!=1200 || inner_calls!=0) return 4;
}
'''
    _compile_and_run(tmp_path, source)


def test_wide_conditions_and_cached_demands_remain_lazy(tmp_path, monkeypatch):
    _reset_emitter_globals(monkeypatch)
    sigs = {name: Signal(name, 256, "input") for name in ("a", "b", "c")}
    sigs["gate"] = Signal("gate", 1, "input")
    expr = Ternary(BinOp("==", Id("a"), Id("b")), Const(11, 32),
        Ternary(Id("gate"), Const(22, 32),
        Ternary(BinOp("==", Id("a"), Id("c")), Const(33, 32), Const(44, 32))))
    lines = []
    emitter._emit_narrow_from_wide("result", expr, 32, {"gate"}, sigs, lines, 2, [0])
    source = '''#include <cstdint>
int fs_ucmp_limbs(const uint64_t *a, const uint64_t *b, unsigned n) {
  while (n--) if (a[n]!=b[n]) return a[n]<b[n] ? -1 : 1;
  return 0;
}
int main() {
  uint64_t a[4]={}, b[4]={}, c[4]={};
  uint8_t gate=0; uint32_t result=0, calls=0;
  auto eval_gate = [&] { ++calls; };
  auto run = [&] {
''' + "\n".join(lines) + '''
  };
  run(); if (result!=11 || calls!=0) return 1;
  b[3]=1; gate=1; run(); if (result!=22 || calls!=1) return 2;
  gate=0; run(); if (result!=33 || calls!=2) return 3;
  c[2]=1; run(); if (result!=44 || calls!=3) return 4;
}
'''
    _compile_and_run(tmp_path, source)
