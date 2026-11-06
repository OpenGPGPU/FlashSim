"""Deep priority muxes keep lazy demand and compile without nested scopes."""

import shutil
import subprocess
import sys

import pytest

import flashsim.emit as emitter
from flashsim.ir import BinOp, Const, Id, If, NbAssign, Signal, Ternary


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


def test_index_equality_mux_is_a_switch_and_evaluates_one_arm(tmp_path, monkeypatch):
    _reset_emitter_globals(monkeypatch)
    expr = Const(99, 32)
    for i in reversed(range(8)):
        expr = Ternary(BinOp("==", Id("idx"), Const(i, 4)), Id(f"r{i}"), expr)
    sigs = {"idx": Signal("idx", 4, "input")}
    sigs.update({f"r{i}": Signal(f"r{i}", 32, "reg") for i in range(8)})
    cached = set(sigs)
    lines: list[str] = []
    emitter._emit_assign("result", expr, cached, sigs, lines, 2, None, [0])
    body = "\n".join(lines)
    assert "switch (idx)" in body
    assert "case 3u" in body
    source = '''#include <cstdint>
int main() {
  uint8_t idx = 0;
  uint32_t result = 0, idx_calls = 0, arm_calls = 0;
  uint32_t r0=10, r1=11, r2=12, r3=13, r4=14, r5=15, r6=16, r7=17;
  auto eval_idx = [&] { ++idx_calls; };
''' + "\n".join(
        f"  auto eval_r{i} = [&] {{ ++arm_calls; }};" for i in range(8)
    ) + '''
  auto run = [&] {
''' + body + '''
  };
  idx = 3; run();
  if (result != 13 || idx_calls != 1 || arm_calls != 1) return 1;
  idx = 9; idx_calls = arm_calls = 0; run();
  if (result != 99 || idx_calls != 1 || arm_calls != 0) return 2;
  return 0;
}
'''
    _compile_and_run(tmp_path, source)


def test_sequential_index_mux_is_a_switch_and_evaluates_one_arm(tmp_path, monkeypatch):
    _reset_emitter_globals(monkeypatch)
    stmt = If(
        BinOp("==", Id("idx"), Const(7, 3)),
        [NbAssign("r", Id("v7"))],
        [NbAssign("r", Const(99, 8))],
    )
    for i in reversed(range(7)):
        stmt = If(
            BinOp("==", Id("idx"), Const(i, 3)),
            [NbAssign("r", Id(f"v{i}"))],
            [stmt],
        )
    sigs = {"idx": Signal("idx", 3, "input"), "r": Signal("r", 8, "reg")}
    sigs.update({f"v{i}": Signal(f"v{i}", 8, "reg") for i in range(8)})
    lines: list[str] = []
    emitter._emit_nba_tree([stmt], set(sigs), {}, set(), sigs, lines, 2, set(), [0])
    body = "\n".join(lines)
    assert "switch (idx)" in body
    assert "case 7u" in body
    source = '''#include <cstdint>
int main() {
  uint8_t idx = 0, r__n = 0, r__w = 0;
  uint32_t idx_calls = 0, arm_calls = 0;
  uint8_t v0=10, v1=11, v2=12, v3=13, v4=14, v5=15, v6=16, v7=17;
  auto eval_idx = [&] { ++idx_calls; };
  auto eval_r = [&] {};
''' + "\n".join(
        f"  auto eval_v{i} = [&] {{ ++arm_calls; }};" for i in range(8)
    ) + '''
  auto run = [&] {
''' + body + '''
  };
  idx = 3; run();
  if (r__n != 13 || idx_calls != 1 || arm_calls != 1) return 1;
  idx = 9; idx_calls = arm_calls = 0; run();
  if (r__n != 99 || idx_calls != 1 || arm_calls != 0) return 2;
  return 0;
}
'''
    _compile_and_run(tmp_path, source)


def test_equality_mux_is_not_split_into_compare_chunks(monkeypatch):
    _reset_emitter_globals(monkeypatch)
    expr = Const(99, 8)
    for i in reversed(range(40)):
        expr = Ternary(BinOp("==", Id("idx"), Const(i, 6)), Id(f"r{i}"), expr)
    name = "l2_slices_0_casez"
    assigns = {name: expr}
    sigs = {"idx": Signal("idx", 6, "input"), name: Signal(name, 8, "wire")}
    sigs.update({f"r{i}": Signal(f"r{i}", 8, "reg") for i in range(40)})
    cached = {name, "idx", *(f"r{i}" for i in range(40))}
    chunks, _body = emitter._extract_deep_mux_chunks(assigns, sigs, cached, None)
    assert chunks == 0
    lines: list[str] = []
    emitter._emit_assign(name, assigns[name], cached, sigs, lines, 2, None, [0])
    body = "\n".join(lines)
    assert body.count("switch (idx)") == 1
    assert "_dmux" not in body
    assert "case 39u" in body


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
