"""Flatten reused nested modules without losing their independent state."""

import shutil
import subprocess
from collections import Counter

import pytest

from flashsim.circt_frontend import (
    _flatten, _lower_module, _reachable_instances, _split_modules, parse_hw_mlir,
)
from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Assign, Const, Id, Module, NbAssign, Signal
from flashsim.opt import optimize


def test_completed_child_is_released_after_its_last_instance():
    src = """
    hw.module private @Leaf(out q : i1) {
      %v = hw.constant true
      hw.output %v : i1
    }
    hw.module @Top(out x : i1, out y : i1) {
      %a = hw.instance "one" @Leaf() -> (q: i1)
      %b = hw.instance "two" @Leaf() -> (q: i1)
      hw.output %a, %b : i1, i1
    }
    """
    modules = {raw.name: _lower_module(raw) for raw in _split_modules(src)}
    remaining, reachable = _reachable_instances(modules["Top"], modules)
    assert reachable == {"Top", "Leaf"}
    assert remaining["Leaf"] == 2
    result = _flatten(modules["Top"], modules, remaining)
    assert set(modules) == {"Top"}
    assert remaining["Leaf"] == 0
    assert {a.lhs for a in result.assigns} >= {"x", "y"}


def test_reused_hierarchy_keeps_every_instance_and_register(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module private @Leaf(in %clock : i1, in %d : i8, out q : i8) {
      %c = seq.to_clock %clock
      %value = seq.firreg %d clock %c : i8
      hw.output %value : i8
    }
    hw.module private @Pair(in %clock : i1, in %a : i8, in %b : i8, out x : i8, out y : i8) {
      %p = hw.instance "first" @Leaf(clock: %clock: i1, d: %a: i8) -> (q: i8)
      %q = hw.instance "second" @Leaf(clock: %clock: i1, d: %b: i8) -> (q: i8)
      hw.output %p, %q : i8, i8
    }
    hw.module @Top(in %clock : i1, in %a : i8, in %b : i8, in %c : i8, out x : i8, out y : i8, out z : i8, out u : i8, out v : i8) {
      %p, %q = hw.instance "pair" @Pair(clock: %clock: i1, a: %a: i8, b: %b: i8) -> (x: i8, y: i8)
      %r = hw.instance "extra" @Leaf(clock: %clock: i1, d: %c: i8) -> (q: i8)
      %s, %t = hw.instance "other" @Pair(clock: %clock: i1, a: %b: i8, b: %c: i8) -> (x: i8, y: i8)
      hw.output %p, %q, %r, %s, %t : i8, i8, i8, i8, i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="Top"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  TopDut d;
  d.a = 11; d.b = 22; d.c = 33; d.tick();
  d.eval_x(); d.eval_y(); d.eval_z(); d.eval_u(); d.eval_v();
  if (d.x != 11 || d.y != 22 || d.z != 33 || d.u != 22 || d.v != 33) return 1;
  d.a = 44; d.b = 55; d.c = 66; d.tick();
  d.eval_x(); d.eval_y(); d.eval_z(); d.eval_u(); d.eval_v();
  return d.x != 44 || d.y != 55 || d.z != 66 || d.u != 55 || d.v != 66;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_private_child_names_remain_distinct_from_parent_and_instance_ports(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    parent_signals = {name: Signal(name, width, kind) for name, width, kind in [
        ("clock", 1, "input"), ("data", 8, "input"), ("out", 8, "output"),
        ("one_x", 8, "wire"), ("one_x__flashsim_local", 8, "wire")]}
    parent = Module("Top", parent_signals,
        [Assign("one_x", Const(55, 8)), Assign("one_x__flashsim_local", Const(66, 8))],
        AlwaysFF("clock", []), ["clock", "data", "out"])
    parent.instances = [(["out"], "one", "Leaf", {"clock": "clock", "d": "data"})]
    leaf_signals = {name: Signal(name, width, kind) for name, width, kind in [
        ("clock", 1, "input"), ("d", 8, "input"), ("q", 8, "output"),
        ("x", 8, "reg")]}
    leaf = Module("Leaf", leaf_signals, [Assign("q", Id("x"))],
                  AlwaysFF("clock", [NbAssign("x", Id("d"))]), ["clock", "d", "q"])
    result = _flatten(parent, {"Leaf": leaf}, Counter(Leaf=1))
    registers = [name for name, signal in result.signals.items() if signal.kind == "reg"]
    assert len(registers) == 1
    assert registers[0] not in parent_signals
    (tmp_path / "dut.h").write_text(emit_cpp(optimize(result)))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  TopDut d;
  d.data = 42; d.tick(); d.eval_out(); if (d.out != 42) return 1;
  d.data = 81; d.tick(); d.eval_out(); return d.out != 81;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
