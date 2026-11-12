"""Hold-group tuning preserves register updates and sleeping/waking behavior."""
import re
import shutil
import subprocess

import pytest

from flashsim import experiment
from flashsim.benches import counter
from flashsim.compile import compile_file
from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, BinOp, Const, Id, If, Module, NbAssign, Signal


def _model():
    signals = {
        "clock": Signal("clock", 1, "input"),
        "reset": Signal("reset", 1, "input"),
        "sel": Signal("sel", 9, "input"),
        "data": Signal("data", 8, "input"),
    }
    body = []
    for i in range(280):
        reg = f"group_bank_t{i}"
        signals[reg] = Signal(reg, 8, "reg")
        body.append(If(Id("reset"), [NbAssign(reg, Const(0, 8))], [
            If(BinOp("==", Id("sel"), Const(i, 9)), [NbAssign(reg, Id("data"))])
        ]))
    return Module("HoldGroup", signals, [], AlwaysFF("clock", body),
                  ["clock", "reset", "sel", "data"])


@pytest.mark.parametrize("value", ["", "0", "-1", "not-a-number"])
def test_invalid_group_size_is_rejected(monkeypatch, value):
    monkeypatch.setenv("FLASHSIM_HOLD_GROUP", value)
    with pytest.raises(ValueError, match="FLASHSIM_HOLD_GROUP must be a positive integer"):
        emit_cpp(_model())


def test_group_sizes_match_reference_through_reset_and_wake(tmp_path, monkeypatch):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    model = _model()
    monkeypatch.delenv("FLASHSIM_HOLD_GROUP", raising=False)
    default = emit_cpp(model)
    monkeypatch.setenv("FLASHSIM_HOLD_GROUP", "24")
    assert emit_cpp(model) == default
    monkeypatch.setenv("FLASHSIM_HOLD_GROUP", "8")
    small = emit_cpp(model)
    # Ensure the fixture actually splits the same 280 writes differently.
    holds = lambda cpp: len(re.findall(r"void _nba_h\d+\(", cpp))
    assert holds(small) > holds(default) > 1
    checks = "\n".join(
        f"if (d.group_bank_t{i} != expected[{i}]) return 1;"
        for i in range(280)
    )
    harness = r'''#include "dut.h"
#include <cstdint>
int main() {
  HoldGroupDut d{};
  uint8_t expected[280] = {};
  for (unsigned t = 0; t < 1200; ++t) {
    // Hold inputs steady for several ticks, then select a different register.
    d.reset = (t == 0 || t == 617);
    d.sel = ((t / 4) * 73) % 280;
    d.data = ((t / 4) * 29 + 7) & 255;
    d.tick();
    if (d.reset) for (auto &v : expected) v = 0;
    else expected[d.sel] = d.data;
''' + checks + r'''
  }
  return 0;
}
'''
    for name, source in (("default", default), ("group8", small)):
        work = tmp_path / name
        work.mkdir()
        (work / "dut.h").write_text(source)
        (work / "check.cpp").write_text(harness)
        subprocess.run([compiler, "-std=c++17", "-O1", str(work / "check.cpp"),
                        "-o", str(work / "check")], check=True)
        subprocess.run([str(work / "check")], check=True)


def test_reuse_rejects_a_different_or_missing_group(tmp_path, monkeypatch):
    rtl = tmp_path / "fixture.v"
    header = tmp_path / "dut.h"
    rtl.write_text("// fixture\n")
    header.write_text("// generated\n")
    monkeypatch.delenv("FLASHSIM_HOLD_GROUP", raising=False)
    assert not experiment._compiled_dut_fresh(rtl, header)
    experiment._record_emit_options(header)
    assert experiment._compiled_dut_fresh(rtl, header)
    monkeypatch.setenv("FLASHSIM_HOLD_GROUP", "8")
    assert not experiment._compiled_dut_fresh(rtl, header)
    experiment._record_emit_options(header)
    assert experiment._compiled_dut_fresh(rtl, header)
    monkeypatch.delenv("FLASHSIM_HOLD_GROUP", raising=False)
    assert not experiment._compiled_dut_fresh(rtl, header)
    # Compiling a replacement outside run_bench must invalidate the record.
    monkeypatch.setenv("FLASHSIM_HOLD_GROUP", "8")
    rtl.write_text(counter())
    compile_file(rtl, header, frontend="native")
    assert not experiment._emit_options_file(header).exists()
    assert not experiment._compiled_dut_fresh(rtl, header)
