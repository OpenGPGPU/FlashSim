"""Bit slicing retains shared cones and publishes every generated slice wire."""

import shutil
import subprocess

import pytest

from flashsim import ir, opt
from flashsim.emit import emit_cpp


def shared_sum(expr, depth):
    for _ in range(depth):
        expr = ir.BinOp("+", expr, expr)
    return expr


def test_shared_extract_is_narrowed_once(monkeypatch):
    root = shared_sum(ir.Extract(ir.Id("value"), 0, 8), 12)
    original = opt._slice_src
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(opt, "_slice_src", counted)
    result = opt._narrow_expr(root, {}, {"value": ir.Signal("value", 64, "input")},
                              {}, {}, {}, {})
    assert result is root
    assert calls == 1


def test_deep_narrowing_does_not_require_recursive_mapping():
    root = ir.Id("value")
    for _ in range(6000):
        root = ir.UnaryOp("~", root)
    assert opt._narrow_expr(root, {}, {"value": ir.Signal("value", 64, "input")},
                            {}, {}, {}, {}) is root


def slice_case(width):
    low = ir.Concat(tuple(ir.Id(f"lo{i}") for i in range(width)))
    high = ir.Concat(tuple(ir.Id(f"hi{i}") for i in range(width)))
    extract = ir.Extract(ir.Id("packed"), 0, width)
    assigns = {"packed": ir.Concat((high, low)),
               "out": shared_sum(extract, 5), "other": extract}
    signals = {name: ir.Signal(name, 1, "input") for name in
               ["clock", *(f"lo{i}" for i in range(width)),
                *(f"hi{i}" for i in range(width))]}
    signals.update(packed=ir.Signal("packed", width * 2, "wire"),
                   out=ir.Signal("out", width, "output"),
                   other=ir.Signal("other", width, "output"))
    return assigns, signals, low


@pytest.mark.parametrize("width", [3, 20])
def test_narrowed_shared_values_and_slice_definitions_match_legacy(
        width, monkeypatch, tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    assigns, base_signals, low = slice_case(width)
    signals = dict(base_signals)
    narrowed, body, writes = opt._narrow_bit_slices(assigns, [], [], signals)
    legacy_signals = dict(base_signals)
    with monkeypatch.context() as context:
        context.setattr(opt, "map_expr_memoized", lambda expr, fn: ir.map_expr(expr, fn))
        legacy = opt._narrow_bit_slices(assigns, [], [], legacy_signals)
    assert (narrowed, body, writes) == legacy
    assert signals == legacy_signals
    fresh = set(narrowed) - set(assigns)
    if width == 20:
        assert fresh == {"packed__b0_20"}
        assert narrowed["packed__b0_20"] == low
        assert signals["packed__b0_20"].width == width
    else:
        assert not fresh
    root = narrowed["out"]
    for _ in range(5):
        assert root.a is root.b
        root = root.a
    assert root == narrowed["other"]

    ports = [name for name, signal in signals.items() if signal.kind in {"input", "output"}]
    model = ir.Module("Narrow", signals,
                      [ir.Assign(name, expr) for name, expr in narrowed.items()],
                      ir.AlwaysFF("clock", []), ports)
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    # Concat's first operand is the most significant bit; independently check
    # the selected half and the shared sum over several actual input changes.
    drive = "\n".join(
        f"d.lo{i} = (pattern >> {width - 1 - i}) & 1; "
        f"d.hi{i} = ((pattern ^ mask) >> {width - 1 - i}) & 1;"
        for i in range(width))
    (tmp_path / "check.cpp").write_text('''#include "dut.h"
#include <initializer_list>
int main() {
  NarrowDut d;
  constexpr unsigned mask = ''' + str((1 << width) - 1) + ''';
  for (unsigned pattern : {0u, 1u, 0xaaaaau, 0xfffffu, 0x5a39bu}) {
    ''' + drive + '''
    d.poke_inputs(); d.eval_out(); d.eval_other();
    if (d.out != ((pattern << 5) & mask) || d.other != (pattern & mask)) return 1;
  }
  return 0;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(tmp_path / "check.cpp"),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_slice_results_do_not_cross_mutable_root_contexts():
    assigns, signals, _ = slice_case(20)
    root = assigns["out"]
    first_fresh, second_fresh = {}, {}
    first = opt._narrow_expr(root, assigns, dict(signals), {"packed": 2},
                            first_fresh, {}, {})
    occupied = dict(signals)
    occupied["packed__b0_20"] = ir.Signal("packed__b0_20", 20, "wire")
    second = opt._narrow_expr(root, assigns, occupied, {"packed": 2},
                             second_fresh, {}, {})
    for _ in range(5):
        first, second = first.a, second.a
    assert first == ir.Id("packed__b0_20")
    assert second == ir.Id("packed__b0_20_1")
    assert set(first_fresh) == {first.name}
    assert set(second_fresh) == {second.name}
