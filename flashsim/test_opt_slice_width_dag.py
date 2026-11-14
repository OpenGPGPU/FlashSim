"""Bit-slice width queries must not expand shared expression operands."""
import gc
import weakref

import pytest

from flashsim import ir, opt
from flashsim.ir import (
    ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const,
    Extract, Id, MemRead, Signal, Ternary, UnaryOp,
)


def test_width_of_a_trillion_occurrence_dag_visits_its_unique_nodes(monkeypatch):
    signals = {"input": Signal("input", 17, "input")}
    expr = Id("input")
    for _ in range(40):
        expr = BinOp("+", expr, expr)
    calls = 0
    original = opt.expr_width

    def bounded(node, sigs):
        nonlocal calls
        calls += 1
        assert calls < 100, "recursive width query expanded shared operands"
        return original(node, sigs)

    monkeypatch.setattr(opt, "expr_width", bounded)
    monkeypatch.setattr(ir, "expr_width", bounded)
    cache = {}
    assert opt._slice_width(expr, signals, cache) == 17
    assert len(cache) == 40 and calls == 2
    assert opt._slice_width(expr, signals, cache) == 17 and calls == 2


def test_overlapping_width_queries_reuse_the_nested_prefixes(monkeypatch):
    signals = {"a": Signal("a", 8, "input"), "b": Signal("b", 16, "input")}
    roots = [Id("a")]
    for _ in range(100):
        roots.append(BinOp("|", roots[-1], Id("b")))
    calls = 0
    original = opt.expr_width

    def bounded(node, sigs):
        nonlocal calls
        calls += 1
        assert calls <= 102, "recomputed each overlapping prefix"
        return original(node, sigs)

    monkeypatch.setattr(opt, "expr_width", bounded)
    monkeypatch.setattr(ir, "expr_width", bounded)
    cache = {}
    for root in reversed(roots[1:]):
        assert opt._slice_width(root, signals, cache) == 16


def test_recursive_width_cache_matches_the_reference_expression_contract():
    signals = {"a": Signal("a", 8, "input"), "b": Signal("b", 16, "input"),
               "array": Signal("array", 12, "mem", 8),
               "memory": Signal("memory", 7, "mem", 4)}
    a, b = Id("a"), Id("b")
    shared = BinOp("|", a, b)
    expressions = [a, b, Const(1, 3), Extract(shared, 1, 4),
                   UnaryOp("~", shared), UnaryOp("!", shared),
                   BinOp("+", a, b), BinOp("s<", a, b),
                   BinOp("<<", a, b), BinOp(">>", b, a),
                   Concat((shared, shared)), Concat(()),
                   Ternary(a, a, b), Ternary(a, b, a),
                   ArrayGet(Id("array"), a),
                   ArrayGet(ArrayZeros(4, 5), a),
                   ArrayInject(ArrayConst((1, 2), 6), a, b),
                   MemRead("memory", a)]
    cache = {}
    for expr in expressions:
        assert opt._slice_width(expr, signals, cache) == ir.expr_width(expr, signals)
    # A comparison and the unused ternary arm do not require their widths.
    assert opt._slice_width(BinOp("==", a, Id("missing")), signals, cache) == 1
    assert opt._slice_width(Ternary(a, b, Id("missing")), signals, cache) == 16
    with pytest.raises(KeyError, match="missing"):
        opt._slice_width(BinOp("+", a, Id("missing")), signals, cache)


def test_recursive_cache_limit_applies_inside_one_root(monkeypatch):
    signals = {"a": Signal("a", 3, "input")}
    expr = Id("a")
    for _ in range(100):
        expr = UnaryOp("~", expr)
    monkeypatch.setattr(opt, "_SLICE_WIDTH_CACHE_MAX", 7)
    cache = {}
    assert opt._slice_width(expr, signals, cache) == 3
    assert len(cache) <= 7 and cache[id(expr)][0] is expr


def test_width_cache_does_not_retain_signal_maps_or_input_graphs_between_passes():
    class TrackedDict(dict):
        pass

    class TrackedBinOp(BinOp):
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        def run(width):
            signals = TrackedDict(a=Signal("a", width, "input"))
            expr = TrackedBinOp("+", Id("a"), Const(1, 1))
            refs = weakref.ref(signals), weakref.ref(expr)
            assert opt._slice_width(expr, signals, {}) == width
            return refs

        assert all(ref() is None for ref in run(8))
        assert all(ref() is None for ref in run(32))
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()
