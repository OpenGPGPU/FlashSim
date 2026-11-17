"""Repeated width queries reuse prefixes without losing depth or width rules."""
import gc
import weakref

import pytest

from flashsim import ir, opt
from flashsim.ir import (
    ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const,
    Extract, Id, MemRead, Signal, Ternary, UnaryOp,
)


def test_overlapping_queries_do_not_revisit_the_same_prefixes():
    class CountReads(dict):
        reads = 0

        def __getitem__(self, name):
            self.reads += 1
            return super().__getitem__(name)

    signals = CountReads(a=Signal("a", 8, "input"), b=Signal("b", 16, "input"))
    roots, b = [Id("a")], Id("b")
    for _ in range(1000):
        roots.append(BinOp("|", roots[-1], b))
    cache = {}
    for root in reversed(roots[1:]):
        assert opt._slice_width(root, signals, cache) == 16
        assert signals.reads <= 2, "recomputed a previously queried prefix"


def test_deep_width_queries_remain_iterative_with_a_tiny_cache(monkeypatch):
    signals = {"a": Signal("a", 8, "input")}
    expr = Id("a")
    for _ in range(10000):
        expr = UnaryOp("~", expr)
    monkeypatch.setattr(opt, "_SLICE_WIDTH_CACHE_MAX", 7)
    cache = {}
    assert opt._slice_width(expr, signals, cache) == 8
    assert len(cache) <= 7 and cache[id(expr)][0] is expr


def test_shared_diamond_width_retains_occurrence_semantics():
    expr = Const(1, 1)
    for _ in range(40):
        expr = Concat((expr, expr))
    assert opt._slice_width(expr, {}, {}) == 2**40


def test_cached_width_matches_the_reference_contract():
    signals = {"a": Signal("a", 8, "input"), "b": Signal("b", 16, "input"),
               "array": Signal("array", 12, "mem", 8),
               "memory": Signal("memory", 7, "mem", 4)}
    a, b, missing = Id("a"), Id("b"), Id("missing")
    shared = BinOp("|", a, b)
    expressions = [a, b, Const(1, 3), Extract(shared, 1, 4),
                   UnaryOp("~", shared), UnaryOp("!", shared),
                   BinOp("+", a, b), BinOp("s<", a, b),
                   BinOp("<<", a, b), BinOp(">>", b, missing),
                   BinOp("==", a, missing), Concat((shared, shared)), Concat(()),
                   Ternary(missing, a, b), Ternary(a, b, missing),
                   ArrayGet(Id("array"), missing),
                   ArrayGet(ArrayZeros(4, 5), a),
                   ArrayInject(ArrayConst((1, 2), 6), missing, b),
                   MemRead("memory", missing)]
    cache = {}
    for expr in expressions:
        assert opt._slice_width(expr, signals, cache) == ir.expr_width(expr, signals)
    with pytest.raises(KeyError, match="missing"):
        opt._slice_width(BinOp("+", shared, missing), signals, cache)
    with pytest.raises(TypeError):
        opt._slice_width(object(), signals, cache)
    signals["missing"] = Signal("missing", 32, "wire")
    assert opt._slice_width(BinOp("+", shared, missing), signals, cache) == 32


def test_width_cache_releases_graphs_and_signal_tables_at_pass_end():
    class Signals(dict):
        pass

    class Expression(BinOp):
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        def run(width):
            signals = Signals(a=Signal("a", width, "input"))
            expr = Expression("+", Id("a"), Const(1, 1))
            refs = weakref.ref(signals), weakref.ref(expr)
            assert opt._slice_width(expr, signals, {}) == width
            return refs

        assert all(ref() is None for ref in run(8))
        assert all(ref() is None for ref in run(32))
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()
