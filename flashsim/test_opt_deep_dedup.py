"""Optional wire deduplication must survive very deep expression trees."""

import sys

from flashsim.ir import BinOp, Const, Id, Signal, UnaryOp
from flashsim.opt import _dedup_assigns, _expr_nodes


def nested_not(depth):
    expr = Id("input")
    for _ in range(depth):
        expr = UnaryOp("~", expr)
    return expr


def inputs(deep):
    expr = BinOp("&", Id("input"), Const(1, 1))
    assigns = {"deep": deep, "first": expr, "second": expr, "output": Id("second")}
    sigs = {name: Signal(name, 1, kind) for name, kind in
            (("input", "input"), ("deep", "wire"), ("first", "wire"),
             ("second", "wire"), ("output", "output"))}
    return assigns, sigs


def test_deep_unhashable_wire_survives_while_other_wires_deduplicate(monkeypatch):
    deep = nested_not(300)
    assigns, sigs = inputs(deep)
    original_hash = UnaryOp.__hash__

    def overflowing_hash(expr):
        if expr is deep:
            raise RecursionError("simulated deep dataclass hash overflow")
        return original_hash(expr)

    monkeypatch.setattr(UnaryOp, "__hash__", overflowing_hash)
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        assert _expr_nodes(deep) == 301
        result = _dedup_assigns(assigns, [], [], [], [], [], sigs)[0]
    finally:
        sys.setrecursionlimit(limit)
    assert result["deep"] is deep
    assert "second" not in result and result["output"] == Id("first")


def test_alias_rewrite_overflow_leaves_every_input_unchanged():
    deep = nested_not(700)
    assigns, sigs = inputs(deep)
    body, reset, writes, assertions, clocks = [], [], [], [], []
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        result = _dedup_assigns(assigns, body, reset, writes, assertions, clocks, sigs)
    finally:
        sys.setrecursionlimit(limit)
    assert all(a is b for a, b in zip(result, (assigns, body, reset, writes, assertions, clocks)))
    assert "second" in assigns and assigns["output"] == Id("second")
