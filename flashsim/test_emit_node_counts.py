"""Emitter cost metadata preserves expanded counts without expanding shared DAGs."""
import sys
import pytest
from flashsim import emit
from flashsim.ir import (ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp,
                         Concat, Const, Extract, Id, MemRead, Ternary, UnaryOp)


def shared(depth):
    expr = Id("input")
    for _ in range(depth):
        expr = BinOp("+", expr, expr)
    return expr


def test_count_shared_dag_has_bounded_operand_visits(monkeypatch):
    expr = shared(40)
    original = BinOp.__getattribute__
    reads = 0

    class WorkBudgetExceeded(Exception):
        pass

    def access(node, name):
        nonlocal reads
        if name in {"a", "b"}:
            reads += 1
            if reads > 4096:
                raise WorkBudgetExceeded
        return original(node, name)

    monkeypatch.setattr(BinOp, "__getattribute__", access)
    try:
        count = emit._expr_node_count(expr)
    except WorkBudgetExceeded:
        pytest.fail("emitter counting exceeded 4096 operand reads on 41 nodes", pytrace=False)
    assert count == (1 << 41) - 1
    assert reads <= 80


def test_deep_counts_do_not_require_recursive_python_frames():
    expr = Id("input")
    for _ in range(6000):
        expr = UnaryOp("~", expr)
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        assert emit._expr_node_count(expr) == 6001
    finally:
        sys.setrecursionlimit(limit)


def test_counts_preserve_all_operand_kinds_and_opaque_constants():
    a = Id("a")
    cases = [(a, 1), (Const(3, 8), 1), (UnaryOp("~", a), 2),
             (BinOp("+", a, a), 3), (Ternary(a, a, a), 4),
             (Extract(a, 1, 2), 2), (Concat((a, a)), 3), (Concat(()), 1),
             (ArrayGet(a, a), 3), (ArrayInject(a, a, a), 4),
             (ArrayZeros(8, 32), 1), (ArrayConst(tuple(range(10000)), 32), 1),
             (MemRead("memory", a), 2)]
    for expr, count in cases:
        assert emit._expr_node_count(expr) == count


def test_gpu_promotion_uses_expanded_occurrences():
    name = "computeUnits_0_core_t3"
    assert emit._is_gpu_hot_ssa(name)
    cached = set()
    emit._promote_large_gpu_ssa({name: shared(5)}, cached)
    assert cached == {name}  # 63 occurrences cross the 48-node promotion threshold.
