"""Expression rewrites reuse immutable branches without skipping the visitor."""

from flashsim.ir import (
    ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const, Extract, Id,
    MemRead, Ternary, UnaryOp, map_expr, replace_id,
)


def test_rewrite_visits_every_node_and_reuses_unchanged_branches():
    stable = ArrayGet(ArrayInject(ArrayZeros(4, 8), Const(1, 2), Id("a")), Id("idx"))
    moving = Concat((UnaryOp("~", Id("b")), Extract(MemRead("mem", Id("addr")), 0, 8)))
    root = Ternary(Id("enable"), stable, BinOp("|", moving, Id("target")))
    visited = []
    assert map_expr(root, lambda expr: visited.append(expr) or expr) is root
    assert Id("target") in visited and Id("addr") in visited and stable in visited

    result = replace_id(root, "target", Id("replacement"))
    assert result is not root
    assert result.cond is root.cond
    assert result.a is stable
    assert result.b.a is moving
    assert result.b.b == Id("replacement")
    assert root.b.b == Id("target")
