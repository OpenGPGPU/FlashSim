"""Compare the hierarchy renamer with the general expression visitor."""

import random

from flashsim.circt_frontend import _rename_expr
from flashsim.ir import (
    ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const, Extract, Id,
    MemRead, Ternary, UnaryOp, map_expr,
)


def reference(expr, rename):
    def visit(node):
        if isinstance(node, Id) and node.name in rename:
            return Id(rename[node.name])
        if isinstance(node, MemRead):
            return MemRead(rename.get(node.mem, node.mem), node.addr)
        return node
    return map_expr(expr, visit)


def test_hierarchy_renaming_matches_general_visitor():
    rng = random.Random(0xF1A7)
    names = ("a", "b", "outside", "unchanged")
    rename = {"a": "child_a", "b": "b", "memory": "child_memory"}

    def expression(depth):
        if not depth:
            return rng.choice((Id(rng.choice(names)), Const(rng.randrange(256), 8),
                               ArrayZeros(4, 8)))
        child = lambda: expression(depth - 1)
        return rng.choice((lambda: UnaryOp("~", child()),
                           lambda: BinOp("^", child(), child()),
                           lambda: Ternary(child(), child(), child()),
                           lambda: Extract(child(), 1, 3),
                           lambda: Concat((child(), child(), child())),
                           lambda: ArrayGet(child(), child()),
                           lambda: ArrayInject(child(), child(), child()),
                           lambda: MemRead(rng.choice(("memory", "other")), child())))()

    for _ in range(1500):
        root = expression(3)
        assert _rename_expr(root, rename) == reference(root, rename)

    untouched = BinOp("+", Id("outside"), Const(1, 8))
    assert _rename_expr(untouched, rename) is untouched
    assert _rename_expr(MemRead("memory", Id("a")), rename) == \
        MemRead("child_memory", Id("child_a"))
