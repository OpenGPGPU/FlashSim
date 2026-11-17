"""Compare the hierarchy renamer with the general expression visitor."""

import random
import pytest

from flashsim.circt_frontend import _RenameMemo, _rename_expr
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


def test_repeated_ssa_leaves_share_only_within_one_instance():
    memo = _RenameMemo()
    rename = {"data": "left_data"}
    literal = Const(17, 8)
    first = BinOp("+", Id("data"), literal)
    second = BinOp("^", Id("data"), Id("data"))
    a = _rename_expr(first, rename, memo)
    b = _rename_expr(second, rename, memo)
    direct = _rename_expr(Id("data"), rename, memo)
    assert a.a is b.a is b.b is direct
    assert a.b is literal and first.a.name == "data"
    assert len(memo) == 2 and len(memo.names) == 1
    assert _rename_expr(first, {"data": "right_data"}).a == Id("right_data")
    assert _rename_expr(Id("outside"), rename).name == "outside"


def test_rename_deep_shared_graph_keeps_composite_identity_and_constant_leaves():
    root = Id("data")
    for _ in range(6000):
        root = BinOp("+", root, root)
    memo = _RenameMemo()
    mapped = _rename_expr(root, {"data": "child_data"}, memo)
    assert len(memo) == 6000 and len(memo.names) == 1
    for _ in range(6000):
        assert mapped.a is mapped.b
        mapped = mapped.a
    assert mapped == Id("child_data")


def test_rename_rejects_cycles_and_foreign_composite_identities():
    root = UnaryOp("~", Id("data"))
    memo = _RenameMemo()
    memo[id(root)] = (Id("foreign"), Const(0, 1))
    with pytest.raises(ValueError, match="foreign identity"):
        _rename_expr(root, {"data": "child_data"}, memo)
    object.__setattr__(root, "a", root)
    with pytest.raises(ValueError, match="cyclic expression"):
        _rename_expr(root, {"data": "child_data"})
