"""Constant folding must preserve shared IR instead of expanding its tree."""
import gc
import sys
import weakref

from flashsim import ir, opt
from flashsim.ir import (
    ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const,
    Extract, Id, MemRead, Ternary, UnaryOp,
)


def test_fold_work_is_bounded_by_unique_nodes_for_a_large_shared_dag():
    expr = Id("input")
    for _ in range(40):
        expr = BinOp("+", expr, expr)
    calls = 0

    def bound(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code.co_filename in {opt.__file__, ir.__file__}:
            calls += 1
            assert calls < 500, "fold expanded a 41-node DAG into its trillion-node tree"

    previous = sys.getprofile()
    try:
        sys.setprofile(bound)
        folded = opt._fold_expr(expr)
    finally:
        sys.setprofile(previous)
    assert folded is expr


def test_unchanged_arrays_and_memory_children_keep_their_sharing():
    shared = BinOp("+", Id("a"), Id("b"))
    expressions = [UnaryOp("~", shared), Ternary(Id("select"), shared, shared),
                   Extract(shared, 1, 3), Concat((shared, shared)),
                   ArrayGet(ArrayZeros(4, 8), shared),
                   ArrayInject(ArrayConst((1, 2, 3, 4), 8), shared, shared),
                   MemRead("memory", shared)]
    for expr in expressions:
        assert opt._fold_expr(expr) is expr


def test_folded_descendants_are_published_once_to_all_operand_kinds():
    seven = BinOp("+", Const(3, 8), Const(4, 8))
    shared = BinOp("+", Id("input"), seven)
    expr = Concat((shared, shared, UnaryOp("~", shared),
                   Ternary(Id("select"), shared, shared), Extract(shared, 1, 3),
                   ArrayGet(ArrayZeros(4, 8), seven),
                   ArrayInject(ArrayConst((1, 2, 3, 4), 8), seven, shared),
                   MemRead("memory", shared)))
    folded = opt._fold_expr(expr)
    result = folded.parts[0]
    assert result == BinOp("+", Id("input"), Const(7, 8))
    assert folded.parts[1] is result
    assert folded.parts[2].a is result
    assert folded.parts[3].a is result and folded.parts[3].b is result
    assert folded.parts[4].a is result
    assert folded.parts[5].index is result.b
    assert folded.parts[6].index is result.b and folded.parts[6].value is result
    assert folded.parts[7].addr is result
    assert shared.b is seven  # The original immutable input was not rewritten.


def test_folding_preserves_constant_widths_branch_selection_and_absorption():
    assert opt._fold_expr(UnaryOp("~", Const(0x55, 8))) == Const(0xaa, 8)
    assert opt._fold_expr(BinOp("+", Const(0xff, 8), Const(1, 8))) == Const(0, 8)
    assert opt._fold_expr(BinOp("s<", Const(0xff, 8), Const(1, 8))) == Const(1, 1)
    assert opt._fold_expr(Extract(Const(0xabcd, 16), 4, 8)) == Const(0xbc, 8)
    branch = Id("branch")
    assert opt._fold_expr(Ternary(Const(0, 1), Id("unused"), branch)) is branch
    assert opt._fold_expr(BinOp("&", branch, BinOp("|", branch, Id("other")))) is branch


def test_fold_memo_releases_its_inputs_without_cyclic_collection():
    class TrackedId(Id):
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        def run(name):
            leaf = TrackedId(name)
            reference = weakref.ref(leaf)
            expr = BinOp("+", leaf, BinOp("+", Const(1, 8), Const(2, 8)))
            folded = opt._fold_expr(expr)
            assert folded.a is leaf and folded.b == Const(3, 8)
            return reference

        assert run("first")() is None
        assert run("second")() is None
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()
