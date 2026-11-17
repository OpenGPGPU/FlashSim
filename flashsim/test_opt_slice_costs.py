"""Slice cost metadata stays exact and bounded on overlapping expression DAGs."""
import gc
import sys
import weakref

import pytest

from flashsim import opt
from flashsim.ir import (
    ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp, Concat, Const,
    Expr, Extract, Id, MemRead, MemWrite, NbAssign, Signal, Ternary, UnaryOp,
    expr_id_counts, expr_ids,
)
from flashsim.test_opt_generated_slices import circuit


def test_shared_dag_costs_reuse_overlapping_roots_without_expanding_occurrences(monkeypatch):
    roots = [Id("a")]
    for _ in range(40):
        roots.append(BinOp("&", roots[-1], roots[-1]))
    original = opt._expr_children
    visits = 0

    def children(expr):
        nonlocal visits
        visits += 1
        assert visits <= 200, "revisited overlapping DAG roots"
        return original(expr)

    monkeypatch.setattr(opt, "_expr_children", children)
    costs = opt._SliceCosts()
    for depth in reversed(range(1, 41)):
        assert costs.node_count(roots[depth]) == (1 << (depth + 1)) - 1
        assert costs.leaf_set(roots[depth]) == {"a"}
    assert visits <= 170
    # Leaf-set metadata must not change multiplicity for the use counter.
    assert expr_id_counts(roots[10]) == {"a": 1024}


def test_leaf_sets_match_every_expression_contract_including_memory_names():
    a, b = Id("a"), Id("b")
    expressions = [a, Const(1), ArrayZeros(4, 8), ArrayConst((1, 2), 8),
                   UnaryOp("~", a), BinOp("+", a, b), Ternary(Id("c"), a, b),
                   Extract(a, 1, 2), Concat((a, a, b)), Concat(()),
                   ArrayGet(Id("array"), a), ArrayInject(Id("array"), a, b),
                   MemRead("memory", BinOp("+", a, b))]
    costs = opt._SliceCosts()
    for expr in expressions:
        assert costs.leaf_set(expr) == expr_ids(expr)
        assert costs.node_count(expr) == opt._expr_nodes(expr)
    with pytest.raises(TypeError):
        costs.leaf_set(Expr())


def test_inline_threshold_matches_exact_occurrence_counts_with_and_without_cache():
    a, b = Id("a"), Id("b")
    roots = [a, Const(1), ArrayZeros(4, 8), ArrayConst((1, 2), 8),
             UnaryOp("~", a), BinOp("+", a, b), Ternary(Id("c"), a, b),
             Extract(a, 1, 2), Concat((a, a, b)), Concat(()),
             ArrayGet(Id("array"), a), ArrayInject(Id("array"), a, b),
             MemRead("memory", BinOp("+", a, b))]
    for _ in range(8):
        roots.append(BinOp("&", roots[-1], roots[-1]))
    for cached in (False, True):
        costs = opt._SliceCosts()
        if cached:
            for root in roots:
                costs.node_count(root)
        for root in roots:
            for copies in (1, 2, 5, 16, 17):
                assert costs.fits_inline(*([root] * copies)) == (
                    copies * opt._expr_nodes(root) <= opt._SLICE_INLINE_NODES)


def test_inline_threshold_bounds_wide_and_deep_walks_without_retaining_metadata(monkeypatch):
    consumed = []

    class LazyParts(tuple):
        def __iter__(self):
            for part in super().__iter__():
                consumed.append(part)
                assert len(consumed) <= opt._SLICE_INLINE_NODES
                yield part

    wide = Concat(LazyParts(Id("a") for _ in range(32768)))
    deep = Id("a")
    for _ in range(6000):
        deep = UnaryOp("~", deep)
    visits = []
    original = opt._expr_children

    def children(expr):
        visits.append(expr)
        assert len(visits) <= opt._SLICE_INLINE_NODES
        return original(expr)

    monkeypatch.setattr(opt, "_expr_children", children)
    for expr in (wide, deep):
        visits.clear()
        costs = opt._SliceCosts()
        assert not costs.fits_inline(expr)
        assert not costs.nodes and not costs.leaves


def test_leaf_sets_keep_the_iterative_reference_walks_depth_tolerance():
    expr = Id("a")
    for _ in range(2000):
        expr = UnaryOp("~", expr)
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        assert opt._SliceCosts().leaf_set(expr) == expr_ids(expr) == {"a"}
    finally:
        sys.setrecursionlimit(previous)


def test_metadata_has_entry_and_aggregate_leaf_name_budgets(monkeypatch):
    monkeypatch.setattr(opt, "_SLICE_COST_CACHE_MAX", 3)
    monkeypatch.setattr(opt, "_SLICE_LEAF_NAMES_MAX", 5)
    monkeypatch.setattr(opt, "_SLICE_LEAF_ENTRY_MAX", 3)
    costs = opt._SliceCosts()
    for index in range(30):
        expr = BinOp("|", Id(f"a{index}"), Id(f"b{index}"))
        assert costs.node_count(expr) == 3
        assert costs.leaf_set(expr) == {f"a{index}", f"b{index}"}
        assert len(costs.nodes) <= 3 and len(costs.leaves) <= 3
        assert costs.leaf_names == sum(len(entry[1]) for entry in costs.leaves.values()) <= 5
        assert all(len(entry[1]) <= 3 for entry in costs.leaves.values())
    large = Concat(tuple(Id(f"x{index}") for index in range(10)))
    assert costs.leaf_set(large) == {f"x{index}" for index in range(10)}
    assert id(large) not in costs.leaves


def test_cache_retains_source_identity_and_releases_inputs_without_cyclic_gc():
    class TrackedBinOp(BinOp):
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        def run():
            expr = TrackedBinOp("&", Id("a"), Id("b"))
            costs = opt._SliceCosts()
            ref = weakref.ref(expr)
            costs.node_count(expr)
            costs.leaf_set(expr)
            assert costs.nodes[id(expr)][0] is expr
            assert costs.leaves[id(expr)][0] is expr
            del expr
            assert ref() is not None
            return ref

        assert run()() is None
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()


def test_metadata_cache_preserves_slice_wires_and_sequential_memory_effects(monkeypatch):
    def run():
        signals, wide = circuit()
        signals.update(r=Signal("r", 1, "reg"), mem=Signal("mem", 1, "mem", 2))
        bit = Extract(Id("wide"), 1, 1)
        assigns = {"wide": wide, "first": BinOp("&", bit, Id("gate")), "second": bit}
        result = opt._narrow_bit_slices(
            assigns, [NbAssign("r", bit)],
            [MemWrite("mem", Const(0, 1), bit, Const(1, 1))], signals)
        return result, signals

    cached = run()
    monkeypatch.setattr(opt._SliceCosts, "node_count", lambda self, expr: opt._expr_nodes(expr))
    monkeypatch.setattr(opt._SliceCosts, "leaf_set", lambda self, expr: frozenset(expr_ids(expr)))
    monkeypatch.setattr(opt._SliceCosts, "fits_inline", lambda self, *exprs:
                        sum(opt._expr_nodes(expr) for expr in exprs) <= opt._SLICE_INLINE_NODES)
    assert run() == cached
