"""Pure compiler rewrites must not expand shared expression graphs into trees."""

from collections import Counter

from flashsim.circt_frontend import _flatten, _rename_expr
from flashsim.ir import (
    AlwaysFF, ArrayConst, ArrayGet, ArrayInject, ArrayZeros, Assign, BinOp, Concat,
    Const, Extract, Id, MemRead, Module, NbAssign, Signal, Ternary, UnaryOp,
    expr_id_counts, expr_ids, expr_width, map_expr, map_expr_memoized,
)
from flashsim.opt import _fold_expr


def shared_sum(depth):
    expr = Id("value")
    for _ in range(depth):
        expr = BinOp("+", expr, expr)
    return expr


def unique_binary_nodes(expr):
    seen = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, BinOp):
            stack.extend((node.a, node.b))
    return len(seen)


def test_pure_mapping_is_linear_in_shared_nodes_and_legacy_visits_are_unchanged():
    visited = []
    expr = shared_sum(40)
    result = map_expr_memoized(expr, lambda node: visited.append(node) or node)
    assert result is expr
    assert len(visited) == 41
    legacy = []
    map_expr(shared_sum(3), lambda node: legacy.append(node) or node)
    assert len(legacy) == 15


def test_renaming_preserves_sharing_and_independent_instance_names():
    expr = shared_sum(18)
    left = _rename_expr(expr, {"value": "left_value"})
    right = _rename_expr(expr, {"value": "right_value"})
    assert unique_binary_nodes(left) == unique_binary_nodes(right) == 19
    for _ in range(18):
        assert left.a is left.b and right.a is right.b
        left, right = left.a, right.a
    assert left == Id("left_value") and right == Id("right_value")


def test_folding_preserves_shared_graph_and_constant_values():
    expr = shared_sum(18)
    folded = _fold_expr(expr)
    assert unique_binary_nodes(folded) == 19
    constant = map_expr_memoized(
        expr, lambda node: Const(3, 64) if isinstance(node, Id) else node)
    assert _fold_expr(constant) == Const(3 << 18, 64)


def test_deep_pure_mapping_and_folding_do_not_require_recursive_walks():
    expr = Id("value")
    for _ in range(6000):
        expr = UnaryOp("~", expr)
    assert map_expr_memoized(expr, lambda node: node) is expr
    result = _fold_expr(expr)
    for _ in range(6000):
        assert isinstance(result, UnaryOp)
        result = result.a
    assert result == Id("value")


def test_all_expression_kinds_match_per_occurrence_mapping_values():
    array = ArrayGet(ArrayInject(ArrayZeros(4, 8), Const(1, 2), Id("value")), Id("idx"))
    root = Ternary(Id("enable"), array,
                   Concat((UnaryOp("~", Id("value")), Extract(MemRead("mem", Id("addr")), 0, 8))))

    def visitor(node):
        if isinstance(node, Id):
            return Id("instance_" + node.name)
        if isinstance(node, MemRead):
            return MemRead("instance_" + node.mem, node.addr)
        return node

    assert map_expr_memoized(root, visitor) == map_expr(root, visitor)
    assert list(expr_ids(root)) == ["addr", "enable", "idx", "mem", "value"]


def test_flatten_shares_rewrites_across_roots_but_not_between_instances():
    expr = shared_sum(8)
    child = Module("Leaf", {
        "value": Signal("value", 64, "input"),
        "a": Signal("a", 64, "output"),
        "b": Signal("b", 64, "output"),
        "state": Signal("state", 64, "reg"),
    }, [Assign("a", expr), Assign("b", expr)],
        AlwaysFF("clock", [NbAssign("state", expr)]), ["value", "a", "b"])
    signals = {name: Signal(name, 64, "input" if name in {"left", "right"} else "output")
               for name in ("left", "right", "a", "b", "c", "d")}
    parent = Module("Top", signals, [], AlwaysFF("clock", []), list(signals))
    parent.instances = [(["a", "b"], "one", "Leaf", {"value": "left"}),
                        (["c", "d"], "two", "Leaf", {"value": "right"})]
    result = _flatten(parent, {"Leaf": child}, Counter(Leaf=2))
    assigns = {a.lhs: a.rhs for a in result.assigns}
    assert assigns["a"] is assigns["b"] is result.always.body[0].rhs
    assert assigns["c"] is assigns["d"] is result.always.body[1].rhs
    assert assigns["a"] is not assigns["c"]
    left, right = assigns["a"], assigns["c"]
    for _ in range(8):
        left, right = left.a, right.a
    assert left == Id("left") and right == Id("right")


def test_shared_graph_queries_preserve_occurrence_counts_without_expansion():
    expr = shared_sum(1000)
    assert expr_ids(expr) == {"value"}
    assert expr_id_counts(expr) == {"value": 1 << 1000}
    assert expr_width(expr, {"value": Signal("value", 64, "input")}) == 64
    read = MemRead("memory", Id("address"))
    expr = Concat((read, read, Id("address")))
    assert list(expr_ids(expr)) == ["address", "memory"]
    assert expr_id_counts(expr) == {"memory": 2, "address": 3}


def test_width_only_resolves_operands_that_determine_result_width():
    sigs = {"value": Signal("value", 8, "input")}
    assert expr_width(Ternary(Id("unknown"), Id("value"), Id("missing")), sigs) == 8
    assert expr_width(BinOp("<<", Id("value"), Id("unknown")), sigs) == 8
    assert expr_width(Extract(Id("unknown"), 0, 3), sigs) == 3
    assert expr_width(BinOp("==", Id("unknown"), Id("missing")), sigs) == 1


def test_aggregate_constants_remain_opaque_to_graph_rewrites_and_queries():
    values = ArrayConst((1, 2, 255, 0), 8)
    assert map_expr_memoized(values, lambda node: node) is values
    assert _rename_expr(values, {"value": "instance_value"}) is values
    assert _fold_expr(values) is values
    assert expr_ids(values) == set()
    assert expr_id_counts(values) == {}
    assert expr_width(values, {}) == 8
