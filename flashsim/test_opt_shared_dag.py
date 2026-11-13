"""Wire deduplication must traverse shared IR without expanding its tree."""
import gc
import weakref

from flashsim import opt
from flashsim.ir import (
    ArrayConst, ArrayGet, ArrayInject, ArrayZeros, BinOp, ClockedAssertion,
    ClockedPrint, ClockedProcess, Concat, Const, DecimalFormat, Extract, Id,
    If, MemRead, MemWrite, NbAssign, Signal, Ternary, UnaryOp, map_expr,
)


def diamond(depth, leaf=None):
    expr = Id("input") if leaf is None else leaf
    for _ in range(depth):
        expr = BinOp("&", expr, expr)
    return expr


def test_expanded_node_count_keeps_exact_threshold_semantics_with_bounded_work(monkeypatch):
    original = opt._expr_nodes
    calls = 0

    def budget(expr):
        nonlocal calls
        calls += 1
        assert calls < 5000, "expanded a shared subtree instead of counting it once"
        return original(expr)

    monkeypatch.setattr(opt, "_expr_nodes", budget)
    assert budget(diamond(40)) == (1 << 41) - 1


def test_structural_keys_match_dataclass_equality_for_every_expression_kind():
    a, b = Id("a"), Id("b")
    expressions = [a, b, Const(1, 1), Const(1, 2), Const(2, 2),
                   UnaryOp("~", a), UnaryOp("!", a),
                   BinOp("&", a, b), BinOp("|", a, b), BinOp("&", b, a),
                   Ternary(a, b, Const(0)), Extract(a, 0, 1), Extract(a, 1, 1),
                   Concat((a, b)), Concat((b, a)), Concat((a,)),
                   ArrayGet(Id("array"), a), ArrayGet(Id("other"), a),
                   ArrayInject(Id("array"), a, b), ArrayInject(Id("array"), b, a),
                   ArrayZeros(4, 8), ArrayZeros(8, 4),
                   ArrayConst((1, 2), 8), ArrayConst((2, 1), 8), ArrayConst((1, 2), 16),
                   MemRead("mem", a), MemRead("other", a), MemRead("mem", b)]
    # Separate allocations with the same structure must compare equal too.
    copies = [map_expr(expr, lambda e: Id(e.name) if isinstance(e, Id) else e)
              for expr in expressions]
    interner = opt._ExprInterner()
    for a in expressions + copies:
        for b in expressions + copies:
            assert (interner.key(a) == interner.key(b)) == (a == b)


def test_dedup_shared_dags_never_recursively_hash_expressions(monkeypatch):
    def forbidden_hash(expr):
        raise AssertionError("recursive Expr hashing expands the DAG")

    monkeypatch.setattr(BinOp, "__hash__", forbidden_hash)
    assigns = {"z": diamond(40), "a": diamond(40), "different": diamond(40, Id("other")),
               "out": Id("z")}
    sigs = {n: Signal(n, 1, "wire") for n in assigns}
    sigs.update(input=Signal("input", 1, "input"), other=Signal("other", 1, "input"))
    sigs["out"] = Signal("out", 1, "output")
    result = opt._dedup_assigns(assigns, [], [], [], [], [], sigs)[0]
    assert set(result) == {"a", "different", "out"}
    assert result["a"] is assigns["a"]
    assert result["different"] is assigns["different"]
    assert result["out"] == Id("a")


def test_explicit_pure_mapping_visits_each_node_once_and_default_visits_occurrences():
    shared = UnaryOp("~", Id("a"))
    expr = BinOp("&", shared, shared)
    seen = []
    assert map_expr(expr, lambda node: seen.append(node) or node) is expr
    assert len(seen) == 5
    seen.clear()
    memo = {}

    def rewrite(node):
        seen.append(node)
        return Id("b") if isinstance(node, Id) else node

    result = map_expr(expr, rewrite, memo=memo)
    assert len(seen) == len(memo) == 3
    assert result.a is result.b and result.a.a == Id("b")
    assert expr.a is shared and shared.a == Id("a")
    assert memo[id(expr)] == (expr, result)


def test_alias_rewrite_preserves_all_observers_effects_and_nonwire_definitions():
    same = BinOp("^", Id("input"), Const(1, 1))
    assigns = {"z": same, "a": same, "out": Id("z"), "reg": same, "array": same}
    sigs = {n: Signal(n, 1, "wire") for n in assigns}
    sigs["input"] = Signal("input", 1, "input")
    sigs["out"].kind = "output"
    sigs["reg"].kind = "reg"
    sigs["array"].depth = 4
    dag = diamond(40, Id("z"))
    body = [If(Id("z"), [NbAssign("reg", dag)], [])]
    reset = [NbAssign("reg", Id("z"))]
    writes = [MemWrite("mem", Id("z"), dag, Id("z"))]
    assertions = [ClockedAssertion(Id("z"), Id("z"), "message", "scope"),
                  ClockedPrint(Id("z"), Id("z"), ("v=", DecimalFormat(Id("z"), 1, 4)),
                               "scope", "stdout")]
    clocks = [ClockedProcess(Id("z"), "scope")]
    result, body, reset, writes, assertions, clocks = opt._dedup_assigns(
        assigns, body, reset, writes, assertions, clocks, sigs)
    assert set(result) == {"a", "out", "reg", "array"}
    assert result["reg"] is same and result["array"] is same
    assert body[0].cond == reset[0].rhs == writes[0].addr == writes[0].enable == Id("a")
    assert body[0].then_body[0].rhs is writes[0].data
    leaf = writes[0].data
    for _ in range(40):
        assert leaf.a is leaf.b
        leaf = leaf.a
    assert leaf == Id("a")
    assert all(a.clock == a.failure == Id("a") for a in assertions)
    assert assertions[0].message == "message" and assertions[0].scope == "scope"
    assert assertions[1].pieces == ("v=", DecimalFormat(Id("a"), 1, 4))
    assert assertions[1].stream == "stdout" and assertions[1].scope == "scope"
    assert clocks == [ClockedProcess(Id("a"), "scope")]


def test_dedup_memos_are_local_and_release_input_maps_without_cyclic_gc():
    class TrackedDict(dict):
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        def run(canonical):
            expr = diamond(20)
            assigns = TrackedDict({canonical: expr, "z": expr, "out": Id("z")})
            sigs = TrackedDict({n: Signal(n, 1, "wire") for n in assigns})
            sigs["out"].kind = "output"
            refs = [weakref.ref(assigns), weakref.ref(sigs)]
            result = opt._dedup_assigns(assigns, [], [], [], [], [], sigs)[0]
            assert result["out"] == Id(canonical)
            return refs

        assert all(ref() is None for ref in run("a"))
        assert all(ref() is None for ref in run("b"))
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()
