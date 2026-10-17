"""DCE preserves every root and traverses deep dependency graphs only once."""

from flashsim import opt
from flashsim.ir import BinOp, Const, Id, If, MemWrite, NbAssign


def test_dce_preserves_statement_memory_and_output_roots_in_source_order():
    assigns = {
        "unused": Const(7),
        "data_leaf": Id("input"),
        "output": Id("output_leaf"),
        "guard": Id("guard_leaf"),
        "output_leaf": Const(1),
        "guard_leaf": Const(1),
        "data": Id("data_leaf"),
        "addr": Id("index"),
        "enable": Id("valid"),
        "ram": Id("memory_root"),
    }
    original = dict(assigns)
    body = [If(Id("guard"), [NbAssign("reg", Id("data"))], [])]
    writes = [MemWrite("ram", Id("addr"), Id("data"), Id("enable"))]

    kept, returned_body = opt._dce(assigns, body, {"output"}, writes)

    assert list(kept) == [name for name in original if name != "unused"]
    assert all(kept[name] is original[name] for name in kept)
    assert returned_body is body
    assert assigns == original


def test_dce_retains_reachable_cycles_and_removes_unreachable_cycles():
    assigns = {
        "dead_a": Id("dead_b"),
        "a": Id("b"),
        "b": Id("a"),
        "dead_b": Id("dead_a"),
    }
    kept, _ = opt._dce(assigns, [], {"a"})
    assert list(kept) == ["a", "b"]


def test_dce_without_roots_removes_every_assignment():
    assigns = {"a": Id("b"), "b": Id("a"), "constant": Const(1)}
    body = []
    kept, returned_body = opt._dce(assigns, body, set())
    assert kept == {}
    assert returned_body is body
    assert len(assigns) == 3


def test_dce_visits_each_assignment_once_in_a_long_chain(monkeypatch):
    length = 4096
    assigns = {f"n{i}": Id(f"n{i + 1}") for i in range(length)}
    assigns[f"n{length}"] = Id("input")
    visits = 0
    expr_ids = opt.expr_ids

    def counted(expr):
        nonlocal visits
        visits += 1
        # A dependency-layer rescan fails promptly, without timing thresholds.
        assert visits <= len(assigns)
        return expr_ids(expr)

    monkeypatch.setattr(opt, "expr_ids", counted)
    kept, _ = opt._dce(assigns, [], {"n0"})
    assert kept == assigns
    assert list(kept) == list(assigns)
    assert visits == len(assigns)


def test_dce_visits_shared_dependencies_once(monkeypatch):
    assigns = {
        "output": BinOp("+", Id("left"), Id("right")),
        "left": Id("shared"),
        "right": Id("shared"),
        "shared": Id("input"),
    }
    visited = []
    expr_ids = opt.expr_ids

    def counted(expr):
        visited.append(expr)
        return expr_ids(expr)

    monkeypatch.setattr(opt, "expr_ids", counted)
    kept, _ = opt._dce(assigns, [], {"output"})
    assert kept == assigns
    assert len(visited) == len(assigns)
