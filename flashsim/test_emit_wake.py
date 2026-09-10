"""Hold-bucket wake tables must be sound for any design.

A hold sleeps once its body stops changing values, so the wake set has to
cover the data path as well as the condition. Waking on condition leaves
alone strands sequential state whenever a next value repeats for a cycle.
"""

from __future__ import annotations

from flashsim.emit import _stmt_stop_leaves
from flashsim.ir import BinOp, Const, Id, If, MemRead, NbAssign


def test_wake_covers_data_path() -> None:
    stmt = If(Id("en"), [NbAssign("r", BinOp("+", Id("x"), Const(1, 32)))])
    assert _stmt_stop_leaves(stmt, {}, {"en", "x"}, {}) == {"en", "x"}


def test_wake_follows_combinational_wires() -> None:
    assigns = {"cond": BinOp("&", Id("a"), Id("b")), "val": Id("c")}
    stmt = If(Id("cond"), [NbAssign("r", Id("val"))])
    leaves = _stmt_stop_leaves(stmt, assigns, {"a", "b", "c"}, {})
    assert leaves == {"a", "b", "c"}


def test_wake_covers_both_arms_of_nested_holds() -> None:
    stmt = If(
        Id("en"),
        [If(Id("sel"), [NbAssign("r", Id("x"))], [NbAssign("r", Id("y"))])],
    )
    leaves = _stmt_stop_leaves(stmt, {}, {"en", "sel", "x", "y"}, {})
    assert leaves == {"en", "sel", "x", "y"}


def test_wake_covers_memory_reads() -> None:
    stmt = If(Id("en"), [NbAssign("r", MemRead("m", Id("addr")))])
    leaves = _stmt_stop_leaves(stmt, {}, {"en", "m", "addr"}, {})
    assert leaves == {"en", "m", "addr"}


def test_wake_uses_cached_wire_deps() -> None:
    assigns = {"w": BinOp("+", Id("p"), Id("q"))}
    wire_deps = {"w": {"p", "q"}}
    stmt = If(Id("en"), [NbAssign("r", Id("w"))])
    leaves = _stmt_stop_leaves(stmt, assigns, {"en", "p", "q"}, wire_deps)
    assert leaves == {"en", "p", "q"}


if __name__ == "__main__":
    test_wake_covers_data_path()
    test_wake_follows_combinational_wires()
    test_wake_covers_both_arms_of_nested_holds()
    test_wake_covers_memory_reads()
    test_wake_uses_cached_wire_deps()
    print("ok")
