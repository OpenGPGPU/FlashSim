"""NBA mux lowering must follow CIRCT semantics, never guess at intent.

CIRCT emits `when (en) r <= v` (undriven else) as a self-referencing mux,
mux(en, v, r). So a self-referencing arm is the only arm that may be dropped
as a hold. A constant arm — zero included — is a value the design really
drives, and dropping it strands whatever state the design meant to clear.
"""

from __future__ import annotations

from flashsim.ir import Const, Id, If, NbAssign, Signal, Ternary
from flashsim.opt import _follow, _nba_to_if


def _sigs(**widths: int) -> dict[str, Signal]:
    return {n: Signal(n, w, "reg") for n, w in widths.items()}


def _flatten(stmts):
    out = []
    for s in stmts:
        if isinstance(s, If):
            out.extend(_flatten(s.then_body))
            out.extend(_flatten(s.else_body))
        else:
            out.append(s)
    return out


def _writes_zero(stmts, lhs: str) -> bool:
    return any(
        isinstance(s, NbAssign)
        and s.lhs == lhs
        and isinstance(s.rhs, Const)
        and s.rhs.value == 0
        for s in _flatten(stmts)
    )


def test_self_reference_else_is_a_hold() -> None:
    expr = Ternary(Id("en"), Id("next"), Id("line"))
    body = _nba_to_if("line", expr, _sigs(line=32, en=1, next=32))
    assert len(body) == 1 and isinstance(body[0], If)
    assert not body[0].else_body


def test_self_reference_then_inverts_the_condition() -> None:
    expr = Ternary(Id("clear"), Id("line"), Id("next"))
    body = _nba_to_if("line", expr, _sigs(line=32, clear=1, next=32))
    assert len(body) == 1 and isinstance(body[0], If)
    assert not body[0].else_body
    assert body[0].cond.op == "!"


def test_zero_else_is_real_data_not_a_hold() -> None:
    # mux(en, val, 0): the design clears the register when !en.
    expr = Ternary(Id("en"), Id("next"), Const(0, 32))
    body = _nba_to_if("addr", expr, _sigs(addr=32, en=1, next=32))
    assert body[0].else_body
    assert _writes_zero(body, "addr")


def test_const_priority_mux_keeps_zero_arm() -> None:
    inner = Ternary(Id("go0"), Const(0, 2), Id("state"))
    expr = Ternary(Id("go1"), Const(1, 2), inner)
    body = _nba_to_if("state", expr, _sigs(state=2, go0=1, go1=1))
    assert _writes_zero(body, "state")


def test_nested_value_mux_keeps_zero_under_a_hold() -> None:
    # mux(latch, mux(sel, v, 0), r): hold on !latch, clear when !sel.
    inner = Ternary(Id("sel"), Id("v"), Const(0, 32))
    expr = Ternary(Id("latch"), inner, Id("param"))
    body = _nba_to_if("param", expr, _sigs(param=32, latch=1, sel=1, v=32))
    assert not body[0].else_body
    assert _writes_zero(body, "param")


def test_follow_unwraps_ssa_to_the_mux() -> None:
    assigns = {"t0": Ternary(Id("en"), Id("next"), Id("line"))}
    sigs = _sigs(line=32, en=1, next=32)
    body = _nba_to_if("line", _follow(Id("t0"), assigns, "line", sigs), sigs)
    assert isinstance(body[0], If)
    assert not body[0].else_body


if __name__ == "__main__":
    test_self_reference_else_is_a_hold()
    test_self_reference_then_inverts_the_condition()
    test_zero_else_is_real_data_not_a_hold()
    test_const_priority_mux_keeps_zero_arm()
    test_nested_value_mux_keeps_zero_under_a_hold()
    test_follow_unwraps_ssa_to_the_mux()
    print("ok")
