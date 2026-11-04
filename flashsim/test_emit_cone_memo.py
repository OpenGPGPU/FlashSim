"""Repeated gated expressions still promote all eligible signal cones."""

from flashsim.emit import _promote_bulky_gated_cones
from flashsim.ir import BinOp, Const, Id, If, NbAssign


def test_shared_expression_in_nested_and_else_arms_promotes_its_cones():
    assigns = {}
    names = [f"system_computeUnits_0_t{i}" for i in range(64)]
    for name in names:
        expr = Id("enable")
        for _ in range(6):
            expr = BinOp("&", expr, Const(1, 1))
        assigns[name] = expr
    shared = Id(names[0])
    for name in names[1:]:
        shared = BinOp("|", shared, Id(name))
    body = [If(Id("enable"), [If(Id("inner"), [NbAssign("a", shared)],
                                     [NbAssign("b", shared)])])]
    cached = set()
    _promote_bulky_gated_cones(body, assigns, cached, {"enable", "inner"})
    assert set(names) <= cached
