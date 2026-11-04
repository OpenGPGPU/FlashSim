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


def test_mega_cones_keep_promotions_in_nested_else_and_bare_arms():
    names = [f"system_computeUnits_0_t{i}" for i in range(300)]
    assigns = {name: BinOp("&", Id("enable"), Const(1, 1)) for name in names}
    # Cold SSA names are traversed but must not be promoted by the GPU rule.
    assigns["other_t0"] = BinOp("&", Id("enable"), Const(1, 1))
    shared = Id("other_t0")
    for name in names:
        shared = BinOp("|", shared, Id(name))
    body = [If(Id("enable"),
               [If(Id("inner"), [NbAssign("x", shared)], [NbAssign("y", shared)])],
               [NbAssign("z", shared)])]
    cached = set(names[:3])
    _promote_bulky_gated_cones(body, assigns, cached, {"enable", "inner"})
    assert cached == set(names)
