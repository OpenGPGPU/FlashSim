"""Single-use inlining preserves independently evaluated logic and roots."""

import random

from flashsim.ir import BinOp, Const, Id, MemWrite, NbAssign, Ternary
from flashsim.opt import _inline_single_use


def _value(expr, inputs, assigns):
    if isinstance(expr, Id):
        return inputs[expr.name] if expr.name in inputs else _value(assigns[expr.name], inputs, assigns)
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, BinOp):
        a, b = _value(expr.a, inputs, assigns), _value(expr.b, inputs, assigns)
        return {"&": a & b, "|": a | b, "^": a ^ b}[expr.op]
    if isinstance(expr, Ternary):
        branch = expr.a if _value(expr.cond, inputs, assigns) else expr.b
        return _value(branch, inputs, assigns)
    raise TypeError(type(expr))


def test_random_graphs_preserve_truth_tables():
    rng = random.Random(2047)
    for _ in range(100):
        inputs = [f"i{i}" for i in range(4)]
        names = list(inputs)
        items = []
        for index in range(35):
            def pick():
                return Id(rng.choice(names))
            expr = pick() if rng.random() < 0.25 else BinOp(rng.choice("&|^"), pick(), pick())
            if rng.random() < 0.1:
                expr = Ternary(pick(), expr, pick())
            name = f"w{index}"
            items.append((name, expr))
            names.append(name)
        rng.shuffle(items)
        original = dict(items)
        outputs = {"w33", "w34"}
        optimized, _ = _inline_single_use(dict(original), [], outputs)
        for bits in range(16):
            values = {name: (bits >> i) & 1 for i, name in enumerate(inputs)}
            for output in outputs:
                assert _value(Id(output), values, optimized) == _value(Id(output), values, original)


def test_register_memory_and_output_roots_are_retained():
    assigns = {"a": Id("input"), "b": Id("a"), "c": Id("a"), "d": Id("b")}
    body = [NbAssign("reg", Id("b"))]
    writes = [MemWrite("mem", Const(0, 1), Id("d"), Const(1, 1))]
    result, returned_body = _inline_single_use(dict(assigns), body, {"c"}, writes)
    assert returned_body is body
    assert {"b", "c", "d"} <= result.keys()
    assert result["b"] == Id("a") and result["c"] == Id("a")
    assert result["d"] == Id("b")


def test_alias_cycle_and_long_chain_do_not_recurse_forever():
    cycle = {"a": Id("b"), "b": Id("a"), "out": Id("a")}
    result, _ = _inline_single_use(cycle, [], {"out"})
    assert result == {"a": Id("a"), "out": Id("a")}
    chain = {f"w{i}": Id(f"w{i-1}" if i else "input") for i in reversed(range(2000))}
    result, _ = _inline_single_use(chain, [], {"w1999"})
    assert result == {"w1999": Id("input")}
