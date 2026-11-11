"""Single-use inlining preserves independently evaluated logic and roots."""

import random
import subprocess

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Assign, BinOp, Const, Id, MemWrite, Module, NbAssign, Signal, Ternary
from flashsim.opt import _inline_single_use, optimize


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


def test_long_operation_chain_expands_without_python_recursion():
    size = 5000
    chain = {f"w{i}": BinOp("&", Id(f"w{i-1}" if i else "input"), Const(1, 1))
             for i in range(size)}
    result, _ = _inline_single_use(chain, [], {f"w{size-1}"})
    expr = result[f"w{size-1}"]
    seen = 0
    while isinstance(expr, BinOp) or (isinstance(expr, Id) and expr.name in result):
        if isinstance(expr, Id):
            expr = result[expr.name]
        else:
            assert expr.op == "&" and expr.b == Const(1, 1)
            expr = expr.a
            seen += 1
    assert seen == size and len(result) > 1
    assert expr == Id("input")


def test_deep_mux_chain_emits_and_executes(tmp_path):
    size = 7500
    sigs = {"clock": Signal("clock", 1, "input"), "out": Signal("out", 1, "output")}
    sigs.update({f"i{i}": Signal(f"i{i}", 1, "input") for i in range(8)})
    assigns = []
    for i in range(size):
        name = f"w{i}"
        sigs[name] = Signal(name, 1, "wire")
        assigns.append(Assign(name, Ternary(Id(f"i{i % 8}"),
                                           Id(f"w{i-1}" if i else "i0"), Id("i1"))))
    assigns.append(Assign("out", Id(f"w{size-1}")))
    module = Module("DeepSynthetic", sigs, assigns, AlwaysFF("clock", []), ["clock"])
    (tmp_path / "dut.h").write_text(emit_cpp(optimize(module)))
    set_inputs = " ".join(f"d.i{i}=(bits>>{i})&1;" for i in range(8))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\nint main() {\n'
        'for (unsigned bits=0; bits<256; ++bits) { DeepSyntheticDut d; ' + set_inputs +
        ' unsigned expected=bits&1; for (int i=0;i<7500;++i) '
        'expected=((bits>>(i%8))&1) ? expected : ((bits>>1)&1); '
        'd.poke_inputs(); d.eval_out(); if (d.out!=expected) return 1; } return 0; }\n')
    subprocess.run(["c++", "-std=c++17", "-O0", str(tmp_path / "check.cpp"),
                    "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)
