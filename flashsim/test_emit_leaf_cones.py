"""Shared leaf-cone computation must preserve wake and eager dependencies."""
import random

import pytest

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, BinOp, Concat, Const, Id, Module, NbAssign, Signal


@pytest.mark.parametrize("limit", [0, 1, 7, 32768])
def test_shared_leaves_match_independent_walks_with_cycles_and_stops(limit):
    randomizer = random.Random(8617)
    for _ in range(40):
        names = [f"w{i}" for i in range(45)]
        choices = names + ["input0", "input1", "unbound"]
        assigns = {}
        for name in names:
            operands = randomizer.sample(choices, randomizer.randrange(1, 4))
            expr = Id(operands[0])
            for operand in operands[1:]:
                expr = BinOp("|", expr, Id(operand))
            assigns[name] = expr
        roots = set(randomizer.sample(names + ["unbound"], 20))
        stop = {"input0", "input1"} | set(randomizer.sample(names, 4))
        expected = {name: emit.cone_leaves(name, assigns, stop) for name in sorted(roots)}
        assert emit._cone_leaves_many(roots, assigns, stop, limit) == expected


def test_deep_shared_chain_does_not_recurse():
    assigns = {"w0": Id("input")}
    for i in range(1, 6000):
        assigns[f"w{i}"] = BinOp("^", Id(f"w{i-1}"), Const(1, 0))
    roots = {f"w{i}" for i in range(0, 6000, 29)}
    assert emit._cone_leaves_many(roots, assigns, {"input"}) == {
        name: {"input"} for name in roots
    }


@pytest.mark.parametrize("budget", [0, 300, 600, 800, 1048576])
def test_large_shared_cones_and_cycles_match_independent_walks(budget):
    leaves = [f"input{i}" for i in range(700)]
    assigns = {
        "shared0": Concat(tuple(Id(name) for name in leaves[:400])),
        "shared1": Concat(tuple(Id(name) for name in leaves[300:])),
        "stopped": Concat(tuple(Id(name) for name in leaves)),
        "cycle0": BinOp("|", Id("cycle1"), Id("shared0")),
        "cycle1": BinOp("|", Id("cycle0"), Id("shared1")),
    }
    for i in range(24):
        assigns[f"root{i}"] = BinOp("|", Id(f"shared{i % 2}"), Id(f"extra{i}"))
    assigns["stopped_root"] = BinOp("|", Id("stopped"), Id("shared0"))
    roots = {f"root{i}" for i in range(24)} | {"cycle0", "cycle1", "stopped_root"}
    stop = set(leaves) | {"stopped"}
    expected = {name: emit.cone_leaves(name, assigns, stop) for name in roots}
    assert emit._cone_leaves_many(roots, assigns, stop, 7, budget) == expected


def test_emitted_model_is_byte_identical_to_independent_leaf_walks(monkeypatch):
    signals = {"clk": Signal("clk", 1, "input"),
               "input0": Signal("input0", 8, "input"),
               "input1": Signal("input1", 8, "input"),
               "out": Signal("out", 8, "output")}
    assigns, body = [], []
    for i in range(280):
        wire, reg = f"unit_pipe_t{i}", f"r{i}"
        signals[wire] = Signal(wire, 8, "wire")
        signals[reg] = Signal(reg, 8, "reg")
        prev = Id(f"unit_pipe_t{i-1}") if i else Id("input0")
        assigns.append(Assign(wire, BinOp("^", prev, Id("input1"))))
        body.append(NbAssign(reg, Id(wire)))
    assigns.append(Assign("out", Id("r279")))
    model = Module("SharedLeaves", signals, assigns, AlwaysFF("clk", body),
                   ["clk", "input0", "input1", "out"])
    candidate = emit._cone_leaves_many
    monkeypatch.setattr(emit, "_cone_leaves_many",
                        lambda roots, mapping, stop: {
                            name: emit.cone_leaves(name, mapping, stop)
                            for name in sorted(roots)
                        })
    reference = emit.emit_cpp(model)
    monkeypatch.setattr(emit, "_cone_leaves_many", candidate)
    assert emit.emit_cpp(model) == reference
