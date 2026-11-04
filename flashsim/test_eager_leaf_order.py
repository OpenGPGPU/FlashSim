"""Preserve eager pass IDs while avoiding unused large lexicographic keys."""
import random

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, BinOp, Id, Module, NbAssign, Signal


def test_leaf_set_order_matches_size_then_full_lexicographic_order():
    randomizer = random.Random(7301)
    names = [f"computeUnits_{i%4}_core_register_{i}" for i in range(50)]
    keys = [frozenset(), frozenset({"t2"}), frozenset({"t10"}), frozenset({"t1"})]
    for _ in range(500):
        keys.append(frozenset(randomizer.sample(names, randomizer.randrange(len(names)+1))))
    randomizer.shuffle(keys)
    expected = sorted(keys, key=lambda leaves: (len(leaves), sorted(leaves)))
    assert emit._ordered_leaf_sets(keys) == expected


def test_codegen_is_byte_identical_to_full_leaf_sort(monkeypatch):
    signals = {"clk": Signal("clk",1,"input"), "out": Signal("out",8,"output")}
    for i in range(6): signals[f"d{i}"] = Signal(f"d{i}",8,"input")
    assigns = []
    dependencies = [["d0"],["d1"],["d0","d1"],["d2","d3"],
                    ["d0","d1","d2"],["d0","d1","d2","d3","d4"]]
    for i in range(160):
        names = dependencies[i%len(dependencies)]
        name = f"unit_pipe_t{i}"
        signals[name] = Signal(name,8,"wire")
        value = Id(names[0])
        for operand in names[1:]: value = BinOp("^",value,Id(operand))
        assigns.append(Assign(name,value))
    body = []
    for i in range(160):
        name = f"r{i}"; signals[name] = Signal(name,8,"reg")
        body.append(NbAssign(name,Id(f"unit_pipe_t{i}")))
    assigns.append(Assign("out",BinOp("^",Id("r0"),Id("r159"))))
    model = Module("LeafOrder",signals,assigns,AlwaysFF("clk",body),
                   ["clk","out",*(f"d{i}" for i in range(6))])
    optimized = emit._ordered_leaf_sets
    observed = []
    def original(keys):
        keys = list(keys)
        observed.append([len(key) for key in keys])
        return sorted(keys,key=lambda leaves:(len(leaves),sorted(leaves)))
    monkeypatch.setattr(emit,"_ordered_leaf_sets",original)
    reference = emit.emit_cpp(model)
    monkeypatch.setattr(emit,"_ordered_leaf_sets",optimized)
    actual = emit.emit_cpp(model)
    assert observed and any(len(set(sizes))>1 for sizes in observed)
    assert actual == reference
