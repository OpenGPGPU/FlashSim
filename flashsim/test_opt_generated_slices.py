"""Generated slice wires must be usable by subsequent optimization passes."""

import subprocess

import pytest

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Assign, BinOp, Concat, Const, Extract, Id, If, MemWrite, Module, NbAssign, Signal
from flashsim.opt import _cse_sibling_or_prefixes, _narrow_bit_slices, _reassoc_and_chains, _slice_src, optimize


def circuit():
    inputs = [f"a{i}" for i in range(20)]
    signals = {name: Signal(name, 1, "input") for name in inputs + ["tail", "gate"]}
    signals["wide"] = Signal("wide", 2, "wire")
    signals["first"] = Signal("first", 1, "output")
    signals["second"] = Signal("second", 1, "output")
    large = Id(inputs[0])
    for name in inputs[1:]:
        large = BinOp("&", large, Id(name))
    return signals, Concat((large, Id("tail")))


def test_assignment_slice_registers_width_before_reassociation():
    signals, wide = circuit()
    bit = Extract(Id("wide"), 1, 1)
    assigns = {"wide": wide, "first": BinOp("&", bit, Id("gate")), "second": bit}
    narrowed, body, writes = _narrow_bit_slices(assigns, [], [], signals)
    assert not body and not writes
    generated = [name for name in narrowed if name.startswith("wide__b1_1")]
    assert len(generated) == 1
    assert signals[generated[0]].width == 1
    assert narrowed["second"] == Id(generated[0])
    _reassoc_and_chains(narrowed, signals)


def test_reused_shared_slice_skips_rewalking_its_definition():
    signals, wide = circuit()

    class CountReads(dict):
        reads = 0

        def __getitem__(self, key):
            self.reads += 1
            return super().__getitem__(key)

    assigns = CountReads(wide=wide)
    fresh, cache = {}, {}
    first = _slice_src(Id("wide"), 1, 1, assigns, signals, {"wide": 2}, fresh, cache, set())
    assert isinstance(first, Id) and first.name in fresh
    reads = assigns.reads
    second = _slice_src(Id("wide"), 1, 1, assigns, signals, {"wide": 2}, fresh, cache, set())
    assert second == first and assigns.reads == reads
    # An active recursion or exhausted fuel must not reuse the cached slice.
    assert _slice_src(Id("wide"), 1, 1, assigns, signals, {"wide": 2}, fresh, cache, {"wide"}) is None
    assert _slice_src(Id("wide"), 1, 1, assigns, signals, {"wide": 2}, fresh, cache, set(), 0) is None


def test_sequential_only_slice_has_a_definition_and_width():
    signals, wide = circuit()
    signals["r0"] = Signal("r0", 1, "reg")
    signals["r1"] = Signal("r1", 1, "reg")
    bit = Extract(Id("wide"), 1, 1)
    narrowed, body, _ = _narrow_bit_slices(
        {"wide": wide}, [NbAssign("r0", bit), NbAssign("r1", bit)], [], signals)
    assert isinstance(body[0].rhs, Id) and body[0].rhs == body[1].rhs
    name = body[0].rhs.name
    assert name in narrowed and signals[name].width == 1


def test_memory_write_only_slice_has_a_definition_and_width():
    signals, wide = circuit()
    signals["mem"] = Signal("mem", 1, "mem", 2)
    bit = Extract(Id("wide"), 1, 1)
    writes = [MemWrite("mem", Const(i, 1), bit, Const(1, 1)) for i in range(2)]
    narrowed, _, writes = _narrow_bit_slices({"wide": wide}, [], writes, signals)
    assert isinstance(writes[0].data, Id) and writes[0].data == writes[1].data
    name = writes[0].data.name
    assert name in narrowed and signals[name].width == 1


def test_cse_temporary_avoids_existing_signals_and_has_a_width():
    inputs = [f"a{i}" for i in range(20)]
    signals = {name: Signal(name, 1, "input") for name in inputs +
               ["_fs_cse_t0", "u0", "u1", "u2"]}
    signals["_fs_cse_t1"] = Signal("_fs_cse_t1", 1, "wire")
    shared = Id(inputs[0])
    for name in inputs[1:]:
        shared = BinOp("&", shared, Id(name))
    shared = BinOp("&", shared, Id("_fs_cse_t0"))
    body = [If(BinOp("&", shared, Id(f"u{i}")), [NbAssign(f"r{i}", Const(1, 1))])
            for i in range(3)]
    assigns = {"_fs_cse_t1": Id(inputs[0])}
    transformed = _cse_sibling_or_prefixes(body, assigns, signals)
    generated = [name for name in assigns if name != "_fs_cse_t1"]
    assert len(generated) == 1 and generated[0] not in {"_fs_cse_t0", "_fs_cse_t1"}
    assert signals[generated[0]].width == 1
    assert all(isinstance(stmt.cond, BinOp) and stmt.cond.a == Id(generated[0])
               for stmt in transformed)


def test_reassociation_leaves_unknown_width_expression_for_validation():
    signals = {"known": Signal("known", 1, "input"), "out": Signal("out", 1, "output")}
    original = BinOp("&", Id("known"), Id("unknown"))
    assert _reassoc_and_chains({"out": original}, signals)["out"] == original


def test_missing_wire_is_still_rejected_by_full_validation():
    signals = {name: Signal(name, 1, kind) for name, kind in
               (("clock", "input"), ("known", "input"), ("out", "output"))}
    mod = Module("Invalid", signals, [Assign("out", BinOp("&", Id("known"), Id("missing")))],
                 AlwaysFF("clock", []), ["clock"])
    with pytest.raises(ValueError, match="dangling wire missing"):
        optimize(mod)


def test_full_optimizer_emits_executable_slice_logic(tmp_path):
    signals, wide = circuit()
    signals["clock"] = Signal("clock", 1, "input")
    signals["r0"] = Signal("r0", 1, "reg")
    signals["r1"] = Signal("r1", 1, "reg")
    bit = Extract(Id("wide"), 1, 1)
    mod = Module("GeneratedSlices", signals,
                 [Assign("wide", wide), Assign("first", BinOp("&", bit, Id("gate"))),
                  Assign("second", bit)],
                 AlwaysFF("clock", [NbAssign("r0", bit), NbAssign("r1", bit)]), ["clock"])
    (tmp_path / "dut.h").write_text(emit_cpp(optimize(mod)))
    setup = " ".join(f"d.a{i}=1;" for i in range(20))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\nint main() { GeneratedSlicesDut d;\n' + setup +
        " d.gate=1; d.tick(); d.eval_first(); d.eval_second();\n"
        " if (!d.r0 || !d.r1 || !d.first || !d.second) return 1;\n"
        " d.a7=0; d.tick(); d.eval_first(); d.eval_second();\n"
        " if (d.r0 || d.r1 || d.first || d.second) return 2;\n"
        " return 0; }\n")
    subprocess.run(["c++", "-std=c++17", str(tmp_path / "check.cpp"), "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)
