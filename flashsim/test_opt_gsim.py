"""GSIM-inspired narrowing: bit slices, one-hot decodes, shared reset, extract cost."""

from __future__ import annotations

import subprocess

from flashsim.emit import _worth_extracting, emit_cpp
from flashsim.ir import (
    AlwaysFF,
    Assign,
    BinOp,
    Concat,
    Const,
    Extract,
    Id,
    If,
    Module,
    NbAssign,
    Signal,
    UnaryOp,
    expr_ids,
)
from flashsim.opt import optimize


def _signals() -> dict[str, Signal]:
    return {
        "clock": Signal("clock", 1, "input"),
        "hi": Signal("hi", 8, "input"),
        "lo": Signal("lo", 8, "input"),
        "idx": Signal("idx", 4, "input"),
        "core_reset": Signal("core_reset", 1, "input"),
        "en": Signal("en", 1, "input"),
        "a_next": Signal("a_next", 8, "input"),
        "b_next": Signal("b_next", 8, "input"),
        "s_next": Signal("s_next", 8, "input"),
        "wide": Signal("wide", 16, "wire"),
        "sh": Signal("sh", 8, "wire"),
        "out": Signal("out", 8, "output"),
        "bit": Signal("bit", 1, "output"),
        "a": Signal("a", 8, "reg"),
        "b": Signal("b", 8, "reg"),
        "s": Signal("s", 8, "reg"),
    }


def _module(assigns: list[Assign], body: list) -> Module:
    return Module("Slice", _signals(), assigns, AlwaysFF("clock", body), ["clock"])


def test_extract_of_concat_drops_the_unused_half() -> None:
    mod = optimize(_module(
        [
            Assign("wide", Concat((Id("hi"), Id("lo")))),
            Assign("out", Extract(Id("wide"), 0, 8)),
        ],
        [],
    ))
    assigns = {item.lhs: item.rhs for item in mod.assigns}
    assert assigns["out"] == Id("lo")
    assert "hi" not in expr_ids(assigns["out"])


def test_one_hot_bit_test_becomes_a_compare() -> None:
    mod = optimize(_module(
        [
            Assign("sh", BinOp("<<", Const(1, 8), Id("idx"))),
            Assign("bit", Extract(Id("sh"), 3, 1)),
        ],
        [],
    ))
    assigns = {item.lhs: item.rhs for item in mod.assigns}
    assert assigns["bit"] == BinOp("==", Id("idx"), Const(3, 4))
    assert "sh" not in assigns


def test_one_hot_ignores_a_shifted_value_other_than_one() -> None:
    mod = optimize(_module(
        [
            Assign("sh", BinOp("<<", Const(2, 8), Id("idx"))),
            Assign("bit", Extract(Id("sh"), 3, 1)),
        ],
        [],
    ))
    assigns = {item.lhs: item.rhs for item in mod.assigns}
    assert not (isinstance(assigns["bit"], BinOp) and assigns["bit"].op == "==")


def test_nested_extract_collapses_to_one_window() -> None:
    sigs = _signals()
    sigs["src"] = Signal("src", 16, "input")
    sigs["out"] = Signal("out", 3, "output")
    mod = optimize(Module(
        "Slice",
        sigs,
        [Assign("out", Extract(Extract(Id("src"), 4, 8), 1, 3))],
        AlwaysFF("clock", []),
        ["clock"],
    ))
    assigns = {item.lhs: item.rhs for item in mod.assigns}
    assert assigns["out"] == Extract(Id("src"), 5, 3)


def test_shared_reset_is_one_epilogue_and_data_is_unguarded() -> None:
    mod = optimize(_module(
        [],
        [
            If(Id("core_reset"), [NbAssign("a", Const(0, 8))], [NbAssign("a", Id("a_next"))]),
            If(Id("core_reset"), [NbAssign("b", Const(7, 8))], [NbAssign("b", Id("b_next"))]),
        ],
    ))
    assert mod.always.body == [NbAssign("a", Id("a_next")), NbAssign("b", Id("b_next"))]
    assert len(mod.reset_epilogue) == 1
    arm = mod.reset_epilogue[0]
    assert isinstance(arm, If)
    writes = {stmt.lhs: stmt.rhs.value for stmt in arm.then_body if isinstance(stmt, NbAssign)}
    assert writes == {"a": 0, "b": 7}
    assert "core_reset" in expr_ids(arm.cond)


def test_active_low_reset_groups_with_the_same_net() -> None:
    mod = optimize(_module(
        [],
        [
            If(Id("core_reset"), [NbAssign("a", Const(0, 8))], [NbAssign("a", Id("a_next"))]),
            If(
                UnaryOp("!", Id("core_reset")),
                [NbAssign("b", Id("b_next"))],
                [NbAssign("b", Const(7, 8))],
            ),
        ],
    ))
    assert len(mod.reset_epilogue) == 1
    assert {stmt.lhs for stmt in mod.reset_epilogue[0].then_body} == {"a", "b"}


def test_enable_clear_is_not_treated_as_reset() -> None:
    mod = optimize(_module(
        [],
        [
            If(Id("en"), [NbAssign("a", Const(0, 8))], [NbAssign("a", Id("a_next"))]),
            If(Id("en"), [NbAssign("b", Const(0, 8))], [NbAssign("b", Id("b_next"))]),
        ],
    ))
    assert mod.reset_epilogue == []


def test_reset_hoist_keeps_a_data_arm_that_writes_an_extra_register() -> None:
    mod = optimize(_module(
        [],
        [
            If(
                Id("core_reset"),
                [NbAssign("a", Const(0, 8))],
                [NbAssign("a", Id("a_next")), NbAssign("s", Id("s_next"))],
            ),
            If(Id("core_reset"), [NbAssign("b", Const(0, 8))], [NbAssign("b", Id("b_next"))]),
        ],
    ))
    assert mod.reset_epilogue == []


def test_extract_cost_promotes_a_fat_tree_before_a_tiny_one() -> None:
    assert _worth_extracting(40, 2)
    assert not _worth_extracting(3, 3)
    assert _worth_extracting(1, 16)
    assert not _worth_extracting(40, 1)


def test_reset_epilogue_overrides_next_state(tmp_path) -> None:
    mod = optimize(_module(
        [],
        [
            If(Id("core_reset"), [NbAssign("a", Const(0, 8))], [NbAssign("a", Id("a_next"))]),
            If(Id("core_reset"), [NbAssign("b", Const(7, 8))], [NbAssign("b", Id("b_next"))]),
        ],
    ))
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\n'
        "int main() {\n"
        "  SliceDut d;\n"
        "  d.a_next = 3; d.b_next = 4; d.core_reset = 0;\n"
        "  d.tick();\n"
        "  if (d.a != 3 || d.b != 4) return 1;\n"
        "  d.a_next = 9; d.b_next = 9; d.core_reset = 1;\n"
        "  d.tick();\n"
        "  if (d.a != 0 || d.b != 7) return 2;\n"
        "  d.a_next = 1; d.b_next = 2; d.core_reset = 0;\n"
        "  d.tick();\n"
        "  if (d.a != 1 || d.b != 2) return 3;\n"
        "  return 0;\n"
        "}\n"
    )
    subprocess.run(
        ["c++", "-std=c++17", str(tmp_path / "check.cpp"), "-o", str(tmp_path / "check")],
        check=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True)
