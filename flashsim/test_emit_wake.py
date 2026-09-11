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


def test_split_dut_methods_out_of_line() -> None:
    from flashsim.emit import split_dut_methods

    # Large body forces out-of-line extraction past _SPLIT_INLINE_MAX_BYTES.
    big = "    x = x + 1;\n" * 80
    mono = f"""\
#include <cstdint>

struct FooDut {{
  uint32_t x;
  uint32_t x__ok = 0;
  uint32_t _pg[1];
  uint8_t __inited = 0;
  uint32_t _chg = 0;

  void _note(uint16_t i) {{ (void)i; }}

  void poke_inputs() {{
{big}    __inited = 1;
  }}

  void eval_x() {{
    if (x__ok == _pg[0]) return;
    x = 1;
    x__ok = _pg[0];
  }}

  void eval_y() {{}}

  void tick() {{
{big}    poke_inputs();
  }}
}};
"""
    header, parts = split_dut_methods(mono, n_shards=2)
    assert "void poke_inputs();" in header
    assert "void tick();" in header
    # tiny eval / empty stay in-class for cross-TU inlining
    assert "void eval_x() {" in header
    assert "void eval_y() {}" in header
    assert "void _note(uint16_t i) { (void)i; }" in header
    assert "void FooDut::poke_inputs()" in parts[0][1]
    bodies = "".join(b for _, b in parts)
    assert "void FooDut::tick()" in bodies
    assert "void FooDut::eval_x()" not in bodies
    assert "void poke_inputs() {" not in header


def test_l2_partition_keeps_slice_submodule() -> None:
    from flashsim.emit import skip_partition_key

    assert (
        skip_partition_key("system_l2_slices_1_memoryRequestArbiter_t29")
        == "l2_slices_1_memoryRequestArbiter"
    )
    assert skip_partition_key("system_l2_slices_1_t59013") == "l2_slices_1_b5"
    assert skip_partition_key("system_l2_slices_1_t64") == "l2_slices_1_b0"
    assert (
        skip_partition_key("system_l2_slices_0_mshrTable_valid_0")
        == "l2_slices_0_mshrTable"
    )
    assert skip_partition_key("system_l2_slices_1_lowerIssued") == "l2_slices_1_lowerIssued"
    assert (
        skip_partition_key("memoryAxi_io_response_bits_fault")
        == "memoryAxi_io_response_bits"
    )
    assert (
        skip_partition_key("memoryAxi_io_response_valid")
        == "memoryAxi_io_response_valid"
    )


if __name__ == "__main__":
    test_wake_covers_data_path()
    test_wake_follows_combinational_wires()
    test_wake_covers_both_arms_of_nested_holds()
    test_wake_covers_memory_reads()
    test_wake_uses_cached_wire_deps()
    test_split_dut_methods_out_of_line()
    test_l2_partition_keeps_slice_submodule()
    print("ok")
