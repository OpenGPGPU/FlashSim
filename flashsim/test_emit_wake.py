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

  void _commit() {{
{big}    _chg++;
  }}

  void tick() {{
{big}    poke_inputs();
  }}
}};
"""
    header, parts = split_dut_methods(mono, n_shards=2)
    assert "void poke_inputs();" in header
    assert "void tick();" in header
    assert "void _commit();" in header
    # tiny eval / empty stay in-class for cross-TU inlining
    assert "void eval_x() {" in header
    assert "void eval_y() {}" in header
    assert "void _note(uint16_t i) { (void)i; }" in header
    assert "void FooDut::poke_inputs()" in parts[0][1]
    bodies = "".join(b for _, b in parts)
    assert "void FooDut::tick()" in bodies
    assert "void FooDut::eval_x()" not in bodies
    assert "void poke_inputs() {" not in header
    # _commit gets its own TU so clang -O2 does not hang on the dirty switch.
    assert any(name == "dut_commit.cpp" for name, _ in parts)
    commit_body = next(b for name, b in parts if name == "dut_commit.cpp")
    assert "void FooDut::_commit()" in commit_body
    assert "void FooDut::_commit()" not in parts[0][1]
    assert "void FooDut::_commit()" not in parts[1][1]


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
    # CU mega-holds: coalescer / fmaAlu must not collapse to one partition.
    c0 = skip_partition_key("system_computeUnits_0_core_vectorCoalescer_t30")
    c1 = skip_partition_key("system_computeUnits_0_core_vectorCoalescer_t94")
    assert c0 == "computeUnits_0_core_vectorCoalescer_b30"
    assert c1 == "computeUnits_0_core_vectorCoalescer_b30"  # 94 % 32 == 30
    c2 = skip_partition_key("system_computeUnits_0_core_vectorCoalescer_t31")
    assert c2 == "computeUnits_0_core_vectorCoalescer_b31"
    assert c0 != c2
    f0 = skip_partition_key(
        "system_computeUnits_0_core_vector_fmaAlu_lanes_0_core_t188"
    )
    assert f0 == "computeUnits_0_core_vector_fmaAlu_l0_b12"  # 188 % 16


def test_stmt_bucket_key_prefers_write_partition() -> None:
    from flashsim.emit import _stmt_bucket_key, skip_partition_key
    from flashsim.ir import Const, Id, If, NbAssign

    # Cond reads sharedCachePort; write is vectorTlb — bucket follows the write.
    stmt = If(
        Id("system_computeUnits_0_core_sharedCachePort_arbiter_t8"),
        [NbAssign("system_computeUnits_0_core_vectorTlb_request_lineAddress", Const(0, 32))],
    )
    key = _stmt_bucket_key(stmt)
    assert key == skip_partition_key("system_computeUnits_0_core_vectorTlb_request_lineAddress")
    assert "vectorTlb" in key
    assert "sharedCachePort" not in key


def test_promote_large_gpu_ssa() -> None:
    from flashsim.emit import _expr_node_count, _promote_large_gpu_ssa
    from flashsim.ir import BinOp, Const, Id

    # Build a deep AND chain (≥ 48 nodes).
    expr: object = Id("leaf")
    for i in range(50):
        expr = BinOp("&", expr, Const(1, 1))  # type: ignore[assignment]
    assert _expr_node_count(expr) >= 48  # type: ignore[arg-type]
    assigns = {
        "system_l2_slices_0_t999": expr,  # type: ignore[dict-item]
        "system_l2_slices_0_t1": BinOp("&", Id("a"), Id("b")),
    }
    cached: set[str] = set()
    _promote_large_gpu_ssa(assigns, cached)  # type: ignore[arg-type]
    assert "system_l2_slices_0_t999" in cached
    assert "system_l2_slices_0_t1" not in cached


def test_promote_bulky_gated_cones() -> None:
    from flashsim.emit import _BULKY_TMP_NODES, _promote_bulky_gated_cones
    from flashsim.ir import BinOp, Const, Id, If, NbAssign

    # Rich SSA under a bulky hold region → promoted; tiny compares stay local.
    assigns: dict = {}
    body_assigns = []
    for i in range(60):
        # Build expr with ≥ _BULKY_TMP_NODES nodes.
        expr: object = Id("en")
        for _ in range(_BULKY_TMP_NODES):
            expr = BinOp("&", expr, Const(1, 1))
        assigns[f"system_computeUnits_0_t{i}"] = expr
        body_assigns.append(NbAssign(f"r{i}", Id(f"system_computeUnits_0_t{i}")))
    # One tiny temp should not be promoted even in a bulky region.
    assigns["system_computeUnits_0_t_tiny"] = BinOp("&", Id("en"), Const(1, 1))
    body_assigns.append(NbAssign("rt", Id("system_computeUnits_0_t_tiny")))
    body = [If(Id("en"), body_assigns)]
    cached: set[str] = set()
    _promote_bulky_gated_cones(body, assigns, cached, stop={"en"})
    assert "system_computeUnits_0_t0" in cached
    assert "system_computeUnits_0_t_tiny" not in cached


def test_promote_bulky_gated_cones_with_else() -> None:
    """Merged same-cond holds keep else; then-arm cones must still promote."""
    from flashsim.emit import _BULKY_TMP_NODES, _promote_bulky_gated_cones
    from flashsim.ir import BinOp, Const, Id, If, NbAssign

    assigns: dict = {}
    then_assigns = []
    for i in range(60):
        expr: object = Id("en")
        for _ in range(_BULKY_TMP_NODES):
            expr = BinOp("&", expr, Const(1, 1))
        assigns[f"system_computeUnits_0_t{i}"] = expr
        then_assigns.append(NbAssign(f"r{i}", Id(f"system_computeUnits_0_t{i}")))
    body = [If(Id("en"), then_assigns, [NbAssign("z", Const(0, 1))])]
    cached: set[str] = set()
    _promote_bulky_gated_cones(body, assigns, cached, stop={"en"})
    assert "system_computeUnits_0_t0" in cached


def test_promote_mega_gated_lowers_node_threshold() -> None:
    """≥256 SSA locals under a gate: promote even shallow temps (coalescer)."""
    from flashsim.emit import _MEGA_GATED_CONE, _promote_bulky_gated_cones
    from flashsim.ir import BinOp, Const, Id, If, NbAssign

    assigns: dict = {}
    then_assigns = []
    n = _MEGA_GATED_CONE + 10
    for i in range(n):
        # 3-node expr: below normal bulky threshold, above mega threshold.
        assigns[f"system_computeUnits_0_core_vectorCoalescer_t{i}"] = BinOp(
            "&", BinOp("&", Id("en"), Const(1, 1)), Const(1, 1)
        )
        then_assigns.append(
            NbAssign(f"r{i}", Id(f"system_computeUnits_0_core_vectorCoalescer_t{i}"))
        )
    body = [If(Id("en"), then_assigns, [NbAssign("z", Const(0, 1))])]
    cached: set[str] = set()
    _promote_bulky_gated_cones(body, assigns, cached, stop={"en"})
    assert "system_computeUnits_0_core_vectorCoalescer_t0" in cached
    assert len(cached) >= 64


def test_extract_deep_mux_chunks() -> None:
    from flashsim.emit import (
        _DEEP_MUX_CHUNK,
        _DEEP_MUX_SPLIT,
        _extract_deep_mux_chunks,
        _right_ternary_arms,
    )
    from flashsim.ir import Const, Id, NbAssign, Signal, Ternary

    # Build a right-nested priority mux deeper than the split threshold.
    n = _DEEP_MUX_SPLIT + _DEEP_MUX_CHUNK
    expr: object = Const(0, 8)
    for i in reversed(range(n)):
        expr = Ternary(Id(f"c{i}"), Const(i & 0xFF, 8), expr)  # type: ignore[arg-type]
    root = "system_commandRouter_completions_t0"
    assigns = {root: expr}
    sigs = {
        root: Signal(name=root, width=8, kind="wire"),
        **{f"c{i}": Signal(name=f"c{i}", width=1, kind="wire") for i in range(n)},
    }
    for i in range(n):
        # Const arms need no extra sigs; conditions are Ids already in sigs.
        pass
    cached = {root}
    n_new, _ = _extract_deep_mux_chunks(assigns, sigs, cached)
    assert n_new >= 2
    assert isinstance(assigns[root], Id)
    top = assigns[root].name
    assert top in cached and "_dmux" in top
    # Top chunk should else-demand another dmux (early arms skip later evals).
    arms, default = _right_ternary_arms(assigns[top])
    assert len(arms) <= _DEEP_MUX_CHUNK
    assert isinstance(default, Id) and default.name in cached


def test_extract_deep_mux_from_nba_rhs() -> None:
    """Mega coalescer-style NBA RHS (not a cached wire) also gets dmux chunks."""
    from flashsim.emit import (
        _DEEP_MUX_CHUNK,
        _DEEP_MUX_SPLIT,
        _extract_deep_mux_chunks,
        _right_ternary_arms,
    )
    from flashsim.ir import Const, Id, NbAssign, Signal, Ternary

    n = _DEEP_MUX_SPLIT + _DEEP_MUX_CHUNK
    expr: object = Const(0, 32)
    for i in reversed(range(n)):
        expr = Ternary(Id(f"c{i}"), Const(i, 32), expr)  # type: ignore[arg-type]
    lhs = "system_computeUnits_0_core_vectorCoalescer_outputBits_readData_0"
    assigns: dict = {}
    sigs = {
        lhs: Signal(name=lhs, width=32, kind="reg"),
        **{f"c{i}": Signal(name=f"c{i}", width=1, kind="wire") for i in range(n)},
    }
    body = [NbAssign(lhs, expr)]  # type: ignore[arg-type]
    cached: set[str] = set()
    n_new, new_body = _extract_deep_mux_chunks(assigns, sigs, cached, body=body)
    assert n_new >= 2 and new_body is not None
    assert isinstance(new_body[0], NbAssign)
    assert isinstance(new_body[0].rhs, Id)
    top = new_body[0].rhs.name
    assert top in cached and "_dmux" in top
    arms, default = _right_ternary_arms(assigns[top])
    assert len(arms) <= _DEEP_MUX_CHUNK


def test_top_gate_demands_skip_nested_ssa() -> None:
    """Hold-entry hoist must not pull cached SSA only used under nested gates."""
    from flashsim.emit import _collect_top_gate_demands
    from flashsim.ir import Id, If, NbAssign

    assigns = {"gate": Id("leaf_g"), "ssa": Id("leaf_s")}
    stmts = [
        If(
            Id("gate"),
            [If(Id("ssa"), [NbAssign("r", Id("ssa"))])],
        )
    ]
    stop = {"leaf_g", "leaf_s", "r"}
    cached = {"gate", "ssa"}
    top = _collect_top_gate_demands(stmts, cached, assigns, stop)
    assert "gate" in top
    assert "ssa" not in top


def test_branch_hoist_shared_ssa() -> None:
    """Then/else arms must not each re-bind the same SSA temp."""
    from flashsim.emit import _emit_nba_tree
    from flashsim.ir import BinOp, Id, If, NbAssign, Signal

    assigns = {"t1": BinOp("&", Id("x"), Id("y"))}
    sigs = {
        "x": Signal(name="x", width=8, kind="wire"),
        "y": Signal(name="y", width=8, kind="wire"),
        "t1": Signal(name="t1", width=8, kind="wire"),
        "r_then": Signal(name="r_then", width=8, kind="reg"),
        "r_else": Signal(name="r_else", width=8, kind="reg"),
        "sel": Signal(name="sel", width=1, kind="wire"),
    }
    stmts = [
        If(
            Id("sel"),
            [NbAssign("r_then", Id("t1"))],
            [NbAssign("r_else", Id("t1"))],
        )
    ]
    lines: list[str] = []
    _emit_nba_tree(stmts, set(), assigns, set(), sigs, lines, 0, set())
    text = "\n".join(lines)
    assert text.count("t1 =") == 1
    assert text.index("t1 =") < text.index("if (sel)")


def test_emit_ternary_else_if_flatten() -> None:
    from flashsim.emit import _emit_assign
    from flashsim.ir import Const, Id, Signal, Ternary

    expr: object = Const(0, 8)
    for i in reversed(range(5)):
        expr = Ternary(Id(f"c{i}"), Const(i, 8), expr)  # type: ignore[arg-type]
    sigs = {
        **{f"c{i}": Signal(name=f"c{i}", width=1, kind="wire") for i in range(5)},
        "out": Signal(name="out", width=8, kind="wire"),
    }
    lines: list[str] = []
    _emit_assign("out", expr, set(), sigs, lines, 2, None, [0])  # type: ignore[arg-type]
    text = "\n".join(lines)
    assert "else if" in text
    assert text.count("else if") >= 3


def test_mem_write_enable_gates_data_evals() -> None:
    """Data/addr skip-evals must sit behind any-enable, not every tick."""
    from flashsim.emit import emit_cpp
    from flashsim.ir import (
        AlwaysFF,
        Assign,
        Const,
        Id,
        MemWrite,
        Module,
        NbAssign,
        Signal,
    )

    mod = Module(
        name="MemGate",
        signals={
            "clock": Signal(name="clock", width=1, kind="input"),
            "en_w": Signal(name="en_w", width=1, kind="wire"),
            "data_w": Signal(name="data_w", width=32, kind="wire"),
            "addr_w": Signal(name="addr_w", width=2, kind="wire"),
            "r": Signal(name="r", width=1, kind="reg"),
            "m": Signal(name="m", width=32, kind="mem", depth=4),
            "out": Signal(name="out", width=32, kind="output"),
        },
        assigns=[
            Assign("en_w", Id("r")),
            Assign("data_w", Id("out")),
            Assign("addr_w", Const(0, 2)),
        ],
        always=AlwaysFF("clock", [NbAssign("r", Id("en_w"))]),
        ports=["clock", "out"],
        mem_writes=[
            MemWrite(mem="m", addr=Id("addr_w"), data=Id("data_w"), enable=Id("en_w")),
        ],
    )
    text = emit_cpp(mod)
    # Enable eval runs unconditionally; data eval only inside any-enable guard.
    assert "uint8_t __we0" in text
    assert "if (__we0) {" in text
    # With a single port the OR-guard collapses to if (__we0) wrapping data.
    # data_w is an output → cached; its eval must not precede the first __we0.
    we = text.index("uint8_t __we0")
    # Find tick_nba body after we0
    rest = text[we:]
    assert "eval_data_w" in rest or "data_w =" in rest or "data_w__ok" in rest
    # The any-enable wrapper precedes data packing / writeback.
    assert rest.count("if (__we0)") >= 2


def test_bump_sig_stable_for_commit_batching() -> None:
    from flashsim import emit as em

    em._LEAF_HOLDS = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [9]}
    leaf_parts = {"a": [10, 11], "b": [10, 11], "c": []}
    assert em._bump_sig("a", leaf_parts) == em._bump_sig("b", leaf_parts)
    assert em._bump_sig("a", leaf_parts) != em._bump_sig("c", leaf_parts)
    lines: list[str] = []
    em._HOLD_WAKE_TABLES = {}
    em._emit_bump_sig((10, 11), (1, 2, 3), {}, lines, 2)
    assert any("_pg[10]++" in ln for ln in lines)
    assert any("_h_need[" in ln for ln in lines)


if __name__ == "__main__":
    test_wake_covers_data_path()
    test_wake_follows_combinational_wires()
    test_wake_covers_both_arms_of_nested_holds()
    test_wake_covers_memory_reads()
    test_wake_uses_cached_wire_deps()
    test_split_dut_methods_out_of_line()
    test_l2_partition_keeps_slice_submodule()
    test_stmt_bucket_key_prefers_write_partition()
    test_promote_large_gpu_ssa()
    test_promote_bulky_gated_cones()
    test_promote_bulky_gated_cones_with_else()
    test_promote_mega_gated_lowers_node_threshold()
    test_extract_deep_mux_chunks()
    test_extract_deep_mux_from_nba_rhs()
    test_top_gate_demands_skip_nested_ssa()
    test_branch_hoist_shared_ssa()
    test_emit_ternary_else_if_flatten()
    test_mem_write_enable_gates_data_evals()
    test_bump_sig_stable_for_commit_batching()
    print("ok")
