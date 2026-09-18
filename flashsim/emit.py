from __future__ import annotations

import re
import sys
import zlib
from collections import defaultdict

from flashsim.ir import (
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    BinOp,
    CMP_OPS,
    Concat,
    Const,
    Expr,
    Extract,
    Id,
    If,
    MemRead,
    Module,
    NbAssign,
    Signal,
    Stmt,
    Ternary,
    UnaryOp,
    expr_ids,
    stmt_writes,
)


def assign_map(mod: Module) -> dict[str, Expr]:
    return {a.lhs: a.rhs for a in mod.assigns}


def cone_leaves(root: str, assigns: dict[str, Expr], stop: set[str]) -> set[str]:
    found: set[str] = set()
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in stop or name not in assigns:
            found.add(name)
            continue
        stack.extend(expr_ids(assigns[name]))
    return found


def collect_regs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "reg"]


def collect_inputs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "input"]


def collect_outputs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "output"]


def collect_mems(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "mem"]


def _is_array(sigs: dict[str, Signal], name: str) -> bool:
    return name in sigs and sigs[name].depth > 0


def seq_skip_roots(body: list[Stmt], gated: bool = False) -> set[str]:
    """Wires that need a skip cache in sequential code.

    Conditions are always cached. Assignment payloads are cached only when the
    write is not behind a hold-style `if (en)` (empty else). Gated payloads are
    emitted as locals in the taken branch so idle cycles skip them entirely.
    """
    names: set[str] = set()
    for stmt in body:
        if isinstance(stmt, If):
            names |= expr_ids(stmt.cond)
            if stmt.else_body:
                names |= seq_skip_roots(stmt.then_body, gated)
                names |= seq_skip_roots(stmt.else_body, gated)
            else:
                names |= seq_skip_roots(stmt.then_body, True)
        elif isinstance(stmt, NbAssign):
            if not gated:
                names |= expr_ids(stmt.rhs)
        else:
            raise TypeError(stmt)
    return names


def always_cond_reads(body: list[Stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in body:
        if isinstance(stmt, If):
            names |= expr_ids(stmt.cond)
            names |= always_cond_reads(stmt.then_body)
            names |= always_cond_reads(stmt.else_body)
    return names


def always_writes(body: list[Stmt]) -> list[str]:
    names: set[str] = set()
    for stmt in body:
        names |= stmt_writes(stmt)
    return sorted(names)


def cached_wires(mod: Module, assigns: dict[str, Expr]) -> set[str]:
    """Wires that need a skip cache: sequential roots, mem enables, and outputs.

    One round of shared-helper promotion: a non-SSA wire used by two skip roots
    gets its own `eval_*`. A fixpoint over SSA temps copies the whole decoder
    net into tens of thousands of methods on GPU-sized designs.

    GPU L2/CU path: also promote SSA temps whose expression trees are huge so
    they are not inlined into a single multi-megabyte `eval_*` (probe DRM
    hotspot). Cap the number promoted to keep method count bounded.

    Hold payloads are normally locals (idle skip). On always-busy draws those
    holds stay live and re-inline thousand-temp cones every cycle — promote
    bulky gated cones into skip-cached evals too.
    """
    names = seq_skip_roots(mod.always.body) | set(collect_outputs(mod))
    for wr in mod.mem_writes:
        names |= expr_ids(wr.enable)
    cached = {n for n in names if n in assigns}
    users: dict[str, int] = defaultdict(int)
    for name in cached:
        for dep in expr_ids(assigns[name]):
            if dep in assigns and dep not in cached:
                users[dep] += 1
    for dep, n in users.items():
        if n >= 2 and not _is_ssa_temp(dep):
            cached.add(dep)
    _promote_large_gpu_ssa(assigns, cached)
    stop = set(collect_regs(mod)) | set(collect_inputs(mod)) | set(collect_mems(mod))
    _promote_bulky_gated_cones(mod.always.body, assigns, cached, stop)
    return cached


# Promote SSA boolean trees this large into their own skip-cached eval.
_LARGE_SSA_NODES = 48
# Hard cap so GpuHostSystemAxi does not grow unbounded method counts.
# Busy CU holds alone can need >1k SSA promotions (vectorCoalescer nests);
# allow enough headroom for several mega-holds.
_LARGE_SSA_PROMOTE_CAP = 16384
# Gated hold RHS whose uncached SSA cone exceeds this gets non-trivial temps
# promoted.
_BULKY_GATED_CONE = 48
# One-hot compares are cheaper as locals; only promote richer SSA temps.
_BULKY_TMP_NODES = 12
# Mega hold arms (merged coalescer cones) are almost all tiny compares — still
# promote mid fan-out temps so skip-cache can sleep across busy hold re-entries.
_MEGA_GATED_CONE = 256
_MEGA_TMP_NODES = 3
_MEGA_REGION_PROMOTE_CAP = 512
# Right-nested priority muxes this deep become separate skip-cached chunk evals
# (commandRouter / coalescer readData). Early arms skip later chunks.
# Coalescer line-match then-arms are ~63-deep; CHUNK must be < that so we get
# ≥2 chunks (else the whole mux stays one inline eval).
_DEEP_MUX_SPLIT = 32
_DEEP_MUX_CHUNK = 16
_DEEP_MUX_CAP = 4096
# Flatten right-nested ternaries to else-if when at least this many arms and
# conditions are pure expressions (no demand stmts between else / if).
_FLAT_TERNARY_ARMS = 3


def _expr_node_count(expr: Expr) -> int:
    if isinstance(expr, (Id, Const)):
        return 1
    if isinstance(expr, UnaryOp):
        return 1 + _expr_node_count(expr.a)
    if isinstance(expr, BinOp):
        return 1 + _expr_node_count(expr.a) + _expr_node_count(expr.b)
    if isinstance(expr, Ternary):
        return (
            1
            + _expr_node_count(expr.cond)
            + _expr_node_count(expr.a)
            + _expr_node_count(expr.b)
        )
    if isinstance(expr, Extract):
        return 1 + _expr_node_count(expr.a)
    if isinstance(expr, Concat):
        return 1 + sum(_expr_node_count(p) for p in expr.parts)
    if isinstance(expr, ArrayGet):
        return 1 + _expr_node_count(expr.arr) + _expr_node_count(expr.index)
    if isinstance(expr, ArrayInject):
        return (
            1
            + _expr_node_count(expr.arr)
            + _expr_node_count(expr.index)
            + _expr_node_count(expr.value)
        )
    if isinstance(expr, ArrayZeros):
        return 1
    if isinstance(expr, MemRead):
        return 1 + _expr_node_count(expr.addr)
    return 1


def _is_gpu_hot_ssa(name: str) -> bool:
    """SSA nets under L2 slices / compute units — DRM probe hot path."""
    return (
        "l2_slices_" in name
        or "computeUnits_" in name
        or "vectorCoalescer_" in name
        or "missEngine_" in name
        or "commandRouter" in name
        or "copyEngine" in name
        or "fillEngine" in name
        or "strided" in name
    )


def _promote_large_gpu_ssa(assigns: dict[str, Expr], cached: set[str]) -> None:
    """Split mega inlined cones by giving large GPU SSA temps their own eval."""
    cands: list[tuple[int, str]] = []
    for name, expr in assigns.items():
        if name in cached or not _is_ssa_temp(name) or not _is_gpu_hot_ssa(name):
            continue
        n = _expr_node_count(expr)
        if n >= _LARGE_SSA_NODES:
            cands.append((n, name))
    cands.sort(reverse=True)
    for _, name in cands[:_LARGE_SSA_PROMOTE_CAP]:
        cached.add(name)


def _promote_bulky_gated_cones(
    body: list[Stmt],
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> None:
    """Cache non-trivial SSA temps from huge hold bodies.

    Tiny one-hot compares stay as locals (cheaper than eval+/__ok). Only temps
    with `_expr_node_count >= _BULKY_TMP_NODES` are promoted, and only from
    regions that would otherwise inline ≥ `_BULKY_GATED_CONE` SSA locals.
    """
    regions: list[tuple[int, list[str]]] = []

    def add_needed(expr: Expr) -> set[str]:
        return set(_uncached_needed(expr, assigns, cached, stop))

    def region_temps(stmts: list[Stmt], min_nodes: int) -> list[str]:
        needed: set[str] = set()

        def gather(ss: list[Stmt]) -> None:
            for stmt in ss:
                if isinstance(stmt, NbAssign):
                    needed.update(add_needed(stmt.rhs))
                elif isinstance(stmt, If):
                    needed.update(add_needed(stmt.cond))
                    gather(stmt.then_body)
                    gather(stmt.else_body)
                else:
                    raise TypeError(stmt)

        gather(stmts)
        out: list[str] = []
        for n in needed:
            if n in cached or not _is_ssa_temp(n) or not _is_gpu_hot_ssa(n):
                continue
            if _expr_node_count(assigns[n]) < min_nodes:
                continue
            out.append(n)
        return out

    def walk(stmts: list[Stmt], gated: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, If):
                def note_arm(arm: list[Stmt]) -> None:
                    raw: set[str] = set()

                    def gather(ss: list[Stmt]) -> None:
                        for s in ss:
                            if isinstance(s, NbAssign):
                                raw.update(add_needed(s.rhs))
                            elif isinstance(s, If):
                                raw.update(add_needed(s.cond))
                                gather(s.then_body)
                                gather(s.else_body)

                    gather(arm)
                    raw_ssa = [
                        n
                        for n in raw
                        if _is_ssa_temp(n) and _is_gpu_hot_ssa(n) and n not in cached
                    ]
                    if len(raw_ssa) >= _BULKY_GATED_CONE:
                        min_nodes = (
                            _MEGA_TMP_NODES
                            if len(raw_ssa) >= _MEGA_GATED_CONE
                            else _BULKY_TMP_NODES
                        )
                        regions.append((len(raw_ssa), region_temps(arm, min_nodes)))

                # Both arms can host mega cones (merged same-cond holds keep
                # else). Promote either arm when it would inline a bulky SSA set.
                note_arm(stmt.then_body)
                if stmt.else_body:
                    note_arm(stmt.else_body)
                walk(stmt.then_body, True)
                if stmt.else_body:
                    walk(stmt.else_body, True)
            elif isinstance(stmt, NbAssign):
                if gated:
                    raw = [
                        n
                        for n in add_needed(stmt.rhs)
                        if _is_ssa_temp(n) and _is_gpu_hot_ssa(n) and n not in cached
                    ]
                    if len(raw) >= _BULKY_GATED_CONE:
                        min_nodes = (
                            _MEGA_TMP_NODES
                            if len(raw) >= _MEGA_GATED_CONE
                            else _BULKY_TMP_NODES
                        )
                        regions.append((len(raw), region_temps([stmt], min_nodes)))
            else:
                raise TypeError(stmt)

    walk(body, False)
    regions.sort(key=lambda rt: rt[0], reverse=True)
    added = 0
    for size, temps in regions:
        # Prefer larger exprs first within a region.
        ranked = sorted(
            temps, key=lambda n: _expr_node_count(assigns[n]), reverse=True
        )
        region_cap = (
            _MEGA_REGION_PROMOTE_CAP if size >= _MEGA_GATED_CONE else len(ranked)
        )
        for tmp in ranked[:region_cap]:
            if added >= _LARGE_SSA_PROMOTE_CAP:
                return
            if tmp in cached:
                continue
            cached.add(tmp)
            added += 1


def _right_ternary_arms(expr: Expr) -> tuple[list[tuple[Expr, Expr]], Expr]:
    """Decompose right-nested `c ? a : (c2 ? a2 : …)` into arms + default."""
    arms: list[tuple[Expr, Expr]] = []
    cur: Expr = expr
    while isinstance(cur, Ternary):
        arms.append((cur.cond, cur.a))
        cur = cur.b
    return arms, cur


def _make_ternary_chain(arms: list[tuple[Expr, Expr]], default: Expr) -> Expr:
    expr = default
    for cond, a in reversed(arms):
        expr = Ternary(cond, a, expr)
    return expr


def _max_ternary_nesting(expr: Expr) -> int:
    if isinstance(expr, Ternary):
        return max(
            _ternary_nesting(expr),
            _max_ternary_nesting(expr.cond),
            _max_ternary_nesting(expr.a),
            _max_ternary_nesting(expr.b),
        )
    if isinstance(expr, UnaryOp):
        return _max_ternary_nesting(expr.a)
    if isinstance(expr, BinOp):
        return max(_max_ternary_nesting(expr.a), _max_ternary_nesting(expr.b))
    if isinstance(expr, Extract):
        return _max_ternary_nesting(expr.a)
    if isinstance(expr, Concat):
        return max((_max_ternary_nesting(p) for p in expr.parts), default=0)
    if isinstance(expr, ArrayGet):
        return max(_max_ternary_nesting(expr.arr), _max_ternary_nesting(expr.index))
    if isinstance(expr, ArrayInject):
        return max(
            _max_ternary_nesting(expr.arr),
            _max_ternary_nesting(expr.index),
            _max_ternary_nesting(expr.value),
        )
    if isinstance(expr, MemRead):
        return _max_ternary_nesting(expr.addr)
    return 0


def _rewrite_deep_mux_expr(
    expr: Expr,
    *,
    prefix: str,
    counter: list[int],
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    cached: set[str],
    budget: list[int],
) -> Expr:
    """Replace deep right-nested ternaries with cached chunk wires."""

    def rec(e: Expr) -> Expr:
        return _rewrite_deep_mux_expr(
            e,
            prefix=prefix,
            counter=counter,
            assigns=assigns,
            sigs=sigs,
            cached=cached,
            budget=budget,
        )

    if isinstance(expr, Ternary):
        arms, default = _right_ternary_arms(expr)
        arms = [(rec(c), rec(a)) for c, a in arms]
        default = rec(default)
        n = len(arms)
        n_chunks = (n + _DEEP_MUX_CHUNK - 1) // _DEEP_MUX_CHUNK
        if n < _DEEP_MUX_SPLIT or n_chunks < 2 or budget[0] < n_chunks:
            return _make_ternary_chain(arms, default)
        next_def = default
        for start in reversed(range(0, n, _DEEP_MUX_CHUNK)):
            chunk_arms = arms[start : start + _DEEP_MUX_CHUNK]
            body = _make_ternary_chain(chunk_arms, next_def)
            tmp = f"{prefix}_dmux{counter[0]}"
            counter[0] += 1
            width = _expr_width(chunk_arms[0][1], sigs)
            assigns[tmp] = body
            sigs[tmp] = Signal(name=tmp, width=width, kind="wire")
            cached.add(tmp)
            budget[0] -= 1
            next_def = Id(tmp)
        return next_def
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, rec(expr.a))
    if isinstance(expr, BinOp):
        return BinOp(expr.op, rec(expr.a), rec(expr.b))
    if isinstance(expr, Extract):
        return Extract(rec(expr.a), expr.low, expr.width)
    if isinstance(expr, Concat):
        return Concat(tuple(rec(p) for p in expr.parts))
    if isinstance(expr, ArrayGet):
        return ArrayGet(rec(expr.arr), rec(expr.index))
    if isinstance(expr, ArrayInject):
        return ArrayInject(rec(expr.arr), rec(expr.index), rec(expr.value))
    if isinstance(expr, MemRead):
        return MemRead(expr.mem, rec(expr.addr))
    return expr


def _extract_deep_mux_chunks(
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    cached: set[str],
    body: list[Stmt] | None = None,
) -> tuple[int, list[Stmt] | None]:
    """Split mega priority-mux evals into skip-cached chunk methods.

    Returns (new_chunk_count, rewritten_body). Early mux arms skip later
    chunk `eval_*` calls (demand sits in the else path). Also rewrites mega
    NBA RHS (coalescer readData Concat trees) which never sit in `cached`.
    """
    budget = [_DEEP_MUX_CAP]
    counter = [0]
    before = budget[0]
    for name in list(cached):
        if name not in assigns:
            continue
        if not _is_gpu_hot_ssa(name):
            continue
        expr = assigns[name]
        if _max_ternary_nesting(expr) < _DEEP_MUX_SPLIT:
            continue
        assigns[name] = _rewrite_deep_mux_expr(
            expr,
            prefix=name,
            counter=counter,
            assigns=assigns,
            sigs=sigs,
            cached=cached,
            budget=budget,
        )

    new_body: list[Stmt] | None = None
    if body is not None:

        def walk(stmts: list[Stmt]) -> list[Stmt]:
            out: list[Stmt] = []
            for stmt in stmts:
                if isinstance(stmt, NbAssign):
                    if (
                        _is_gpu_hot_ssa(stmt.lhs)
                        and _max_ternary_nesting(stmt.rhs) >= _DEEP_MUX_SPLIT
                    ):
                        out.append(
                            NbAssign(
                                stmt.lhs,
                                _rewrite_deep_mux_expr(
                                    stmt.rhs,
                                    prefix=stmt.lhs,
                                    counter=counter,
                                    assigns=assigns,
                                    sigs=sigs,
                                    cached=cached,
                                    budget=budget,
                                ),
                            )
                        )
                    else:
                        out.append(stmt)
                elif isinstance(stmt, If):
                    out.append(
                        If(
                            stmt.cond,
                            walk(stmt.then_body),
                            walk(stmt.else_body),
                        )
                    )
                else:
                    out.append(stmt)
            return out

        new_body = walk(body)
    return before - budget[0], new_body


def _ternary_chain_flat_ok(
    arms: list[tuple[Expr, Expr]], cached: set[str], sigs: dict[str, Signal]
) -> bool:
    """True when else-if flatten is safe (no stmts between else and if)."""
    for cond, _then in arms:
        if _needs_limb_emit(cond, sigs) or is_wide(_expr_width(cond, sigs)):
            return False
        if any(d in cached for d in expr_ids(cond)):
            return False
    return True


def _emit_ternary_priority(
    dest: str,
    expr: Ternary,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
    emit_arm,
) -> None:
    """Emit right-nested ternaries as flat else-if when safe."""
    arms, default = _right_ternary_arms(expr)
    sp = " " * indent
    if len(arms) >= _FLAT_TERNARY_ARMS and _ternary_chain_flat_ok(arms, cached, sigs):
        for i, (cond, then_e) in enumerate(arms):
            _emit_demand(cond, cached, lines, indent)
            kw = "if" if i == 0 else "else if"
            lines.append(
                f"{sp}{kw} ({_cond_code(cond, cached, sigs, lines, indent, scratch)}) {{"
            )
            emit_arm(dest, then_e, indent + 2)
            lines.append(f"{sp}}}")
        lines.append(f"{sp}else {{")
        emit_arm(dest, default, indent + 2)
        lines.append(f"{sp}}}")
        return
    _emit_demand(expr.cond, cached, lines, indent)
    lines.append(
        f"{sp}if ({_cond_code(expr.cond, cached, sigs, lines, indent, scratch)}) {{"
    )
    emit_arm(dest, expr.a, indent + 2)
    lines.append(f"{sp}}} else {{")
    emit_arm(dest, expr.b, indent + 2)
    lines.append(f"{sp}}}")


def _collect_uncached_temps(
    stmts: list[Stmt],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
) -> set[str]:
    """SSA temps that would be bound as locals anywhere in an NBA stmt tree."""
    temps: set[str] = set()

    def note_expr(expr: Expr) -> None:
        temps.update(_uncached_needed(expr, assigns, cached, stop))

    def walk(ss: list[Stmt]) -> None:
        for stmt in ss:
            if isinstance(stmt, NbAssign):
                note_expr(stmt.rhs)
            elif isinstance(stmt, If):
                note_expr(stmt.cond)
                walk(stmt.then_body)
                if stmt.else_body:
                    walk(stmt.else_body)
            else:
                raise TypeError(stmt)

    walk(stmts)
    return temps


def _order_ssa_temps(
    temps: set[str],
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> list[str]:
    """Dependency order for a set of SSA temps (deps before users)."""
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(tmp: str) -> None:
        if tmp in seen or tmp not in temps:
            return
        for dep in _uncached_needed(assigns[tmp], assigns, cached, stop):
            if dep in temps:
                visit(dep)
        seen.add(tmp)
        ordered.append(tmp)

    for tmp in sorted(temps):
        visit(tmp)
    return ordered


def _emit_ssa_assign(
    tmp: str,
    cached: set[str],
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    sp = " " * indent
    tw = sigs[tmp].width if tmp in sigs else 32
    td = sigs[tmp].depth if tmp in sigs else 0
    if td:
        _emit_array_assign(tmp, assigns[tmp], cached, sigs, lines, indent)
    elif is_wide(tw):
        _emit_wide_assign(tmp, assigns[tmp], cached, sigs, lines, indent, scratch)
    elif _needs_limb_emit(assigns[tmp], sigs):
        tmask = mask_expr(tw)
        _emit_assign(tmp, assigns[tmp], cached, sigs, lines, indent, tmask, scratch)
    else:
        tmask = mask_expr(tw)
        rhs = emit_expr(assigns[tmp], sigs)
        if tmask:
            lines.append(f"{sp}{tmp} = ({rhs}) & {tmask};")
        else:
            lines.append(f"{sp}{tmp} = {rhs};")


def _emit_ssa_binding(
    tmp: str,
    cached: set[str],
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int],
    branch_assigned: set[str] | None = None,
) -> None:
    if branch_assigned is not None and tmp in branch_assigned:
        return
    if tmp in declared:
        _emit_ssa_assign(tmp, cached, assigns, sigs, lines, indent, scratch)
        if branch_assigned is not None:
            branch_assigned.add(tmp)
        return
    declared.add(tmp)
    if branch_assigned is not None:
        branch_assigned.add(tmp)
    sp = " " * indent
    tw = sigs[tmp].width if tmp in sigs else 32
    td = sigs[tmp].depth if tmp in sigs else 0
    if td:
        lines.append(f"{sp}{_storage_decl(tmp, tw, td, init=False)}")
        _emit_array_assign(tmp, assigns[tmp], cached, sigs, lines, indent)
    elif is_wide(tw):
        lines.append(f"{sp}{_storage_decl(tmp, tw, init=False)}")
        _emit_wide_assign(tmp, assigns[tmp], cached, sigs, lines, indent, scratch)
    elif _needs_limb_emit(assigns[tmp], sigs):
        tmask = mask_expr(tw)
        lines.append(f"{sp}{c_type(tw)} {tmp};")
        _emit_assign(tmp, assigns[tmp], cached, sigs, lines, indent, tmask, scratch)
    else:
        tmask = mask_expr(tw)
        rhs = emit_expr(assigns[tmp], sigs)
        if tmask:
            lines.append(f"{sp}{c_type(tw)} {tmp} = ({rhs}) & {tmask};")
        else:
            lines.append(f"{sp}{c_type(tw)} {tmp} = {rhs};")


def _emit_branch_hoist(
    hoist: set[str],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int],
    demanded: set[str],
    branch_assigned: set[str] | None = None,
) -> None:
    """Bind SSA temps shared by then/else once before the enclosing if."""
    if not hoist:
        return
    if branch_assigned is None:
        branch_assigned = set()
    order = _order_ssa_temps(hoist, assigns, cached, stop)
    demand: set[str] = set()
    for tmp in order:
        for dep in expr_ids(assigns[tmp]):
            if dep in cached:
                demand.add(dep)
    for dep in sorted(demand):
        if dep in demanded:
            continue
        demanded.add(dep)
        lines.append(_eval_invoke(dep, indent))
    for tmp in order:
        _emit_ssa_binding(
            tmp, cached, assigns, sigs, lines, indent, declared, scratch, branch_assigned
        )


def _collect_top_gate_demands(
    stmts: list[Stmt],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
) -> set[str]:
    """Cached wires needed only by top-level hold stmt gates / bare NBAs.

    Mega coalescer SSA under nested `if (arbiter)` stays out of the hold-entry
    eval list so a false outer gate does not pay hundreds of `__ok` checks;
    `_emit_nba_tree` still CSE-s those evals at first use inside the arm.
    """
    demand: set[str] = set()

    def note_expr(expr: Expr) -> None:
        for dep in expr_ids(expr):
            if dep in cached:
                demand.add(dep)
        for tmp in _uncached_needed(expr, assigns, cached, stop):
            for dep in expr_ids(assigns[tmp]):
                if dep in cached:
                    demand.add(dep)

    for stmt in stmts:
        if isinstance(stmt, NbAssign):
            note_expr(stmt.rhs)
        elif isinstance(stmt, If):
            note_expr(stmt.cond)
        else:
            raise TypeError(stmt)
    return demand


def internal_cone(
    root: str,
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> list[str]:
    """Topological internals used only to compute `root`."""
    needed: list[str] = []
    seen: set[str] = set()
    stack: list[tuple[str, bool]] = [(root, False)]
    while stack:
        name, expanded = stack.pop()
        if expanded:
            if name != root:
                needed.append(name)
            continue
        if name in seen or name in stop or name not in assigns:
            continue
        if name != root and name in cached:
            continue
        seen.add(name)
        stack.append((name, True))
        for dep in expr_ids(assigns[name]):
            stack.append((dep, False))
    return needed


def cached_deps(root: str, assigns: dict[str, Expr], cached: set[str], stop: set[str]) -> list[str]:
    deps: list[str] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen or name in stop or name not in assigns:
            continue
        seen.add(name)
        if name != root and name in cached:
            deps.append(name)
            continue
        stack.extend(expr_ids(assigns[name]))
    return deps


def _is_ssa_temp(name: str) -> bool:
    toks = name.split("_")
    return bool(toks) and len(toks[-1]) > 1 and toks[-1][0] == "t" and toks[-1][1:].isdigit()


def skip_partition_key(name: str) -> str:
    """Coarsen skip activity the way ESSENT coarsens CCSS partitions.

    Per-wire invalidation stores explode on GPU-sized nets. Wires that share a
    Chisel instance prefix share one generation counter, so a leaf change bumps
    O(partitions) instead of O(wires). Strip SSA temps (`t12`) first so a
    decoder helper does not get its own partition.

    L2 slice SSA nets are special: after stripping `tN`, ~10k eval caches share
    a single `l2_slices_N` generation, so any slice activity re-evaluates the
    whole cone (dominant cost on the ARTI DRM path). Named submodules keep a
    4-token key; pure-SSA names under a slice are hashed into 64 buckets so
    related temps can skip independently without one part per wire.
    """
    toks = name.split("_")
    while toks and _is_ssa_temp("_".join(toks)):
        toks.pop()
    if toks and toks[0] in {"system", "gpu"}:
        toks = toks[1:]
        while toks and _is_ssa_temp("_".join(toks)):
            toks.pop()
    if not toks:
        return name
    # vectorCoalescer / fmaAlu holds were one multi-100KB method each because
    # every reg shared `computeUnits_N_core_vectorCoalescer`. Bucket like L2
    # SSA so idle subtrees can sleep independently on the busy CU path.
    if (
        len(toks) >= 4
        and toks[0] == "computeUnits"
        and toks[2] == "core"
        and toks[3] == "vectorCoalescer"
    ):
        m = re.search(r"_t(\d+)$", name)
        b = int(m.group(1)) % 32 if m else (zlib.adler32(name.encode()) & 31)
        return f"computeUnits_{toks[1]}_core_vectorCoalescer_b{b}"
    if (
        len(toks) >= 5
        and toks[0] == "computeUnits"
        and toks[2] == "core"
        and toks[3] == "vector"
        and toks[4] == "fmaAlu"
    ):
        lane = toks[6] if len(toks) > 6 and toks[5] == "lanes" else "x"
        m = re.search(r"_t(\d+)$", name)
        b = int(m.group(1)) % 16 if m else (zlib.adler32(name.encode()) & 15)
        return f"computeUnits_{toks[1]}_core_vector_fmaAlu_l{lane}_b{b}"
    if toks[0] == "computeUnits" and len(toks) >= 7:
        return "_".join(toks[:7])
    if toks[0] == "l2":
        if len(toks) >= 4:
            return "_".join(toks[:4])
        # Pure `l2_slices_N` after SSA strip — bucket by the original temp id.
        # 64 buckets: enough to split the ~10k SSA caches without exploding
        # per-leaf invalidation fanout (256 buckets regressed probe wall clock).
        if len(toks) == 3 and toks[1] == "slices" and toks[2].isdigit():
            m = re.search(r"_t(\d+)$", name)
            if m:
                return f"l2_slices_{toks[2]}_b{int(m.group(1)) % 64}"
        return "_".join(toks)
    # Host AXI response fanout: keep bits/ready/valid in separate gens so a
    # single leaf bump does not invalidate the whole `memoryAxi_io_response`
    # family (tiny evals called from nearly every NBA shard).
    if toks[0] == "memoryAxi" and len(toks) >= 4:
        return "_".join(toks[:4])
    # Compute-unit pipeline stages: keep CU index + first submodule so a
    # scoreboard leaf does not invalidate the whole CU SSA fanout.
    if toks[0] == "computeUnits" and len(toks) >= 4:
        return "_".join(toks[:4])
    if len(toks) >= 3:
        return "_".join(toks[:3])
    return "_".join(toks)


def _partition_maps(
    cached: set[str], wire_deps: dict[str, set[str]]
) -> tuple[dict[str, int], dict[str, list[int]], int]:
    if len(cached) <= 128:
        uniq_sets = sorted(
            {frozenset(deps) for deps in wire_deps.values()},
            key=lambda s: tuple(sorted(s)),
        )
        part_id = {s: i for i, s in enumerate(uniq_sets)}
        wire_part = {w: part_id[frozenset(wire_deps[w])] for w in cached}
    else:
        keys = sorted({skip_partition_key(w) for w in cached})
        part_of = {k: i for i, k in enumerate(keys)}
        wire_part = {w: part_of[skip_partition_key(w)] for w in cached}
    leaf_parts: dict[str, set[int]] = defaultdict(set)
    for w, deps in wire_deps.items():
        p = wire_part[w]
        for leaf in deps:
            leaf_parts[leaf].add(p)
    return wire_part, {k: sorted(v) for k, v in leaf_parts.items()}, max(wire_part.values(), default=-1) + 1


_BUMP_INLINE = 6


def _invalidation_tables(leaf_parts: dict[str, list[int]]) -> dict[tuple[int, ...], int]:
    tables: dict[tuple[int, ...], int] = {}
    for parts in leaf_parts.values():
        key = tuple(parts)
        if len(key) > _BUMP_INLINE and key not in tables:
            tables[key] = len(tables)
    return tables


def _cond_stop_leaves(
    cond: Expr,
    assigns: dict[str, Expr],
    stop: set[str],
    wire_deps: dict[str, set[str]],
) -> set[str]:
    leaves: set[str] = set()
    for name in expr_ids(cond):
        if name in stop or name not in assigns:
            leaves.add(name)
        elif name in wire_deps:
            leaves |= wire_deps[name]
        else:
            leaves |= cone_leaves(name, assigns, stop)
    return leaves


def _stmt_stop_leaves(
    stmt: Stmt,
    assigns: dict[str, Expr],
    stop: set[str],
    wire_deps: dict[str, set[str]],
) -> set[str]:
    """Leaves that can change what a hold statement does, data path included.

    Waking on condition leaves alone is unsound: `_h_busy` only survives while
    the body keeps changing a value, so a hold whose next value happens to
    repeat for one cycle goes to sleep and then misses every later data-path
    edge under a steady condition.
    """
    leaves: set[str] = set()

    def walk(stmts: list[Stmt]) -> None:
        for s in stmts:
            if isinstance(s, NbAssign):
                leaves.update(_cond_stop_leaves(s.rhs, assigns, stop, wire_deps))
            elif isinstance(s, If):
                leaves.update(_cond_stop_leaves(s.cond, assigns, stop, wire_deps))
                walk(s.then_body)
                walk(s.else_body)
            else:
                raise TypeError(s)

    walk([stmt])
    return leaves


def _hold_wake_tables(leaf_holds: dict[str, list[int]]) -> dict[tuple[int, ...], int]:
    tables: dict[tuple[int, ...], int] = {}
    for ks in leaf_holds.values():
        key = tuple(ks)
        if len(key) > _BUMP_INLINE and key not in tables:
            tables[key] = len(tables)
    return tables


def _bump_sig(
    leaf: str, leaf_parts: dict[str, list[int]]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Invalidation + wake signature for a committed leaf."""
    return (tuple(leaf_parts.get(leaf, ())), tuple(_LEAF_HOLDS.get(leaf, ())))


def _emit_bump_sig(
    parts: tuple[int, ...],
    holds: tuple[int, ...],
    tables: dict[tuple[int, ...], int],
    lines: list[str],
    indent: int,
) -> None:
    sp = " " * indent
    if parts:
        if len(parts) <= _BUMP_INLINE:
            for p in parts:
                lines.append(f"{sp}_pg[{p}]++;")
        else:
            i = tables[parts]
            lines.append(f"{sp}fs_bump(_pg, _inv{i}, {len(parts)}u);")
    if not holds:
        return
    if len(holds) <= _BUMP_INLINE:
        for k in holds:
            word, bit = k >> 6, k & 63
            lines.append(f"{sp}_h_need[{word}] |= 1ull << {bit};")
        return
    i = _HOLD_WAKE_TABLES[holds]
    lines.append(f"{sp}fs_wake(_h_need, _hw{i}, {len(holds)}u);")


def _bump_parts(
    leaf: str,
    leaf_parts: dict[str, list[int]],
    tables: dict[tuple[int, ...], int],
    lines: list[str],
    indent: int,
) -> None:
    parts, holds = _bump_sig(leaf, leaf_parts)
    _emit_bump_sig(parts, holds, tables, lines, indent)


_WIRE_PART: dict[str, int] = {}
_WRITE_INDEX: dict[str, int] = {}
_WRITE_WIDE: set[str] = set()
_LEAF_HOLDS: dict[str, list[int]] = {}
_HOLD_WAKE_TABLES: dict[tuple[int, ...], int] = {}
_ARRAY_INDEX: dict[str, int] = {}
_HOLD_TAKEN = ""
_LARGE = False
_LARGE_WRITES = 256
_HOLD_GROUP = 24


def _eval_invoke(name: str, indent: int) -> str:
    sp = " " * indent
    p = _WIRE_PART.get(name)
    if p is None:
        return f"{sp}eval_{name}();"
    return f"{sp}if ({name}__ok != _pg[{p}]) eval_{name}();"


def _mark_write(name: str, lines: list[str], indent: int) -> None:
    sp = " " * indent
    extra = f" {_HOLD_TAKEN}" if _HOLD_TAKEN else ""
    if _LARGE:
        idx = _WRITE_INDEX.get(name)
        if idx is None:
            return
        if name in _WRITE_WIDE:
            lines.append(
                f"{sp}if (memcmp({name}__n, {name}, sizeof({name}))) {{ _note({idx});{extra} }}"
            )
        else:
            lines.append(f"{sp}if ({name}__n != {name}) {{ _note({idx});{extra} }}")
        return
    lines.append(f"{sp}{name}__w = 1;")


def _stmt_names(stmt: Stmt) -> set[str]:
    acc: list[str] = []

    def walk(ss: list[Stmt]) -> None:
        for s in ss:
            if isinstance(s, NbAssign):
                acc.append(s.lhs)
            elif isinstance(s, If):
                acc.extend(expr_ids(s.cond))
                walk(s.then_body)
                walk(s.else_body)

    if isinstance(stmt, NbAssign):
        acc.append(stmt.lhs)
    elif isinstance(stmt, If):
        acc.extend(expr_ids(stmt.cond))
        walk(stmt.then_body)
        walk(stmt.else_body)
    return set(acc)


def _stmt_bucket_key(stmt: Stmt) -> str:
    # Bucket by write target, not alphabetically-first wire read in the cond
    # (otherwise vectorTlb holds absorb sharedCachePort/coalescer cones).
    writes = stmt_writes(stmt)
    if writes:
        return max((skip_partition_key(w) for w in writes), key=len)
    names = _stmt_names(stmt)
    return skip_partition_key(sorted(names)[0]) if names else "_"


def c_type(width: int) -> str:
    if width <= 8:
        return "uint8_t"
    if width <= 16:
        return "uint16_t"
    if width <= 32:
        return "uint32_t"
    if width <= 64:
        return "uint64_t"
    if width <= 128:
        return "unsigned __int128"
    raise ValueError(f"width {width} > 128 needs limb storage")


def is_wide(width: int) -> bool:
    return width > 128


def limb_count(width: int) -> int:
    return (width + 63) // 64


def _storage_decl(name: str, width: int, depth: int = 0, init: bool = True) -> str:
    suffix = " = {}" if init else ""
    zero = " = 0" if init else ""
    if is_wide(width):
        n = limb_count(width)
        if depth:
            return f"uint64_t {name}[{depth}][{n}]{suffix};"
        return f"uint64_t {name}[{n}]{suffix};"
    ty = c_type(width)
    if depth:
        return f"{ty} {name}[{depth}]{suffix};"
    return f"{ty} {name}{zero};"


def mask_expr(width: int) -> str | None:
    if width >= 128:
        return None
    if width >= 64:
        if width == 64:
            return None
        return f"((((unsigned __int128)1) << {width}) - 1)"
    return hex((1 << width) - 1)


def _emit_const(value: int, width: int) -> str:
    value &= (1 << width) - 1 if width < 128 else (1 << 128) - 1
    if width <= 32:
        return f"{value}u"
    if width <= 64:
        return f"{value}ull"
    lo = value & ((1 << 64) - 1)
    hi = value >> 64
    if hi == 0:
        return f"((unsigned __int128){lo}ull)"
    return f"((((unsigned __int128){hi}ull) << 64) | {lo}ull)"


def _zero_lit(width: int) -> str:
    if width <= 32:
        return "0u"
    if width <= 64:
        return "0ull"
    return "(unsigned __int128)0"


def _shift_call(op: str, src: str, amt: str, width: int) -> str:
    if width <= 32:
        fn = "shl" if op == "<<" else "shr"
    elif width <= 64:
        fn = "shl64" if op == "<<" else "shr64"
    else:
        fn = "shl128" if op == "<<" else "shr128"
    return f"{fn}({src}, {amt})"


def _signed_cast(code: str, width: int) -> str:
    sign = 1 << (width - 1)
    if width <= 64:
        sl = f"{sign}ull"
        return f"((int64_t)(((uint64_t)({code})) ^ {sl}) - (int64_t){sl})"
    sl = _emit_const(sign, 128)
    return (
        f"((__int128)(((unsigned __int128)({code})) ^ {sl}) - (__int128){sl})"
    )


def emit_expr(expr: Expr, sigs: dict[str, Signal]) -> str:
    if isinstance(expr, Id):
        return expr.name
    if isinstance(expr, Const):
        return _emit_const(expr.value, expr.width)
    if isinstance(expr, UnaryOp):
        inner = emit_expr(expr.a, sigs)
        width = _expr_width(expr, sigs)
        if expr.op in {"~", "!"} and width == 1:
            return f"(!{inner})"
        code = f"({expr.op}{inner})"
        mask = mask_expr(width)
        return f"({code} & {mask})" if mask else code
    if isinstance(expr, BinOp):
        if expr.op == "<<":
            w = _expr_width(expr.a, sigs)
            return _shift_call("<<", emit_expr(expr.a, sigs), emit_expr(expr.b, sigs), w)
        if expr.op == ">>":
            w = _expr_width(expr.a, sigs)
            return _shift_call(">>", emit_expr(expr.a, sigs), emit_expr(expr.b, sigs), w)
        if expr.op in {"s<", "s<=", "s>", "s>="}:
            op = {"s<": "<", "s<=": "<=", "s>": ">", "s>=": ">="}[expr.op]
            w = _expr_width(expr.a, sigs)
            left = _signed_cast(emit_expr(expr.a, sigs), w)
            right = _signed_cast(emit_expr(expr.b, sigs), w)
            return f"({left} {op} {right})"
        if expr.op in {"s/", "s%"}:
            op = {"s/": "/", "s%": "%"}[expr.op]
            w = _expr_width(expr.a, sigs)
            left = _signed_cast(emit_expr(expr.a, sigs), w)
            right = _signed_cast(emit_expr(expr.b, sigs), w)
            return f"({left} {op} {right})"
        code = f"({emit_expr(expr.a, sigs)} {expr.op} {emit_expr(expr.b, sigs)})"
        if expr.op in {"+", "-", "*"}:
            width = _expr_width(expr, sigs)
            mask = mask_expr(width)
            if mask:
                return f"({code} & {mask})"
        return code
    if isinstance(expr, Ternary):
        return f"({emit_expr(expr.cond, sigs)} ? {emit_expr(expr.a, sigs)} : {emit_expr(expr.b, sigs)})"
    if isinstance(expr, Extract):
        src = emit_expr(expr.a, sigs)
        if isinstance(expr.a, Id) and expr.a.name in sigs and is_wide(sigs[expr.a.name].width):
            return f"fs_getbits({expr.a.name}, {expr.low}u, {expr.width}u)"
        mask = mask_expr(expr.width)
        if expr.low:
            src_w = _expr_width(expr.a, sigs)
            shifted = _shift_call(">>", src, f"{expr.low}u", src_w)
        else:
            shifted = src
        return f"(({shifted}) & {mask})" if mask else f"({shifted})"
    if isinstance(expr, Concat):
        total = sum(_expr_width(part, sigs) for part in expr.parts)
        acc = _zero_lit(total)
        for part in expr.parts:
            width = _expr_width(part, sigs)
            piece = emit_expr(part, sigs)
            mask = mask_expr(width)
            if mask:
                piece = f"(({piece}) & {mask})"
            acc = f"(({acc} << {width}) | {piece})"
        return acc
    if isinstance(expr, ArrayGet):
        arr = emit_expr(expr.arr, sigs)
        idx = emit_expr(expr.index, sigs)
        depth = _array_depth(expr.arr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        return f"{arr}[{idx}]"
    if isinstance(expr, MemRead):
        depth = sigs[expr.mem].depth
        idx = emit_expr(expr.addr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        return f"{expr.mem}[{idx}]"
    raise TypeError(type(expr))


def _expr_width(expr: Expr, sigs: dict[str, Signal]) -> int:
    if isinstance(expr, Const):
        return expr.width
    if isinstance(expr, Id):
        return sigs[expr.name].width
    if isinstance(expr, Extract):
        return expr.width
    if isinstance(expr, Concat):
        return sum(_expr_width(part, sigs) for part in expr.parts)
    if isinstance(expr, UnaryOp):
        return _expr_width(expr.a, sigs)
    if isinstance(expr, BinOp):
        if expr.op in CMP_OPS:
            return 1
        if expr.op in {"<<", ">>"}:
            return _expr_width(expr.a, sigs)
        return max(_expr_width(expr.a, sigs), _expr_width(expr.b, sigs))
    if isinstance(expr, Ternary):
        return _expr_width(expr.a, sigs)
    if isinstance(expr, ArrayGet):
        if isinstance(expr.arr, Id):
            return sigs[expr.arr.name].width
        return _expr_width(expr.arr, sigs)
    if isinstance(expr, MemRead):
        return sigs[expr.mem].width
    raise TypeError(type(expr))


def _array_depth(expr: Expr, sigs: dict[str, Signal]) -> int:
    if isinstance(expr, Id):
        return sigs[expr.name].depth
    if isinstance(expr, ArrayInject) or isinstance(expr, ArrayZeros):
        if isinstance(expr, ArrayZeros):
            return expr.depth
        return _array_depth(expr.arr, sigs)
    return 0


def emit_cpp(mod: Module) -> str:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    assigns = assign_map(mod)
    regs = collect_regs(mod)
    mems = collect_mems(mod)
    inputs = [n for n in collect_inputs(mod) if n != mod.always.clock]
    outputs = collect_outputs(mod)
    sigs = mod.signals
    stop = set(regs) | set(collect_inputs(mod)) | set(mems)
    writes = always_writes(mod.always.body)
    cached = cached_wires(mod, assigns)
    _, always_body = _extract_deep_mux_chunks(
        assigns, sigs, cached, body=mod.always.body
    )
    if always_body is None:
        always_body = mod.always.body
    wire_deps = {w: cone_leaves(w, assigns, stop) for w in cached}
    wire_part, leaf_parts, nparts = _partition_maps(cached, wire_deps)
    inv_tables = _invalidation_tables(leaf_parts)
    write_flags = [n for n in writes if not sigs[n].depth]
    large = len(writes) >= _LARGE_WRITES
    hold_buckets: list[tuple[str, list[Stmt]]] = []
    live_stmts: list[Stmt] = []
    leaf_holds: dict[str, list[int]] = {}
    if large:
        # Any statement can be skipped, not just holds: if nothing it reads
        # changed since it last ran, re-running it would recommit the value
        # already in the register. Statements with an else arm behave the
        # same way, so they are bucketed too.
        grouped: dict[str, list[Stmt]] = defaultdict(list)
        for stmt in always_body:
            grouped[_stmt_bucket_key(stmt)].append(stmt)
        hold_buckets = []
        for key, stmts in sorted(grouped.items()):
            if len(stmts) <= _HOLD_GROUP:
                hold_buckets.append((key, stmts))
                continue
            for off in range(0, len(stmts), _HOLD_GROUP):
                hold_buckets.append((f"{key}#{off // _HOLD_GROUP}", stmts[off : off + _HOLD_GROUP]))
        holds_map: dict[str, set[int]] = defaultdict(set)
        for i, (_key, stmts) in enumerate(hold_buckets):
            for stmt in stmts:
                for leaf in _stmt_stop_leaves(stmt, assigns, stop, wire_deps):
                    holds_map[leaf].add(i)
        leaf_holds = {k: sorted(v) for k, v in holds_map.items()}
    used_inputs = {
        name for name in inputs if name in leaf_parts or name in leaf_holds
    }
    array_writes = [n for n in writes if sigs[n].depth]
    global _WIRE_PART, _WRITE_INDEX, _WRITE_WIDE, _LEAF_HOLDS, _HOLD_WAKE_TABLES, _ARRAY_INDEX, _HOLD_TAKEN, _LARGE
    _WIRE_PART = wire_part
    _WRITE_INDEX = {n: i for i, n in enumerate(write_flags)} if large else {}
    _WRITE_WIDE = {n for n in write_flags if is_wide(sigs[n].width)} if large else set()
    _LEAF_HOLDS = leaf_holds
    _HOLD_WAKE_TABLES = _hold_wake_tables(leaf_holds)
    _ARRAY_INDEX = {n: i for i, n in enumerate(array_writes)} if large else {}
    _LARGE = large

    lines: list[str] = [
        "#include <cstdint>",
        "#include <cstring>",
        "",
        "static inline uint32_t shl(uint32_t x, uint32_t n) {",
        "  return n >= 32u ? 0u : (x << n);",
        "}",
        "static inline uint32_t shr(uint32_t x, uint32_t n) {",
        "  return n >= 32u ? 0u : (x >> n);",
        "}",
        "static inline uint64_t shl64(uint64_t x, uint32_t n) {",
        "  return n >= 64u ? 0ull : (x << n);",
        "}",
        "static inline uint64_t shr64(uint64_t x, uint32_t n) {",
        "  return n >= 64u ? 0ull : (x >> n);",
        "}",
        "static inline unsigned __int128 shl128(unsigned __int128 x, uint32_t n) {",
        "  return n >= 128u ? (unsigned __int128)0 : (x << n);",
        "}",
        "static inline unsigned __int128 shr128(unsigned __int128 x, uint32_t n) {",
        "  return n >= 128u ? (unsigned __int128)0 : (x >> n);",
        "}",
        "static inline uint64_t fs_getbits(const uint64_t *w, unsigned low, unsigned width) {",
        "  unsigned li = low / 64u;",
        "  unsigned off = low % 64u;",
        "  uint64_t v = w[li] >> off;",
        "  if (off && (off + width > 64u)) v |= w[li + 1] << (64u - off);",
        "  return width >= 64u ? v : (v & ((1ull << width) - 1ull));",
        "}",
        "static inline void fs_setbits(uint64_t *w, unsigned low, unsigned width, uint64_t val) {",
        "  unsigned li = low / 64u;",
        "  unsigned off = low % 64u;",
        "  uint64_t mask = width >= 64u ? ~0ull : ((1ull << width) - 1ull);",
        "  val &= mask;",
        "  if (off + width <= 64u) {",
        "    w[li] = (w[li] & ~(mask << off)) | (val << off);",
        "    return;",
        "  }",
        "  unsigned lo_bits = 64u - off;",
        "  uint64_t lo_mask = (1ull << lo_bits) - 1ull;",
        "  w[li] = (w[li] & ~(lo_mask << off)) | ((val & lo_mask) << off);",
        "  unsigned hi_bits = width - lo_bits;",
        "  uint64_t hi_mask = (1ull << hi_bits) - 1ull;",
        "  w[li + 1] = (w[li + 1] & ~hi_mask) | (val >> lo_bits);",
        "}",
        "static inline void fs_copy_bits(uint64_t *d, unsigned dlow, const uint64_t *s, unsigned slow, unsigned width) {",
        "  unsigned done = 0;",
        "  while (done < width) {",
        "    unsigned chunk = width - done;",
        "    if (chunk > 64u) chunk = 64u;",
        "    fs_setbits(d, dlow + done, chunk, fs_getbits(s, slow + done, chunk));",
        "    done += chunk;",
        "  }",
        "}",
        "static inline void fs_shl_limbs(uint64_t *d, const uint64_t *s, unsigned n, unsigned sh) {",
        "  memset(d, 0, n * sizeof(uint64_t));",
        "  if (sh >= n * 64u) return;",
        "  unsigned w = sh / 64u, b = sh % 64u;",
        "  if (b == 0) {",
        "    for (unsigned i = 0; i + w < n; i++) d[i + w] = s[i];",
        "    return;",
        "  }",
        "  for (unsigned i = 0; i + w < n; i++) {",
        "    d[i + w] |= s[i] << b;",
        "    if (i + w + 1u < n) d[i + w + 1u] |= s[i] >> (64u - b);",
        "  }",
        "}",
        "static inline void fs_shr_limbs(uint64_t *d, const uint64_t *s, unsigned n, unsigned sh) {",
        "  memset(d, 0, n * sizeof(uint64_t));",
        "  if (sh >= n * 64u) return;",
        "  unsigned w = sh / 64u, b = sh % 64u;",
        "  if (b == 0) {",
        "    for (unsigned i = 0; i + w < n; i++) d[i] = s[i + w];",
        "    return;",
        "  }",
        "  for (unsigned i = 0; i + w < n; i++) {",
        "    d[i] = s[i + w] >> b;",
        "    if (i + w + 1u < n) d[i] |= s[i + w + 1u] << (64u - b);",
        "  }",
        "}",
        "static inline void fs_and_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] & b[i];",
        "}",
        "static inline void fs_or_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] | b[i];",
        "}",
        "static inline void fs_xor_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] ^ b[i];",
        "}",
        "static inline void fs_add_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  unsigned char c = 0;",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 s = (unsigned __int128)a[i] + b[i] + c;",
        "    d[i] = (uint64_t)s;",
        "    c = (unsigned char)(s >> 64);",
        "  }",
        "}",
        "static inline void fs_sub_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  unsigned char br = 0;",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 x = a[i];",
        "    unsigned __int128 y = (unsigned __int128)b[i] + br;",
        "    br = (unsigned char)(x < y);",
        "    d[i] = (uint64_t)(x - y);",
        "  }",
        "}",
        "static inline void fs_mul_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  uint64_t tmp[64];",
        "  memset(tmp, 0, n * sizeof(uint64_t));",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 carry = 0;",
        "    for (unsigned j = 0; i + j < n; j++) {",
        "      unsigned __int128 t =",
        "          (unsigned __int128)a[i] * b[j] + tmp[i + j] + carry;",
        "      tmp[i + j] = (uint64_t)t;",
        "      carry = t >> 64;",
        "    }",
        "  }",
        "  memcpy(d, tmp, n * sizeof(uint64_t));",
        "}",
        "static inline int fs_ucmp_limbs(const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = n; i-- > 0u; ) {",
        "    if (a[i] > b[i]) return 1;",
        "    if (a[i] < b[i]) return -1;",
        "  }",
        "  return 0;",
        "}",
        "static inline void fs_mask_limbs(uint64_t *d, unsigned n, unsigned width) {",
        "  unsigned full = width / 64u;",
        "  unsigned bits = width % 64u;",
        "  for (unsigned i = full + (bits ? 1u : 0u); i < n; i++) d[i] = 0;",
        "  if (bits && full < n) d[full] &= (1ull << bits) - 1ull;",
        "}",
        "static inline void fs_neg_limbs(uint64_t *d, const uint64_t *a, unsigned n) {",
        "  unsigned char br = 1;",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 t = (unsigned __int128)(uint64_t)(~a[i]) + br;",
        "    d[i] = (uint64_t)t;",
        "    br = (unsigned char)(t >> 64);",
        "  }",
        "}",
        "static inline int fs_zero_limbs(const uint64_t *a, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) if (a[i]) return 0;",
        "  return 1;",
        "}",
        "static inline void fs_divrem_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b,",
        "    unsigned n, unsigned width, int is_signed, int want_quot) {",
        "  uint64_t ua[64], ub[64], rem[64], quot[64];",
        "  memcpy(ua, a, n * sizeof(uint64_t));",
        "  memcpy(ub, b, n * sizeof(uint64_t));",
        "  fs_mask_limbs(ua, n, width);",
        "  fs_mask_limbs(ub, n, width);",
        "  int sa = 0, sb = 0;",
        "  if (is_signed && width) {",
        "    unsigned sbit = width - 1u;",
        "    sa = (int)((ua[sbit / 64u] >> (sbit % 64u)) & 1u);",
        "    sb = (int)((ub[sbit / 64u] >> (sbit % 64u)) & 1u);",
        "    if (sa) { fs_neg_limbs(ua, ua, n); fs_mask_limbs(ua, n, width); }",
        "    if (sb) { fs_neg_limbs(ub, ub, n); fs_mask_limbs(ub, n, width); }",
        "  }",
        "  memset(quot, 0, n * sizeof(uint64_t));",
        "  memset(rem, 0, n * sizeof(uint64_t));",
        "  if (fs_zero_limbs(ub, n)) {",
        "    memset(d, 0xff, n * sizeof(uint64_t));",
        "    fs_mask_limbs(d, n, width);",
        "    return;",
        "  }",
        "  for (unsigned bit = width; bit-- > 0u; ) {",
        "    unsigned char c = (unsigned char)((ua[bit / 64u] >> (bit % 64u)) & 1u);",
        "    for (unsigned i = 0; i < n; i++) {",
        "      unsigned char next = (unsigned char)(rem[i] >> 63);",
        "      rem[i] = (rem[i] << 1) | c;",
        "      c = next;",
        "    }",
        "    if (fs_ucmp_limbs(rem, ub, n) >= 0) {",
        "      fs_sub_limbs(rem, rem, ub, n);",
        "      quot[bit / 64u] |= 1ull << (bit % 64u);",
        "    }",
        "  }",
        "  if (want_quot) {",
        "    if (sa ^ sb) fs_neg_limbs(quot, quot, n);",
        "    fs_mask_limbs(quot, n, width);",
        "    memcpy(d, quot, n * sizeof(uint64_t));",
        "  } else {",
        "    if (sa) fs_neg_limbs(rem, rem, n);",
        "    fs_mask_limbs(rem, n, width);",
        "    memcpy(d, rem, n * sizeof(uint64_t));",
        "  }",
        "}",
        "static inline void fs_bump(uint32_t *pg, const uint16_t *ids, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) pg[ids[i]]++;",
        "}",
        "static inline void fs_wake(uint64_t *need, const uint16_t *ids, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned k = ids[i];",
        "    need[k >> 6] |= 1ull << (k & 63);",
        "  }",
        "}",
    ]
    for key, idx in sorted(inv_tables.items(), key=lambda kv: kv[1]):
        inner = ", ".join(str(p) for p in key)
        lines.append(f"static const uint16_t _inv{idx}[] = {{{inner}}};")
    for key, idx in sorted(_HOLD_WAKE_TABLES.items(), key=lambda kv: kv[1]):
        inner = ", ".join(str(p) for p in key)
        lines.append(f"static const uint16_t _hw{idx}[] = {{{inner}}};")
    lines.extend(
        [
        "",
        f"struct {mod.name}Dut {{",
        ]
    )
    for name in regs:
        sig = sigs[name]
        lines.append(f"  {_storage_decl(name, sig.width, sig.depth)}")
    for name in mems:
        sig = sigs[name]
        lines.append(f"  {_storage_decl(name, sig.width, sig.depth)}")
    for name in inputs:
        width = sigs[name].width
        lines.append(f"  {_storage_decl(name, width)}")
        if name in used_inputs:
            lines.append(f"  {_storage_decl(name + '__prev', width)}")
    for name in sorted(cached):
        sig = sigs.get(name)
        width = sig.width if sig else 32
        depth = sig.depth if sig else 0
        lines.append(f"  {_storage_decl(name, width, depth)}")
        lines.append(f"  uint32_t {name}__ok = 0;")
    if nparts:
        lines.append(f"  uint32_t _pg[{nparts}];")
        lines.append(f"  uint32_t _seq_seen[{nparts}] = {{}};")
    if large:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"  {_storage_decl(name + '__n', sig.width, sig.depth)}")
            else:
                lines.append(f"  {_storage_decl(name + '__n', sig.width)}")
        if write_flags:
            nwf = len(write_flags)
            lines.append(f"  uint8_t _w[{nwf}] = {{}};")
            lines.append(f"  uint16_t _wl[{nwf}] = {{}};")
            lines.append("  unsigned _nw = 0;")
            lines.append("  void _note(uint16_t i) { if (!_w[i]) { _w[i] = 1; _wl[_nw++] = i; } }")
        if _ARRAY_INDEX:
            lines.append(f"  uint8_t _ac[{len(_ARRAY_INDEX)}] = {{}};")
        if hold_buckets:
            nh = len(hold_buckets)
            nw = (nh + 63) // 64
            busy_init = ", ".join("~0ull" for _ in range(nw))
            lines.append(f"  uint64_t _h_need[{nw}] = {{}};")
            lines.append(f"  uint64_t _h_busy[{nw}] = {{{busy_init}}};")
    lines.append("  uint8_t __inited = 0;")
    # Design-independent activity signal: how many state elements actually
    # changed in the last tick(). Hosts use it to run a model to quiescence
    # without knowing anything about the design's internals.
    lines.append("  uint32_t _chg = 0;")
    lines.append("")
    lines.append("  void poke_inputs() {")
    lines.append("    if (!__inited) {")
    if nparts:
        lines.append(f"      for (unsigned i = 0; i < {nparts}u; i++) _pg[i] = 1;")
    for name in inputs:
        if name not in used_inputs:
            continue
        width = sigs[name].width
        if is_wide(width):
            lines.append(f"      memcpy({name}__prev, {name}, sizeof({name}));")
        else:
            lines.append(f"      {name}__prev = {name};")
    lines.append("      __inited = 1;")
    lines.append("      return;")
    lines.append("    }")
    for name in inputs:
        if name not in used_inputs:
            continue
        width = sigs[name].width
        if is_wide(width):
            lines.append(f"    if (memcmp({name}, {name}__prev, sizeof({name}))) {{")
        else:
            lines.append(f"    if ({name} != {name}__prev) {{")
        _bump_parts(name, leaf_parts, inv_tables, lines, 6)
        if is_wide(width):
            lines.append(f"      memcpy({name}__prev, {name}, sizeof({name}));")
        else:
            lines.append(f"      {name}__prev = {name};")
        lines.append("    }")
    lines.append("  }")
    lines.append("")
    for name in sorted(cached):
        _emit_eval_method(name, assigns, sigs, stop, cached, lines, wire_part[name])
    for name in outputs:
        if name not in cached:
            lines.append(f"  void eval_{name}() {{}}")
            lines.append("")
    if large:
        for i, (_key, stmts) in enumerate(hold_buckets):
            word, bit = i >> 6, i & 63
            mask = f"(1ull << {bit})"
            lines.append(f"  __attribute__((noinline)) void _nba_h{i}() {{")
            lines.append(f"    _h_need[{word}] &= ~{mask};")
            lines.append(f"    _h_busy[{word}] &= ~{mask};")
            _HOLD_TAKEN = f"_h_busy[{word}] |= {mask};"
            # Hoist only top-level gate/NBA demands. Full-tree hoist forced mega
            # coalescer skip-evals even when outer enables were false.
            demanded = _collect_top_gate_demands(stmts, cached, assigns, stop)
            for dep in sorted(demanded):
                lines.append(_eval_invoke(dep, 4))
            _emit_nba_tree(
                stmts,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                4,
                set(),
                [0],
                demanded,
            )
            _HOLD_TAKEN = ""
            lines.append("  }")
            lines.append("")
        if live_stmts:
            lines.append("  __attribute__((noinline)) void _nba_live() {")
            _emit_nba_tree(
                live_stmts, cached, assigns, stop, sigs, lines, 4, set(), [0]
            )
            lines.append("  }")
            lines.append("")
        if write_flags:
            lines.append("  void _commit() {")
            lines.append("    for (unsigned i = 0; i < _nw; i++) {")
            lines.append("      switch (_wl[i]) {")
            for name, idx in _WRITE_INDEX.items():
                sig = sigs[name]
                lines.append(f"      case {idx}u: {{")
                if is_wide(sig.width):
                    lines.append(
                        f"        if (memcmp({name}__n, {name}, sizeof({name}))) {{"
                    )
                    _bump_parts(name, leaf_parts, inv_tables, lines, 10)
                    lines.append("          _chg++;")
                    lines.append(f"          memcpy({name}, {name}__n, sizeof({name}));")
                    lines.append("        }")
                else:
                    lines.append(f"        if ({name}__n != {name}) {{")
                    _bump_parts(name, leaf_parts, inv_tables, lines, 10)
                    lines.append("          _chg++;")
                    lines.append(f"          {name} = {name}__n;")
                    lines.append("        }")
                lines.append("      } break;")
            lines.append("      }")
            lines.append("    }")
            lines.append("  }")
            lines.append("")
    lines.append("  void tick() {")
    lines.append("    _chg = 0;")
    lines.append("    poke_inputs();")
    lines.append("    tick_nba();")
    lines.append("  }")
    lines.append("")
    # Posedge body without re-scanning inputs (hosts may poke first).
    lines.append("  void tick_nba() {")
    lines.append("    _chg = 0;")
    seq_body = live_stmts if large else always_body
    seq_cached = always_cond_reads(seq_body) & cached
    for wr in mod.mem_writes:
        seq_cached |= expr_ids(wr.enable) & cached
    _emit_eval_calls(seq_cached, cached, assigns, stop, lines, indent=4)
    if large:
        if write_flags:
            lines.append("    memset(_w, 0, sizeof(_w));")
            lines.append("    _nw = 0;")
        if _ARRAY_INDEX:
            lines.append("    memset(_ac, 0, sizeof(_ac));")
        if hold_buckets:
            nh = len(hold_buckets)
            cls = f"{mod.name}Dut"
            nw = (nh + 63) // 64
            lines.append(f"    using HoldFn = void ({cls}::*)();")
            lines.append("    static const HoldFn kHolds[] = {")
            for i in range(nh):
                comma = "," if i + 1 < nh else ""
                lines.append(f"      &{cls}::_nba_h{i}{comma}")
            lines.append("    };")
            lines.append(f"    for (unsigned w = 0; w < {nw}u; w++) {{")
            lines.append("      uint64_t bits = _h_need[w] | _h_busy[w];")
            if nh & 63:
                lines.append(f"      if (w == {nw - 1}u) bits &= (1ull << {nh & 63}u) - 1ull;")
            lines.append("      while (bits) {")
            lines.append("        unsigned b = (unsigned)__builtin_ctzll(bits);")
            lines.append("        bits &= bits - 1ull;")
            lines.append("        (this->*kHolds[(w << 6) + b])();")
            lines.append("      }")
            lines.append("    }")
        if live_stmts:
            lines.append("    _nba_live();")
    else:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"    {_storage_decl(name + '__n', sig.width, sig.depth, init=False)}")
                lines.append(f"    memcpy({name}__n, {name}, sizeof({name}__n));")
            elif is_wide(sig.width):
                lines.append(f"    {_storage_decl(name + '__n', sig.width, init=False)}")
                lines.append(f"    uint8_t {name}__w = 0;")
            else:
                lines.append(f"    {c_type(sig.width)} {name}__n;")
                lines.append(f"    uint8_t {name}__w = 0;")
        tick_scratch = [0]
        _emit_nba_tree(
            always_body,
            cached,
            assigns,
            stop,
            sigs,
            lines,
            indent=4,
            declared=set(),
            scratch=tick_scratch,
        )
    tick_scratch = [0]
    # Mem-write path is two-phase: evaluate enable cones every tick (cheap —
    # usually a handful of skip-cached wires), then only if any enable fires
    # pull data/addr cones. Quiet settle ticks otherwise paid hundreds of
    # unused __ok checks + wide pack temps every cycle.
    mem_en_demanded: set[str] = set()
    mem_data_demanded: set[str] = set()
    for wr in mod.mem_writes:
        mem_en_demanded |= _cached_deps_for_expr(wr.enable, assigns, cached, stop)
        for expr in (wr.data, wr.addr):
            mem_data_demanded |= _cached_deps_for_expr(expr, assigns, cached, stop)
    mem_data_demanded -= mem_en_demanded
    mem_all_demanded = mem_en_demanded | mem_data_demanded
    for dep in sorted(mem_en_demanded):
        lines.append(_eval_invoke(dep, 4))
    for i, wr in enumerate(mod.mem_writes):
        w = sigs[wr.mem].width
        _emit_compute(
            wr.enable,
            cached,
            assigns,
            stop,
            sigs,
            lines,
            4,
            set(),
            tick_scratch,
            set(mem_en_demanded),
        )
        lines.append(f"    uint8_t __we{i} = (uint8_t)({emit_expr(wr.enable, sigs)});")
        if is_wide(w):
            lines.append(f"    uint64_t __wd{i}[{limb_count(w)}];")
        else:
            lines.append(f"    {c_type(w)} __wd{i} = 0;")
        lines.append(f"    uint32_t __wa{i} = 0;")
    if mod.mem_writes:
        we_or = " | ".join(f"__we{i}" for i in range(len(mod.mem_writes)))
        lines.append(f"    if ({we_or}) {{")
        for dep in sorted(mem_data_demanded):
            lines.append(_eval_invoke(dep, 6))
        for i, wr in enumerate(mod.mem_writes):
            depth = sigs[wr.mem].depth
            mask = depth - 1 if depth else 0
            w = sigs[wr.mem].width
            lines.append(f"      if (__we{i}) {{")
            decl: set[str] = set()
            _emit_compute(
                wr.data,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                8,
                decl,
                tick_scratch,
                set(mem_all_demanded),
            )
            _emit_compute(
                wr.addr,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                8,
                decl,
                tick_scratch,
                set(mem_all_demanded),
            )
            if is_wide(w):
                _emit_wide_assign(
                    f"__wd{i}", wr.data, cached, sigs, lines, 8, tick_scratch
                )
            else:
                lines.append(f"        __wd{i} = {emit_expr(wr.data, sigs)};")
            lines.append(
                f"        __wa{i} = ({emit_expr(wr.addr, sigs)}) & {mask}u;"
            )
            lines.append("      }")
        lines.append("    }")
    if large:
        if write_flags:
            lines.append("    _commit();")
        for name, idx in _ARRAY_INDEX.items():
            lines.append(
                f"    if (_ac[{idx}] && memcmp({name}__n, {name}, sizeof({name}))) {{"
            )
            _bump_parts(name, leaf_parts, inv_tables, lines, 6)
            lines.append("      _chg++;")
            lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
            lines.append("    }")
    else:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"    if (memcmp({name}__n, {name}, sizeof({name}))) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append("      _chg++;")
                lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
                lines.append("    }")
            elif is_wide(sig.width):
                lines.append(f"    if ({name}__w && memcmp({name}__n, {name}, sizeof({name}))) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append("      _chg++;")
                lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
                lines.append("    }")
            else:
                lines.append(f"    if ({name}__w && {name}__n != {name}) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append("      _chg++;")
                lines.append(f"      {name} = {name}__n;")
                lines.append("    }")
    if mod.mem_writes:
        we_or = " | ".join(f"__we{i}" for i in range(len(mod.mem_writes)))
        lines.append(f"    if ({we_or}) {{")
        for i, wr in enumerate(mod.mem_writes):
            w = sigs[wr.mem].width
            lines.append(f"      if (__we{i}) {{")
            if is_wide(w):
                lines.append(
                    f"        if (memcmp({wr.mem}[__wa{i}], __wd{i}, sizeof(__wd{i}))) {{"
                )
            else:
                lines.append(f"        if ({wr.mem}[__wa{i}] != __wd{i}) {{")
            _bump_parts(wr.mem, leaf_parts, inv_tables, lines, 10)
            lines.append("          _chg++;")
            lines.append("        }")
            if is_wide(w):
                lines.append(
                    f"        memcpy({wr.mem}[__wa{i}], __wd{i}, sizeof(__wd{i}));"
                )
            else:
                lines.append(f"        {wr.mem}[__wa{i}] = __wd{i};")
            lines.append("      }")
        lines.append("    }")
    lines.append("  }")
    lines.append("};")
    lines.append("")
    _WIRE_PART = {}
    _WRITE_INDEX = {}
    _WRITE_WIDE = set()
    _LEAF_HOLDS = {}
    _HOLD_WAKE_TABLES = {}
    _LARGE = False
    return "\n".join(lines) + "\n"


# Methods that stay in the class body even when splitting: tiny helpers called
# from many NBA shards must remain visible without a cross-TU call.
_SPLIT_KEEP_INLINE = frozenset({"_note"})

# Keep tiny eval helpers in the header so cross-shard call sites can inline
# them. Profile of GpuHostSystemAxi: ~14k call sites of a 4-line
# `eval_memoryAxi_io_response_bits_fault` dominate samples via call/ret, not
# the body. Keep the threshold tight: inlining thousands of mid-size helpers
# into `tick_nba` makes clang -O2 hang on dut_0.
_SPLIT_INLINE_MAX_BYTES = 320

# Core control path — always shard 0 so arti_rtl_model's hot loop shares a TU
# with tick/poke when useful for inlining within that shard.
# `_commit` is intentionally NOT in core: the dirty-list switch is tens of
# thousands of lines and clang -O2 on it can hang for hours. It gets its own
# `dut_commit.cpp` translation unit (see split_dut_methods).
_SPLIT_CORE = frozenset({"poke_inputs", "tick", "tick_nba", "_nba_live"})
_SPLIT_OWN_FILE = frozenset({"_commit"})

_METHOD_START_RE = re.compile(
    r"^  ((?:__attribute__\(\(noinline\)\) )?)void ([A-Za-z_][A-Za-z0-9_]*)\(\) \{(})?$"
)


def _shard_index(name: str, n_shards: int) -> int:
    """Stable, well-mixed shard id (avoid little-endian prefix bias on eval_*)."""
    if name in _SPLIT_CORE:
        return 0
    return zlib.adler32(name.encode()) % n_shards


def split_dut_methods(
    cpp: str, n_shards: int = 8
) -> tuple[str, list[tuple[str, str]]]:
    """Move in-class method bodies into `n_shards` out-of-line .cpp files.

    Large FlashSim tops (GpuHostSystemAxi) are a ~300MB translation unit; clang
    -O2 never finishes on that. Splitting eval/NBA methods lets each shard
    compile at -O2 in parallel while arti_rtl_model.cpp stays a thin wrapper.

    Tiny methods (≤ `_SPLIT_INLINE_MAX_BYTES`) stay in the class body so the
    thousands of cross-shard early-return evals do not pay an out-of-line call.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    m = re.search(r"^struct (\w+) \{", cpp, re.M)
    if not m:
        raise ValueError("dut header has no struct")
    cls = m.group(1)
    lines = cpp.splitlines(keepends=True)
    out_lines: list[str] = []
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    own_files: dict[str, list[str]] = {}
    for s in shards:
        s.append(f'#include "dut.h"\n')
        s.append("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        mm = _METHOD_START_RE.match(line.rstrip("\n"))
        if not mm:
            out_lines.append(line)
            i += 1
            continue
        attr, name, empty = mm.group(1), mm.group(2), mm.group(3)
        if name in _SPLIT_KEEP_INLINE:
            out_lines.append(line)
            i += 1
            continue
        # One-line empty body: keep inline (no cross-TU stub tax).
        if empty == "}":
            out_lines.append(line)
            i += 1
            continue
        # Multi-line body: brace-match from this line's `{`
        depth = line.count("{") - line.count("}")
        body_lines = [line]
        i += 1
        while i < len(lines) and depth > 0:
            body_lines.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
        trailing_blank = ""
        if i < len(lines) and lines[i].strip() == "":
            trailing_blank = lines[i]
            i += 1
        body_bytes = sum(len(bl) for bl in body_lines)
        # Tiny helpers: stay in-class (implicit inline across all shards).
        if body_bytes <= _SPLIT_INLINE_MAX_BYTES:
            out_lines.extend(body_lines)
            if trailing_blank:
                out_lines.append(trailing_blank)
            continue
        out_lines.append(f"  {attr}void {name}();\n")
        if trailing_blank:
            out_lines.append(trailing_blank)
        # Convert `  void name() {` / attr form → `void Class::name() {`
        first = body_lines[0]
        first = re.sub(
            r"^  ((?:__attribute__\(\(noinline\)\) )?)void "
            + re.escape(name)
            + r"\(\) \{",
            rf"void {cls}::{name}() {{",
            first,
        )
        # Dedent body by 2 spaces (class indent)
        converted = [first]
        for bl in body_lines[1:]:
            if bl.startswith("  "):
                converted.append(bl[2:])
            else:
                converted.append(bl)
        if name in _SPLIT_OWN_FILE:
            bucket = own_files.setdefault(name, [f'#include "dut.h"\n', "\n"])
            bucket.extend(converted)
            if not converted[-1].endswith("\n"):
                bucket.append("\n")
            bucket.append("\n")
            continue
        shard_i = _shard_index(name, n_shards)
        shards[shard_i].extend(converted)
        if not converted[-1].endswith("\n"):
            shards[shard_i].append("\n")
        shards[shard_i].append("\n")

    header = "".join(out_lines)
    parts = [(f"dut_{k}.cpp", "".join(shards[k])) for k in range(n_shards)]
    for name, body_lines in sorted(own_files.items()):
        parts.append((f"dut_{name.lstrip('_')}.cpp", "".join(body_lines)))
    return header, parts


def _emit_eval_method(
    name: str,
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    stop: set[str],
    cached: set[str],
    lines: list[str],
    part: int,
) -> None:
    expr = assigns[name]
    width = sigs[name].width if name in sigs else 32
    depth = sigs[name].depth if name in sigs else 0
    mask = None if depth else mask_expr(width)
    scratch = [0]
    lines.append(f"  void eval_{name}() {{")
    lines.append(f"    if ({name}__ok == _pg[{part}]) return;")
    for dep in cached_deps(name, assigns, cached, stop):
        lines.append(f"    eval_{dep}();")
    for tmp in internal_cone(name, assigns, cached, stop):
        tw = sigs[tmp].width if tmp in sigs else 32
        td = sigs[tmp].depth if tmp in sigs else 0
        if td:
            lines.append(f"    {_storage_decl(tmp, tw, td, init=False)}")
            _emit_array_assign(tmp, assigns[tmp], cached, sigs, lines, 4)
        elif is_wide(tw):
            lines.append(f"    {_storage_decl(tmp, tw, init=False)}")
            _emit_wide_assign(tmp, assigns[tmp], cached, sigs, lines, 4, scratch)
        elif _needs_limb_emit(assigns[tmp], sigs):
            tmask = mask_expr(tw)
            lines.append(f"    {c_type(tw)} {tmp};")
            _emit_assign(tmp, assigns[tmp], cached, sigs, lines, 4, tmask, scratch)
        else:
            tmask = mask_expr(tw)
            rhs = emit_expr(assigns[tmp], sigs)
            if tmask:
                lines.append(f"    {c_type(tw)} {tmp} = ({rhs}) & {tmask};")
            else:
                lines.append(f"    {c_type(tw)} {tmp} = {rhs};")
    if depth:
        _emit_array_assign(name, expr, cached, sigs, lines, 4)
    elif is_wide(width):
        _emit_wide_assign(name, expr, cached, sigs, lines, 4, scratch)
    else:
        _emit_assign(name, expr, cached, sigs, lines, 4, mask, scratch)
    lines.append(f"    {name}__ok = _pg[{part}];")
    lines.append("  }")
    lines.append("")


def _emit_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    mask: str | None,
    scratch: list[int] | None = None,
) -> None:
    sp = " " * indent
    if scratch is None:
        scratch = [0]
    if _expr_width(expr, sigs) > 128:
        _emit_wide_assign(dest, expr, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, (ArrayInject, ArrayZeros)) or (
        isinstance(expr, Id) and _is_array(sigs, expr.name)
    ):
        _emit_array_assign(dest, expr, cached, sigs, lines, indent)
        return
    if isinstance(expr, Ternary):
        def arm(d: str, e: Expr, ind: int) -> None:
            _emit_assign(d, e, cached, sigs, lines, ind, mask, scratch)

        _emit_ternary_priority(
            dest, expr, cached, sigs, lines, indent, scratch, arm
        )
        return
    if _needs_limb_emit(expr, sigs):
        _emit_narrow_from_wide(
            dest, expr, _expr_width(expr, sigs), cached, sigs, lines, indent, scratch
        )
        if mask:
            lines.append(f"{sp}{dest} &= {mask};")
        return
    _emit_demand(expr, cached, lines, indent)
    rhs = emit_expr(expr, sigs)
    if mask:
        lines.append(f"{sp}{dest} = ({rhs}) & {mask};")
    else:
        lines.append(f"{sp}{dest} = {rhs};")


def _ternary_nesting(expr: Expr) -> int:
    n = 0
    while isinstance(expr, Ternary):
        n += 1
        expr = expr.b
    return n


def _needs_limb_emit(expr: Expr, sigs: dict[str, Signal]) -> bool:
    """True when emit_expr would apply C operators to limb arrays or deep muxes."""
    if isinstance(expr, Const):
        return False
    if isinstance(expr, Id):
        return False
    if isinstance(expr, Extract):
        if isinstance(expr.a, Id) and expr.a.name in sigs and is_wide(sigs[expr.a.name].width):
            return False
        return _expr_width(expr.a, sigs) >= 128 or _needs_limb_emit(expr.a, sigs)
    if isinstance(expr, Concat):
        total = sum(_expr_width(part, sigs) for part in expr.parts)
        if total >= 128:
            return True
        return any(_needs_limb_emit(part, sigs) for part in expr.parts)
    if isinstance(expr, UnaryOp):
        return _expr_width(expr.a, sigs) >= 128 or _needs_limb_emit(expr.a, sigs)
    if isinstance(expr, BinOp):
        if _expr_width(expr.a, sigs) >= 128 or _expr_width(expr.b, sigs) >= 128:
            return True
        return _needs_limb_emit(expr.a, sigs) or _needs_limb_emit(expr.b, sigs)
    if isinstance(expr, Ternary):
        if _ternary_nesting(expr) > 8:
            return True
        return (
            _needs_limb_emit(expr.cond, sigs)
            or _needs_limb_emit(expr.a, sigs)
            or _needs_limb_emit(expr.b, sigs)
        )
    if isinstance(expr, ArrayGet):
        return _needs_limb_emit(expr.arr, sigs) or _needs_limb_emit(expr.index, sigs)
    if isinstance(expr, MemRead):
        return expr.mem in sigs and is_wide(sigs[expr.mem].width)
    return False


def _cond_code(
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int] | None = None,
) -> str:
    """C expression for a 1-bit condition. Wide compares become temps."""
    if scratch is None:
        scratch = [0]
    if _needs_limb_emit(expr, sigs) or is_wide(_expr_width(expr, sigs)):
        name = f"__c{scratch[0]}"
        scratch[0] += 1
        sp = " " * indent
        lines.append(f"{sp}uint8_t {name};")
        _emit_assign(name, expr, cached, sigs, lines, indent, None, scratch)
        return name
    return emit_expr(expr, sigs)


def _load_bits(src: str, low: int, width: int) -> str:
    if width <= 64:
        return f"fs_getbits({src}, {low}u, {width}u)"
    lo = f"fs_getbits({src}, {low}u, 64u)"
    hi = f"fs_getbits({src}, {low + 64}u, {width - 64}u)"
    return f"(((unsigned __int128)({hi}) << 64) | (unsigned __int128)({lo}))"


def _bind_narrow(
    expr: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    if isinstance(expr, Const) and not is_wide(expr.width):
        return _emit_const(expr.value, min(width, 128))
    if isinstance(expr, Id) and expr.name in sigs and not is_wide(sigs[expr.name].width):
        _emit_demand(expr, cached, lines, indent)
        return expr.name
    if not _needs_limb_emit(expr, sigs) and not is_wide(width):
        _emit_demand(expr, cached, lines, indent)
        return f"({emit_expr(expr, sigs)})"
    tmp = f"__ns{scratch[0]}"
    scratch[0] += 1
    lines.append(f"{' ' * indent}{c_type(width)} {tmp};")
    _emit_narrow_from_wide(tmp, expr, width, cached, sigs, lines, indent, scratch)
    return tmp


def _emit_narrow_from_wide(
    dest: str,
    expr: Expr,
    dest_w: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    sp = " " * indent
    ty = c_type(dest_w)
    if isinstance(expr, Ternary):
        def arm(d: str, e: Expr, ind: int) -> None:
            _emit_narrow_from_wide(d, e, dest_w, cached, sigs, lines, ind, scratch)

        _emit_ternary_priority(
            dest, expr, cached, sigs, lines, indent, scratch, arm
        )
        return
    if isinstance(expr, Extract):
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, expr.low, expr.width)};")
            return
        src = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        shifted = _shift_call(">>", src, f"{expr.low}u", aw)
        mask = mask_expr(expr.width)
        lines.append(
            f"{sp}{dest} = ({ty})({shifted} & {mask});" if mask else f"{sp}{dest} = ({ty})({shifted});"
        )
        return
    if isinstance(expr, BinOp) and expr.op in {"<<", ">>"}:
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
            tmp = _wtmp(lines, indent, limb_count(aw), scratch)
            fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
            lines.append(f"{sp}{fn}({tmp}, {src}, {limb_count(aw)}u, {amt});")
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(tmp, 0, dest_w)};")
            return
        src = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}{dest} = ({ty}){_shift_call(expr.op, src, amt, aw)};")
        return
    if isinstance(expr, BinOp) and expr.op in CMP_OPS:
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        if is_wide(aw) or is_wide(bw):
            if expr.op.startswith("s"):
                raise TypeError(f"signed compare of width {max(aw, bw)}")
            left = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            right = _materialize_limbs(expr.b, bw, cached, sigs, lines, indent, scratch)
            n = max(limb_count(aw), limb_count(bw))
            cop = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[expr.op]
            lines.append(
                f"{sp}{dest} = ({ty})(fs_ucmp_limbs({left}, {right}, {n}u) {cop} 0);"
            )
            return
        left = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _bind_narrow(expr.b, bw, cached, sigs, lines, indent, scratch)
        if expr.op.startswith("s"):
            op = {"s<": "<", "s<=": "<=", "s>": ">", "s>=": ">="}[expr.op]
            left = _signed_cast(left, aw)
            right = _signed_cast(right, bw)
            lines.append(f"{sp}{dest} = ({ty})({left} {op} {right});")
            return
        lines.append(f"{sp}{dest} = ({ty})({left} {expr.op} {right});")
        return
    if isinstance(expr, BinOp):
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        if is_wide(aw) or is_wide(bw):
            src = _materialize_limbs(expr, max(aw, bw), cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        left = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _bind_narrow(expr.b, bw, cached, sigs, lines, indent, scratch)
        if expr.op in {"s/", "s%"}:
            op = {"s/": "/", "s%": "%"}[expr.op]
            left = _signed_cast(left, aw)
            right = _signed_cast(right, bw)
            lines.append(f"{sp}{dest} = ({ty})({left} {op} {right});")
            return
        lines.append(f"{sp}{dest} = ({ty})({left} {expr.op} {right});")
        return
    if isinstance(expr, Concat):
        total = _expr_width(expr, sigs)
        if is_wide(total):
            src = _materialize_limbs(expr, total, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        bit = 0
        first = True
        for part in reversed(expr.parts):
            pw = _expr_width(part, sigs)
            val = _bind_narrow(part, pw, cached, sigs, lines, indent, scratch)
            mask = mask_expr(pw)
            piece = f"(({val}) & {mask})" if mask else f"({val})"
            term = f"((({ty}){piece}) << {bit})"
            if first:
                lines.append(f"{sp}{dest} = {term};")
                first = False
            else:
                lines.append(f"{sp}{dest} |= {term};")
            bit += pw
        if first:
            lines.append(f"{sp}{dest} = 0;")
        return
    if isinstance(expr, UnaryOp):
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr, aw, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        inner = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        if expr.op in {"~", "!"} and dest_w == 1:
            lines.append(f"{sp}{dest} = ({ty})(!{inner});")
            return
        mask = mask_expr(dest_w)
        code = f"({expr.op}{inner})"
        lines.append(
            f"{sp}{dest} = ({ty})({code} & {mask});" if mask else f"{sp}{dest} = ({ty}){code};"
        )
        return
    if isinstance(expr, Id) and is_wide(_expr_width(expr, sigs)):
        _emit_demand(expr, cached, lines, indent)
        lines.append(f"{sp}{dest} = ({ty}){_load_bits(expr.name, 0, dest_w)};")
        return
    if isinstance(expr, Const):
        lines.append(f"{sp}{dest} = {_emit_const(expr.value, dest_w)};")
        return
    if isinstance(expr, MemRead) and is_wide(_expr_width(expr, sigs)):
        src = _materialize_limbs(expr, _expr_width(expr, sigs), cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
        return
    _emit_demand(expr, cached, lines, indent)
    lines.append(f"{sp}{dest} = {emit_expr(expr, sigs)};")


def _wtmp(lines: list[str], indent: int, nlimbs: int, scratch: list[int]) -> str:
    i = scratch[0]
    scratch[0] += 1
    name = f"__wl{i}"
    lines.append(f"{' ' * indent}uint64_t {name}[{nlimbs}];")
    return name


def _try_fold_shift_amt(expr: Expr, sigs: dict[str, Signal]) -> str | None:
    if isinstance(expr, Const):
        return f"{expr.value}u"
    if isinstance(expr, Id):
        w = sigs[expr.name].width if expr.name in sigs else 32
        if w <= 32:
            return f"(unsigned)({expr.name})"
        return None
    if not isinstance(expr, Concat):
        return None
    bit = 0
    terms: list[str] = []
    for part in reversed(expr.parts):
        pw = _expr_width(part, sigs)
        if isinstance(part, Const):
            if part.value and bit < 32:
                terms.append(f"({part.value}u << {bit})")
        elif isinstance(part, Id) and pw <= 32:
            terms.append(f"((unsigned)({part.name}) << {bit})")
        else:
            return None
        bit += pw
        if bit >= 32:
            break
    return "(" + " | ".join(terms) + ")" if terms else "0u"


def _emit_shift_amt(
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    folded = _try_fold_shift_amt(expr, sigs)
    if folded is not None:
        _emit_demand(expr, cached, lines, indent)
        return folded
    w = _expr_width(expr, sigs)
    if not is_wide(w):
        _emit_demand(expr, cached, lines, indent)
        return f"(unsigned)({emit_expr(expr, sigs)})"
    tmp = _wtmp(lines, indent, limb_count(w), scratch)
    _emit_wide_assign(tmp, expr, cached, sigs, lines, indent, scratch)
    return f"(unsigned)fs_getbits({tmp}, 0u, 32u)"


def _emit_setbits_scalar(
    dest: str,
    offset: int,
    width: int,
    src_c: str,
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    """Write a ≤128-bit C integer into a limb array. 65–128 bits need two stores."""
    sp = " " * indent
    if width <= 64:
        lines.append(
            f"{sp}fs_setbits({dest}, {offset}u, {width}u, (uint64_t)({src_c}));"
        )
        return
    tmp = f"__i{scratch[0]}"
    scratch[0] += 1
    lines.append(f"{sp}unsigned __int128 {tmp} = (unsigned __int128)({src_c});")
    lines.append(f"{sp}fs_setbits({dest}, {offset}u, 64u, (uint64_t){tmp});")
    lines.append(
        f"{sp}fs_setbits({dest}, {offset + 64}u, {width - 64}u, "
        f"(uint64_t)({tmp} >> 64));"
    )


def _place_bits(
    dest: str,
    offset: int,
    part: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    sp = " " * indent
    if width <= 0:
        return
    if isinstance(part, Const):
        if part.value == 0:
            return
        val = part.value
        done = 0
        while done < width:
            chunk = min(64, width - done)
            chunk_val = (val >> done) & ((1 << chunk) - 1 if chunk < 64 else (1 << 64) - 1)
            lines.append(
                f"{sp}fs_setbits({dest}, {offset + done}u, {chunk}u, {chunk_val}ull);"
            )
            done += chunk
        return
    if isinstance(part, Id) and not is_wide(_expr_width(part, sigs)):
        _emit_demand(part, cached, lines, indent)
        _emit_setbits_scalar(
            dest, offset, width, emit_expr(part, sigs), lines, indent, scratch
        )
        return
    src = _materialize_limbs(part, width, cached, sigs, lines, indent, scratch)
    lines.append(f"{sp}fs_copy_bits({dest}, {offset}u, {src}, 0u, {width}u);")


def _materialize_limbs(
    expr: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    if isinstance(expr, Id) and expr.name in sigs and is_wide(sigs[expr.name].width):
        _emit_demand(expr, cached, lines, indent)
        return expr.name
    tmp = _wtmp(lines, indent, limb_count(width), scratch)
    _emit_wide_assign(tmp, expr, cached, sigs, lines, indent, scratch)
    return tmp


def _emit_wide_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int] | None = None,
) -> None:
    if scratch is None:
        scratch = [0]
    sp = " " * indent
    width = _expr_width(expr, sigs)
    n = limb_count(width)
    if isinstance(expr, Ternary):
        def arm(d: str, e: Expr, ind: int) -> None:
            _emit_wide_assign(d, e, cached, sigs, lines, ind, scratch)

        _emit_ternary_priority(
            dest, expr, cached, sigs, lines, indent, scratch, arm
        )
        return
    lines.append(f"{sp}memset({dest}, 0, sizeof({dest}));")
    if isinstance(expr, Const):
        if expr.value:
            _place_bits(dest, 0, expr, width, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, Id):
        src_w = sigs[expr.name].width if expr.name in sigs else width
        if is_wide(src_w):
            copy_w = min(width, src_w)
            lines.append(f"{sp}fs_copy_bits({dest}, 0u, {expr.name}, 0u, {copy_w}u);")
        else:
            _emit_demand(expr, cached, lines, indent)
            _emit_setbits_scalar(
                dest, 0, min(width, src_w), emit_expr(expr, sigs), lines, indent, scratch
            )
        return
    if isinstance(expr, MemRead):
        _emit_demand(expr, cached, lines, indent)
        depth = sigs[expr.mem].depth
        idx = emit_expr(expr.addr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        mem_w = sigs[expr.mem].width
        if is_wide(mem_w):
            lines.append(f"{sp}memcpy({dest}, {expr.mem}[{idx}], sizeof({dest}));")
        else:
            _emit_setbits_scalar(
                dest, 0, mem_w, f"{expr.mem}[{idx}]", lines, indent, scratch
            )
        return
    if isinstance(expr, Concat):
        _emit_demand(expr, cached, lines, indent)
        offset = width
        for part in expr.parts:
            w = _expr_width(part, sigs)
            offset -= w
            _place_bits(dest, offset, part, w, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, Extract):
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        lines.append(
            f"{sp}fs_copy_bits({dest}, 0u, {src}, {expr.low}u, {expr.width}u);"
        )
        return
    if isinstance(expr, BinOp) and expr.op in {"<<", ">>"}:
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
        src_n = limb_count(src_w)
        if src_n == n and dest != src:
            fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
            lines.append(f"{sp}{fn}({dest}, {src}, {src_n}u, {amt});")
            return
        tmp = _wtmp(lines, indent, src_n, scratch)
        fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
        lines.append(f"{sp}{fn}({tmp}, {src}, {src_n}u, {amt});")
        lines.append(f"{sp}fs_copy_bits({dest}, 0u, {tmp}, 0u, {width}u);")
        return
    if isinstance(expr, BinOp) and expr.op in {"&", "|", "^", "+", "-", "*"}:
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        left = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _materialize_limbs(expr.b, bw, cached, sigs, lines, indent, scratch)
        fn = {
            "&": "fs_and_limbs",
            "|": "fs_or_limbs",
            "^": "fs_xor_limbs",
            "+": "fs_add_limbs",
            "-": "fs_sub_limbs",
            "*": "fs_mul_limbs",
        }[expr.op]
        use_n = max(limb_count(aw), limb_count(bw), n)
        if use_n == n:
            lines.append(f"{sp}{fn}({dest}, {left}, {right}, {n}u);")
        else:
            tmp = _wtmp(lines, indent, use_n, scratch)
            lines.append(f"{sp}{fn}({tmp}, {left}, {right}, {use_n}u);")
            lines.append(f"{sp}fs_copy_bits({dest}, 0u, {tmp}, 0u, {width}u);")
        return
    if isinstance(expr, BinOp) and expr.op in {"/", "s/", "%", "s%"}:
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        left = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _materialize_limbs(expr.b, bw, cached, sigs, lines, indent, scratch)
        use_n = max(limb_count(aw), limb_count(bw), n)
        bit_w = max(aw, bw, width)
        signed = 1 if expr.op.startswith("s") else 0
        want_quot = 1 if expr.op in {"/", "s/"} else 0
        if use_n == n:
            lines.append(
                f"{sp}fs_divrem_limbs({dest}, {left}, {right}, {n}u, {bit_w}u, {signed}, {want_quot});"
            )
        else:
            tmp = _wtmp(lines, indent, use_n, scratch)
            lines.append(
                f"{sp}fs_divrem_limbs({tmp}, {left}, {right}, {use_n}u, {bit_w}u, {signed}, {want_quot});"
            )
            lines.append(f"{sp}fs_copy_bits({dest}, 0u, {tmp}, 0u, {width}u);")
        return
    if isinstance(expr, UnaryOp) and expr.op == "~":
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}for (unsigned __i = 0; __i < {n}u; __i++) {dest}[__i] = ~{src}[__i];")
        return
    if isinstance(expr, ArrayGet):
        _emit_demand(expr, cached, lines, indent)
        arr = emit_expr(expr.arr, sigs)
        idx = emit_expr(expr.index, sigs)
        depth = _array_depth(expr.arr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        elem_w = _expr_width(expr, sigs)
        if is_wide(elem_w):
            lines.append(f"{sp}memcpy({dest}, {arr}[{idx}], sizeof({dest}));")
        else:
            _emit_setbits_scalar(
                dest, 0, elem_w, f"{arr}[{idx}]", lines, indent, scratch
            )
        return
    raise TypeError(f"wide assign {type(expr)} {getattr(expr, 'op', '')}")


def _emit_array_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int] | None = None,
) -> None:
    if scratch is None:
        scratch = [0]
    sp = " " * indent
    if isinstance(expr, Ternary):
        def arm(d: str, e: Expr, ind: int) -> None:
            _emit_array_assign(d, e, cached, sigs, lines, ind, scratch)

        _emit_ternary_priority(
            dest, expr, cached, sigs, lines, indent, scratch, arm
        )
        return
    if isinstance(expr, ArrayZeros):
        lines.append(f"{sp}memset({dest}, 0, sizeof({dest}));")
        return
    if isinstance(expr, Id):
        lines.append(f"{sp}memcpy({dest}, {expr.name}, sizeof({dest}));")
        return
    if isinstance(expr, ArrayInject):
        _emit_array_assign(dest, expr.arr, cached, sigs, lines, indent, scratch)
        _emit_demand(expr.index, cached, lines, indent)
        _emit_demand(expr.value, cached, lines, indent)
        depth = _array_depth(expr, sigs)
        idx = emit_expr(expr.index, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        val_w = _expr_width(expr.value, sigs)
        val = emit_expr(expr.value, sigs)
        if is_wide(val_w):
            lines.append(f"{sp}memcpy({dest}[{idx}], {val}, sizeof({dest}[{idx}]));")
        else:
            lines.append(f"{sp}{dest}[{idx}] = {val};")
        return
    raise TypeError(f"array assign {type(expr)}")


def _emit_eval_calls(
    names: set[str],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    lines: list[str],
    indent: int,
) -> None:
    seen: set[str] = set()
    ordered: list[str] = []
    stack: list[tuple[str, bool]] = [(name, False) for name in reversed(sorted(names))]
    while stack:
        name, expanded = stack.pop()
        if name not in cached:
            continue
        if expanded:
            ordered.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        stack.append((name, True))
        for dep in cached_deps(name, assigns, cached, stop):
            stack.append((dep, False))
    if not _WIRE_PART:
        for name in ordered:
            lines.append(_eval_invoke(name, indent))
        return
    groups: dict[int, list[str]] = {}
    for name in ordered:
        groups.setdefault(_WIRE_PART[name], []).append(name)
    sp = " " * indent
    for p, group in sorted(groups.items()):
        lines.append(f"{sp}if (_seq_seen[{p}] != _pg[{p}]) {{")
        for name in group:
            lines.append(f"{sp}  eval_{name}();")
        lines.append(f"{sp}  _seq_seen[{p}] = _pg[{p}];")
        lines.append(f"{sp}}}")


def _emit_demand(
    expr: Expr,
    cached: set[str],
    lines: list[str],
    indent: int,
    demanded: set[str] | None = None,
) -> None:
    if demanded is None:
        demanded = set()
    seen: set[str] = set()
    for dep in expr_ids(expr):
        if dep in cached and dep not in seen:
            seen.add(dep)
            if dep in demanded:
                continue
            demanded.add(dep)
            lines.append(_eval_invoke(dep, indent))


def _cached_deps_for_expr(
    expr: Expr,
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> set[str]:
    """Skip-cached wires demanded to evaluate `expr` (including via SSA temps)."""
    out: set[str] = set()
    for dep in expr_ids(expr):
        if dep in cached:
            out.add(dep)
    for tmp in _uncached_needed(expr, assigns, cached, stop):
        for dep in expr_ids(assigns[tmp]):
            if dep in cached:
                out.add(dep)
    return out


def _uncached_needed(
    expr: Expr,
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> list[str]:
    needed: list[str] = []
    seen: set[str] = set()
    stack: list[tuple[str, bool]] = [(name, False) for name in expr_ids(expr)]
    while stack:
        name, expanded = stack.pop()
        if expanded:
            needed.append(name)
            continue
        if name in seen or name in stop or name in cached or name not in assigns:
            continue
        seen.add(name)
        stack.append((name, True))
        for dep in expr_ids(assigns[name]):
            stack.append((dep, False))
    return needed


def _emit_compute(
    expr: Expr,
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int] | None = None,
    demanded: set[str] | None = None,
    branch_assigned: set[str] | None = None,
) -> None:
    """Eval cached deps and bind uncached cone wires as locals."""
    if scratch is None:
        scratch = [0]
    if demanded is None:
        demanded = set()
    sp = " " * indent
    needed = _uncached_needed(expr, assigns, cached, stop)
    demand: set[str] = set()
    for dep in expr_ids(expr):
        if dep in cached:
            demand.add(dep)
    for tmp in needed:
        for dep in expr_ids(assigns[tmp]):
            if dep in cached:
                demand.add(dep)
    for dep in sorted(demand):
        if dep in demanded:
            continue
        demanded.add(dep)
        lines.append(_eval_invoke(dep, indent))
    for tmp in needed:
        _emit_ssa_binding(
            tmp, cached, assigns, sigs, lines, indent, declared, scratch, branch_assigned
        )


def _emit_nba_tree(
    stmts: list[Stmt],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int] | None = None,
    demanded: set[str] | None = None,
    branch_assigned: set[str] | None = None,
) -> None:
    if scratch is None:
        scratch = [0]
    if demanded is None:
        demanded = set()
    sp = " " * indent
    for stmt in stmts:
        if isinstance(stmt, NbAssign):
            tmp = f"{stmt.lhs}__n"
            _emit_compute(
                stmt.rhs,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                indent,
                declared,
                scratch,
                demanded,
                branch_assigned,
            )
            if _is_array(sigs, stmt.lhs):
                aidx = _ARRAY_INDEX.get(stmt.lhs)
                if aidx is not None:
                    lines.append(f"{sp}_ac[{aidx}] = 1;")
                    if _HOLD_TAKEN:
                        lines.append(f"{sp}{_HOLD_TAKEN}")
                _emit_array_assign(tmp, stmt.rhs, cached, sigs, lines, indent)
            elif is_wide(sigs[stmt.lhs].width):
                _emit_wide_assign(tmp, stmt.rhs, cached, sigs, lines, indent, scratch)
                _mark_write(stmt.lhs, lines, indent)
            elif _needs_limb_emit(stmt.rhs, sigs):
                mask = mask_expr(sigs[stmt.lhs].width)
                _emit_assign(tmp, stmt.rhs, cached, sigs, lines, indent, mask, scratch)
                _mark_write(stmt.lhs, lines, indent)
            else:
                rhs_c = emit_expr(stmt.rhs, sigs)
                mask = mask_expr(sigs[stmt.lhs].width)
                if mask:
                    lines.append(f"{sp}{tmp} = ({rhs_c}) & {mask};")
                else:
                    lines.append(f"{sp}{tmp} = {rhs_c};")
                _mark_write(stmt.lhs, lines, indent)
        elif isinstance(stmt, If):
            _emit_compute(
                stmt.cond,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                indent,
                declared,
                scratch,
                demanded,
                branch_assigned,
            )
            if_branch_assigned: set[str] = set()
            if stmt.else_body:
                then_t = _collect_uncached_temps(stmt.then_body, cached, assigns, stop)
                else_t = _collect_uncached_temps(stmt.else_body, cached, assigns, stop)
                _emit_branch_hoist(
                    then_t & else_t,
                    cached,
                    assigns,
                    stop,
                    sigs,
                    lines,
                    indent,
                    declared,
                    scratch,
                    demanded,
                    if_branch_assigned,
                )
            lines.append(
                f"{sp}if ({_cond_code(stmt.cond, cached, sigs, lines, indent, scratch)}) {{"
            )
            # Branch-local demanded copies: then/else must not suppress each
            # other's eval calls (only one arm runs).
            _emit_nba_tree(
                stmt.then_body,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                indent + 2,
                set(declared),
                scratch,
                set(demanded),
                if_branch_assigned,
            )
            if stmt.else_body:
                lines.append(f"{sp}}} else {{")
                _emit_nba_tree(
                    stmt.else_body,
                    cached,
                    assigns,
                    stop,
                    sigs,
                    lines,
                    indent + 2,
                    set(declared),
                    scratch,
                    set(demanded),
                    if_branch_assigned,
                )
                lines.append(f"{sp}}}")
            else:
                lines.append(f"{sp}}}")
        else:
            raise TypeError(stmt)
