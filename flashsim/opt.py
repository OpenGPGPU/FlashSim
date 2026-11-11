"""Simplify FlashSim IR so activity skipping sees muxes, not SSA soup.

CIRCT lowers `x << n` to extract/concat and gives every op an SSA name.
Emitting each name as a struct field + `eval_*` kills both skip and the busy
path. Fold constants, restore shifts, and inline single-use ops into users.
Keep values that feed `always_ff` named so `__ok` can still skip a sticky cone.

Bit extracts are forwarded through concatenations so a consumer's leaf set
drops the half it does not read. Shared reset-to-constant arms are lifted
into one epilogue check instead of a predicate on every register.
"""

from __future__ import annotations

import os
import re
import sys
from collections import ChainMap, Counter, deque
from collections.abc import Mapping
from functools import lru_cache

from flashsim.ir import (
    AlwaysFF,
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    Assign,
    BinOp,
    ClockedAssertion,
    ClockedPrint,
    ClockedProcess,
    CMP_OPS,
    Concat,
    Const,
    Expr,
    Extract,
    Id,
    If,
    MemRead,
    MemWrite,
    Module,
    NbAssign,
    Signal,
    Stmt,
    Ternary,
    UnaryOp,
    expr_id_counts,
    expr_ids,
    expr_width,
    map_expr,
    map_observation,
    observation_exprs,
    stmt_reads,
    stmt_writes,
)


def optimize(mod: Module) -> Module:
    # CIRCT's single-use SSA chains become deeply nested mux expressions as
    # they are inlined. C++ emission already allows this depth; optimization
    # must use the same limit before traversing those expressions.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    assigns = {a.lhs: a.rhs for a in mod.assigns}
    sigs = dict(mod.signals)
    body = mod.always.body
    outputs = {n for n, s in sigs.items() if s.kind == "output"}
    regs = {n for n, s in sigs.items() if s.kind == "reg"}
    inputs = {n for n, s in sigs.items() if s.kind == "input"}
    mems = {n for n, s in sigs.items() if s.kind == "mem"}
    keep = outputs | regs | inputs | mems | {mod.always.clock}

    # CIRCT may materialize a constant as a named wire and then reference that
    # wire directly from an always_ff body. `_subst_consts` removes the named
    # assignment after rewriting other assignments, so carry the same constant
    # map into sequential and memory-write expressions as well. Otherwise the
    # later dangling-wire check rejects valid RTL (for example the dynamic
    # viewport reset constants in GpuHostAxi).
    consts = {n: e for n, e in assigns.items() if isinstance(e, Const)}

    def rewrite(e: Expr) -> Expr:
        e = map_expr(e, lambda node: (
            consts.get(node.name, node)
            if isinstance(node, Id) else node
        ))
        return _rewrite_expr(e, assigns, sigs)

    assigns = _subst_consts(assigns)
    assigns = {k: _rewrite_expr(v, assigns, sigs) for k, v in assigns.items()}
    body = _map_stmts(body, rewrite)
    assertions = [map_observation(a, rewrite) for a in mod.assertions]
    clocked_processes = [ClockedProcess(rewrite(p.clock), p.scope)
                         for p in mod.clocked_processes]
    def observer_reads():
        return set().union(*(expr_ids(expr) for a in assertions
                             for expr in observation_exprs(a)),
                           *(expr_ids(p.clock) for p in clocked_processes))
    assertion_reads = observer_reads()
    observed = outputs | assertion_reads
    mem_writes = [
        MemWrite(
            w.mem,
            rewrite(w.addr),
            rewrite(w.data),
            rewrite(w.enable),
        )
        for w in mod.mem_writes
    ]
    # Array reads index named storage in the emitter.  Inlining an array-valued
    # SSA definition would turn that storage into an unsupported expression.
    array_wires = {n for n, s in sigs.items() if s.kind == "wire" and s.depth}
    # Discard unreachable assignment cones before constructing the much larger
    # per-assignment reference and reverse-use graphs for inlining.
    assigns, body = _dce(assigns, body, observed, mem_writes)
    assigns, body = _inline_single_use(assigns, body, observed | array_wires, mem_writes)
    assigns = {k: _fold_expr(v) for k, v in assigns.items()}
    body = _map_stmts(body, _fold_expr)
    mem_writes = [
        MemWrite(w.mem, _fold_expr(w.addr), _fold_expr(w.data), _fold_expr(w.enable))
        for w in mem_writes
    ]
    if _BIT_SLICE:
        assigns, body, mem_writes = _narrow_bit_slices(assigns, body, mem_writes, sigs)
        assigns = {k: _fold_expr(v) for k, v in assigns.items()}
        body = _map_stmts(body, _fold_expr)
        mem_writes = [
            MemWrite(w.mem, _fold_expr(w.addr), _fold_expr(w.data), _fold_expr(w.enable))
            for w in mem_writes
        ]
    body = _lower_seq_mux(body, assigns, sigs)
    if _RESET_HOIST:
        body, reset_epilogue = _hoist_shared_resets(body, sigs)
    else:
        reset_epilogue = []
    body = _flatten_nested_holds(body)
    body = _merge_hold_ifs(body)
    body = _self_gate_const_holds(body, sigs)
    # self_gate may and-in `valid_k` onto a cond that already ORs all valids;
    # re-fold so `v & (…|v|…)` collapses before emit.
    body = _map_stmts(body, _fold_expr)
    n_writes = 0
    seen_w: set[str] = set()
    for stmt in body + reset_epilogue:
        for name in stmt_writes(stmt):
            if name not in seen_w:
                seen_w.add(name)
                n_writes += 1
    if n_writes >= 256:
        body = _wrap_sticky_nbas(body, sigs)
        body = _flatten_nested_holds(body)
        body = _merge_hold_ifs(body)
        body = _self_gate_const_holds(body, sigs)
        body = _map_stmts(body, _fold_expr)
    body = _cse_sibling_or_prefixes(body, assigns, sigs)
    reset_epilogue = _self_gate_const_holds(reset_epilogue, sigs)
    reset_epilogue = _map_stmts(reset_epilogue, _fold_expr)
    assigns, body = _dce(assigns, body, observed, mem_writes)
    if _MERGE_SOP:
        assigns = _merge_sop_terms(assigns, sigs)
        assigns, body = _dce(assigns, body, observed, mem_writes)
    if _REASSOC_AND:
        assigns = _reassoc_and_chains(assigns, sigs)
        assigns, body = _dce(assigns, body, observed, mem_writes)
    if _DEDUP:
        assigns, body, reset_epilogue, mem_writes, assertions, clocked_processes = _dedup_assigns(
            assigns, body, reset_epilogue, mem_writes, assertions, clocked_processes, sigs
        )
        assertion_reads = observer_reads()
        observed = outputs | assertion_reads
        assigns = {k: _fold_expr(v) for k, v in assigns.items()}
        body = _map_stmts(body, _fold_expr)
        reset_epilogue = _map_stmts(reset_epilogue, _fold_expr)
        assigns, body = _dce(assigns, body, observed, mem_writes)
    if _USED_BITS:
        observed_exprs = [expr for a in assertions for expr in observation_exprs(a)]
        observed_exprs.extend(p.clock for p in clocked_processes)
        assigns, sigs = _apply_used_bits(assigns, body, reset_epilogue, mem_writes,
                                        sigs, observed_exprs)
        assigns = {k: _fold_expr(v) for k, v in assigns.items()}
        body = _map_stmts(body, _fold_expr)
        reset_epilogue = _map_stmts(reset_epilogue, _fold_expr)
        mem_writes = [
            MemWrite(w.mem, _fold_expr(w.addr), _fold_expr(w.data), _fold_expr(w.enable))
            for w in mem_writes
        ]
        assigns, body = _dce(assigns, body, observed, mem_writes)
    if _LUT:
        assigns = _apply_lut(assigns, sigs)

    referred: set[str] = set(assigns)
    for expr in assigns.values():
        referred |= expr_ids(expr)
    for stmt in body:
        referred |= stmt_reads(stmt) | stmt_writes(stmt)
    for stmt in reset_epilogue:
        referred |= stmt_reads(stmt) | stmt_writes(stmt)
    for wr in mem_writes:
        referred |= expr_ids(wr.addr) | expr_ids(wr.data) | expr_ids(wr.enable)
        referred.add(wr.mem)
    referred |= assertion_reads
    # Sorted, not set order: `keep | referred` is a set union, so iterating it
    # directly makes the signal order (and therefore the generated struct
    # layout) depend on PYTHONHASHSEED. Two runs on the same input then emit
    # different C++, which guarantees a ccache miss on every 280 MB ARTI
    # rebuild and makes a binary unreproducible from source.
    new_sigs: dict[str, Signal] = {
        n: sigs[n] for n in sorted(keep | referred) if n in sigs
    }
    _fill_missing_wire_signals(sigs, new_sigs, assigns)
    new_assigns = [Assign(lhs, rhs) for lhs, rhs in assigns.items()]
    for name, expr in assigns.items():
        for dep in expr_ids(expr):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used by {name}")
    for stmt in body:
        for dep in stmt_reads(stmt):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used in sequential body")
    for stmt in reset_epilogue:
        for dep in stmt_reads(stmt):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used in reset epilogue")
    for wr in mem_writes:
        for dep in expr_ids(wr.addr) | expr_ids(wr.data) | expr_ids(wr.enable):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used by mem write")
    for dep in assertion_reads:
        if dep not in assigns and dep not in keep:
            raise ValueError(f"dangling wire {dep} used by clocked assertion")
    return Module(
        mod.name, new_sigs, new_assigns, AlwaysFF(mod.always.clock, body), mod.ports,
        mem_writes, assertions, clocked_processes, reset_epilogue
    )


def _fill_missing_wire_signals(
    sigs: Mapping[str, Signal], new_sigs: dict[str, Signal], assigns: dict[str, Expr]
) -> None:
    # The result map grows as generated temporaries are discovered. A live
    # overlay avoids copying both large signal tables once per temporary.
    width_sigs = ChainMap(new_sigs, sigs)
    for name, expr in assigns.items():
        if name in new_sigs:
            continue
        width = expr_width(expr, width_sigs)
        depth = sigs[name].depth if name in sigs else 0
        new_sigs[name] = Signal(name, width, "wire", depth)


def _subst_consts(assigns: dict[str, Expr]) -> dict[str, Expr]:
    consts = {n: e for n, e in assigns.items() if isinstance(e, Const)}

    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name in consts:
            return consts[node.name]
        return node

    out = {n: map_expr(e, fn) for n, e in assigns.items() if n not in consts}
    return out


def _unwrap(expr: Expr, assigns: dict[str, Expr], seen: set[str] | None = None) -> Expr:
    seen = set() if seen is None else seen
    while isinstance(expr, Id) and expr.name in assigns and expr.name not in seen:
        seen.add(expr.name)
        expr = assigns[expr.name]
    return expr


def _rewrite_expr(expr: Expr, assigns: dict[str, Expr], sigs: dict[str, Signal]) -> Expr:
    return map_expr(expr, lambda n: _rewrite_node(n, assigns, sigs))


def _rewrite_node(expr: Expr, assigns: dict[str, Expr], sigs: dict[str, Signal]) -> Expr:
    if isinstance(expr, Concat) and len(expr.parts) == 2:
        left = _unwrap(expr.parts[0], assigns)
        right = _unwrap(expr.parts[1], assigns)
        shifted = _concat_as_shift(left, right, sigs)
        if shifted is not None:
            return shifted
    if isinstance(expr, BinOp) and expr.op == "^":
        a, b = expr.a, expr.b
        if _is_const(b, 1, 1) and expr_width(a, sigs) == 1:
            return UnaryOp("~", a)
        if _is_const(a, 1, 1) and expr_width(b, sigs) == 1:
            return UnaryOp("~", b)
    if isinstance(expr, Ternary):
        cond = _unwrap(expr.cond, assigns)
        if isinstance(cond, Const):
            return expr.a if cond.value else expr.b
    return expr


def _is_const(expr: Expr, value: int, width: int | None = None) -> bool:
    return isinstance(expr, Const) and expr.value == value and (
        width is None or expr.width == width
    )


def _as_signed(value: int, width: int) -> int:
    value &= (1 << width) - 1
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def _concat_as_shift(left: Expr, right: Expr, sigs: dict[str, Signal]) -> Expr | None:
    # {x[w-n-1:0], n'b0} == x << n
    if isinstance(left, Extract) and isinstance(right, Const) and right.value == 0:
        n = right.width
        src_w = expr_width(left.a, sigs)
        if left.low == 0 and left.width + n == src_w:
            return BinOp("<<", left.a, Const(n, n.bit_length() or 1))
    # {n'b0, x[w-1:n]} == x >> n
    if isinstance(right, Extract) and isinstance(left, Const) and left.value == 0:
        n = left.width
        src_w = expr_width(right.a, sigs)
        if right.low == n and right.width + n == src_w:
            return BinOp(">>", right.a, Const(n, n.bit_length() or 1))
    return None


def _fold_expr(expr: Expr) -> Expr:
    if isinstance(expr, UnaryOp):
        a = _fold_expr(expr.a)
        if isinstance(a, Const) and expr.op in {"~", "!"}:
            width = a.width
            mask = (1 << width) - 1
            return Const((~a.value) & mask, width)
        return UnaryOp(expr.op, a)
    if isinstance(expr, BinOp):
        a, b = _fold_expr(expr.a), _fold_expr(expr.b)
        if isinstance(a, Const) and isinstance(b, Const):
            width = max(a.width, b.width)
            mask = (1 << width) - 1
            ops = {
                "+": lambda x, y: x + y,
                "-": lambda x, y: x - y,
                "*": lambda x, y: x * y,
                "&": lambda x, y: x & y,
                "|": lambda x, y: x | y,
                "^": lambda x, y: x ^ y,
                "<<": lambda x, y: x << y,
                ">>": lambda x, y: x >> y,
                "==": lambda x, y: int(x == y),
                "!=": lambda x, y: int(x != y),
                "<": lambda x, y: int(x < y),
                "<=": lambda x, y: int(x <= y),
                ">": lambda x, y: int(x > y),
                ">=": lambda x, y: int(x >= y),
                "s<": lambda x, y, w=width: int(_as_signed(x, w) < _as_signed(y, w)),
                "s<=": lambda x, y, w=width: int(_as_signed(x, w) <= _as_signed(y, w)),
                "s>": lambda x, y, w=width: int(_as_signed(x, w) > _as_signed(y, w)),
                "s>=": lambda x, y, w=width: int(_as_signed(x, w) >= _as_signed(y, w)),
            }
            if expr.op in ops:
                width = 1 if expr.op in CMP_OPS else width
                return Const(ops[expr.op](a.value, b.value) & mask, width)
        if expr.op == "&":
            absorbed = _absorb_and(BinOp("&", a, b))
            if absorbed is not None:
                return absorbed
        return BinOp(expr.op, a, b)
    if isinstance(expr, Ternary):
        cond, a, b = _fold_expr(expr.cond), _fold_expr(expr.a), _fold_expr(expr.b)
        if isinstance(cond, Const):
            return a if cond.value else b
        return Ternary(cond, a, b)
    if isinstance(expr, Extract):
        a = _fold_expr(expr.a)
        if isinstance(a, Const):
            mask = (1 << expr.width) - 1
            return Const((a.value >> expr.low) & mask, expr.width)
        return Extract(a, expr.low, expr.width)
    if isinstance(expr, Concat):
        return Concat(tuple(_fold_expr(p) for p in expr.parts))
    if isinstance(expr, ArrayGet):
        return ArrayGet(_fold_expr(expr.arr), _fold_expr(expr.index))
    if isinstance(expr, ArrayInject):
        return ArrayInject(_fold_expr(expr.arr), _fold_expr(expr.index), _fold_expr(expr.value))
    if isinstance(expr, MemRead):
        return MemRead(expr.mem, _fold_expr(expr.addr))
    if isinstance(expr, ArrayZeros):
        return expr
    return expr


def _count_uses(
    assigns: dict[str, Expr],
    body: list[Stmt],
    mem_writes: list[MemWrite] | None = None,
) -> dict[str, int]:
    uses: dict[str, int] = {}

    def add(expr: Expr) -> None:
        for name, n in expr_id_counts(expr).items():
            uses[name] = uses.get(name, 0) + n

    for expr in assigns.values():
        add(expr)
    for stmt in body:
        if isinstance(stmt, NbAssign):
            add(stmt.rhs)
        elif isinstance(stmt, If):
            add(stmt.cond)
            nested = _count_uses({}, stmt.then_body + stmt.else_body)
            for name, n in nested.items():
                uses[name] = uses.get(name, 0) + n
        else:
            raise TypeError(stmt)
    for wr in mem_writes or []:
        add(wr.addr)
        add(wr.data)
        add(wr.enable)
        uses[wr.mem] = uses.get(wr.mem, 0) + 1
    return uses


_INLINE_NODE_BUDGET = 512


def _bounded_inline_size(expr: Expr) -> int:
    """Conservatively bound the expanded tree without recursive hashing."""
    size = 0
    stack = [expr]
    while stack:
        node = stack.pop()
        size += 1
        if size > _INLINE_NODE_BUDGET:
            return size
        if isinstance(node, (Id, Const, ArrayZeros)):
            continue
        if isinstance(node, (UnaryOp, Extract)):
            stack.append(node.a)
        elif isinstance(node, BinOp):
            stack.extend((node.a, node.b))
        elif isinstance(node, Ternary):
            stack.extend((node.cond, node.a, node.b))
        elif isinstance(node, Concat):
            stack.extend(node.parts)
        elif isinstance(node, ArrayGet):
            stack.extend((node.arr, node.index))
        elif isinstance(node, ArrayInject):
            stack.extend((node.arr, node.index, node.value))
        elif isinstance(node, MemRead):
            stack.append(node.addr)
        else:
            raise TypeError(type(node))
    return size


def _inline_single_use(
    assigns: dict[str, Expr],
    body: list[Stmt],
    outputs: set[str],
    mem_writes: list[MemWrite] | None = None,
) -> tuple[dict[str, Expr], list[Stmt]]:
    # Track the graph as names are removed, then expand substitutions once.
    # Rebuilding an ever-growing expression after each inlining step is very
    # expensive for the deep, single-use chains in large CIRCT models.
    uses = _count_uses(assigns, body, mem_writes or [])
    refs = {name: expr_id_counts(expr) for name, expr in assigns.items()}
    users: dict[str, set[str]] = {}
    for user, dependencies in refs.items():
        for dep in dependencies:
            if dep in assigns:
                users.setdefault(dep, set()).add(user)
    pending = deque(assigns)
    queued = set(assigns)
    substitutions: dict[str, Expr] = {}
    sizes: dict[str, int] = {}

    def size(name: str) -> int:
        cached = sizes.get(name)
        if cached is None:
            cached = _bounded_inline_size(assigns[name])
            sizes[name] = cached
        return cached

    def enqueue(name: str) -> None:
        if name in assigns and name not in queued:
            pending.append(name)
            queued.add(name)

    while pending:
        name = pending.popleft()
        queued.remove(name)
        if name not in assigns or name in outputs:
            continue
        n = uses.get(name, 0)
        if n == 0:
            del assigns[name]
            sizes.pop(name, None)
            for dep, count in refs.pop(name).items():
                if dep in assigns:
                    uses[dep] -= count
                    users.get(dep, set()).discard(name)
                    enqueue(dep)
            continue
        if n != 1 or len(users.get(name, ())) != 1:
            continue
        user = next(iter(users[name]))
        if user not in assigns or user == name:
            continue
        source_size = size(name)
        user_size = size(user)
        expanded_size = source_size + user_size - 1
        if source_size > 1 and expanded_size > _INLINE_NODE_BUDGET:
            # Keep the named intermediate: a huge inlined tree can overflow
            # Python hashing/emission or the C++ compiler's nesting limit.
            continue
        sizes[user] = min(expanded_size, _INLINE_NODE_BUDGET + 1)
        sizes.pop(name, None)
        dependencies = refs.pop(name)
        destination = refs[user]
        assert destination.pop(name) == 1
        for dep, count in dependencies.items():
            destination[dep] = destination.get(dep, 0) + count
            if dep in assigns:
                users.setdefault(dep, set()).discard(name)
                users[dep].add(user)
                enqueue(dep)
        substitutions[name] = assigns.pop(name)
        uses[name] -= 1
        users.pop(name, None)
        enqueue(user)

    def expand(expr: Expr) -> Expr:
        # Explicit traversal avoids Python recursion for long chains of
        # non-alias operations as well as for chains of plain aliases.
        stack: list[tuple[Expr, bool]] = [(expr, False)]
        values: list[Expr] = []
        while stack:
            node, ready = stack.pop()
            if not ready and isinstance(node, Id) and node.name in substitutions:
                stack.append((substitutions[node.name], False))
                continue
            if isinstance(node, (Id, Const, ArrayZeros)):
                values.append(node)
                continue
            if isinstance(node, (UnaryOp, Extract)):
                children = (node.a,)
            elif isinstance(node, BinOp):
                children = (node.a, node.b)
            elif isinstance(node, Ternary):
                children = (node.cond, node.a, node.b)
            elif isinstance(node, Concat):
                children = node.parts
            elif isinstance(node, ArrayGet):
                children = (node.arr, node.index)
            elif isinstance(node, ArrayInject):
                children = (node.arr, node.index, node.value)
            elif isinstance(node, MemRead):
                children = (node.addr,)
            else:
                raise TypeError(type(node))
            if not children:
                values.append(node)
            elif not ready:
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(children))
            else:
                rewritten = values[-len(children):]
                del values[-len(children):]
                if all(a is b for a, b in zip(rewritten, children)):
                    values.append(node)
                elif isinstance(node, UnaryOp):
                    values.append(UnaryOp(node.op, rewritten[0]))
                elif isinstance(node, Extract):
                    values.append(Extract(rewritten[0], node.low, node.width))
                elif isinstance(node, BinOp):
                    values.append(BinOp(node.op, rewritten[0], rewritten[1]))
                elif isinstance(node, Ternary):
                    values.append(Ternary(*rewritten))
                elif isinstance(node, Concat):
                    values.append(Concat(tuple(rewritten)))
                elif isinstance(node, ArrayGet):
                    values.append(ArrayGet(*rewritten))
                elif isinstance(node, ArrayInject):
                    values.append(ArrayInject(*rewritten))
                else:
                    assert isinstance(node, MemRead)
                    values.append(MemRead(node.mem, rewritten[0]))
        assert len(values) == 1
        return values[0]

    return {name: expand(expr) for name, expr in assigns.items()}, body


def _dce(
    assigns: dict[str, Expr],
    body: list[Stmt],
    outputs: set[str],
    mem_writes: list[MemWrite] | None = None,
) -> tuple[dict[str, Expr], list[Stmt]]:
    live: set[str] = set(outputs)
    for stmt in body:
        live |= stmt_reads(stmt)
    for wr in mem_writes or []:
        live |= expr_ids(wr.addr) | expr_ids(wr.data) | expr_ids(wr.enable)
        live.add(wr.mem)
    # The assignment graph is fixed during this pass. Queue each newly live
    # name once rather than rescanning the entire growing live set per layer.
    pending = list(live)
    while pending:
        name = pending.pop()
        if name not in assigns:
            continue
        for dep in expr_ids(assigns[name]):
            if dep not in live:
                live.add(dep)
                pending.append(dep)
    assigns = {n: e for n, e in assigns.items() if n in live}
    return assigns, body


_BIT_SLICE = os.environ.get("FLASHSIM_BIT_SLICE", "1") != "0"
_USED_BITS = os.environ.get("FLASHSIM_USED_BITS", "1") != "0"
_LUT = os.environ.get("FLASHSIM_LUT", "1") != "0"
_LUT_MAX_BITS = int(os.environ.get("FLASHSIM_LUT_MAX_BITS", "6"))
_LUT_MIN_NODES = int(os.environ.get("FLASHSIM_LUT_MIN_NODES", "20"))
_DEDUP = os.environ.get("FLASHSIM_DEDUP", "1") != "0"
_DEDUP_MIN_NODES = int(os.environ.get("FLASHSIM_DEDUP_MIN_NODES", "2"))
_RESET_HOIST = os.environ.get("FLASHSIM_RESET_HOIST", "1") != "0"
_RESET_HOIST_MIN = int(os.environ.get("FLASHSIM_RESET_HOIST_MIN", "2"))
_RESET_NAME = re.compile(r"(^|_)reset($|_)", re.IGNORECASE)
_SLICE_INLINE_NODES = 16
_SLICE_WIDTH_CACHE_MAX = 8192


def _slice_width(expr: Expr, sigs: dict[str, Signal],
                 cache: dict[int, tuple[Expr, int]]) -> int:
    # Keep the frequently repeated wide subtrees, not every tiny atom in a
    # whole-host netlist. Holding the object prevents identity reuse.
    if isinstance(expr, (Id, Const, Extract, ArrayZeros, MemRead)):
        return expr_width(expr, sigs)
    key = id(expr)
    hit = cache.get(key)
    if hit is not None and hit[0] is expr:
        return hit[1]
    width = expr_width(expr, sigs)
    if len(cache) >= _SLICE_WIDTH_CACHE_MAX:
        cache.clear()
    cache[key] = (expr, width)
    return width


def _slice_src(
    src: Expr,
    low: int,
    width: int,
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    uses: dict[str, int],
    fresh: dict[str, Expr],
    cache: dict[tuple[str, int, int], Expr],
    seen: set[str],
    fuel: int = 64,
    width_cache: dict[int, tuple[Expr, int]] | None = None,
) -> Expr | None:
    """Narrow `src[low +: width]` onto the operands that actually feed it.

    Concatenations are the common case GSIM splits: a consumer of one half
    must not keep the other half in its leaf set. Returns None when the
    slice cannot be made smaller without copying a large shared tree.
    """
    if fuel <= 0 or width <= 0 or low < 0:
        return None
    if width_cache is None:
        width_cache = {}
    try:
        src_w = _slice_width(src, sigs, width_cache)
    except (KeyError, TypeError):
        return None
    if low + width > src_w:
        return None
    if low == 0 and width == src_w:
        return src
    if isinstance(src, Const):
        mask = (1 << width) - 1
        return Const((src.value >> low) & mask, width)
    if isinstance(src, Extract):
        if low + width > src.width:
            return None
        inner = _slice_src(
            src.a, src.low + low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        return inner if inner is not None else Extract(src.a, src.low + low, width)
    if isinstance(src, Concat) and src.parts:
        widths = []
        for part in src.parts:
            try:
                widths.append(_slice_width(part, sigs, width_cache))
            except (KeyError, TypeError):
                return None
        if any(part_w <= 0 for part_w in widths):
            return None
        bases = [0] * len(src.parts)
        cursor = 0
        for index in range(len(src.parts) - 1, -1, -1):
            bases[index] = cursor
            cursor += widths[index]
        if cursor != src_w:
            return None
        hi = low + width
        pieces: list[Expr] = []
        for index, part in enumerate(src.parts):
            part_lo = bases[index]
            part_hi = part_lo + widths[index]
            if part_hi <= low or part_lo >= hi:
                continue
            take_lo = max(low, part_lo) - part_lo
            take_w = min(hi, part_hi) - max(low, part_lo)
            if take_lo == 0 and take_w == widths[index]:
                pieces.append(part)
                continue
            inner = _slice_src(
                part, take_lo, take_w, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
            )
            pieces.append(inner if inner is not None else Extract(part, take_lo, take_w))
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0]
        return Concat(tuple(pieces))
    if (
        width == 1
        and src_w <= 128
        and isinstance(src, BinOp)
        and src.op == "<<"
    ):
        base = src.a
        if isinstance(base, Id) and isinstance(assigns.get(base.name), Const):
            base = assigns[base.name]
        if isinstance(base, Const) and base.value == 1 and low < src_w:
            try:
                amt_w = _slice_width(src.b, sigs, width_cache)
            except (KeyError, TypeError):
                return None
            if low.bit_length() > amt_w:
                return Const(0, 1)
            return BinOp("==", src.b, Const(low, amt_w))
    if isinstance(src, UnaryOp) and src.op == "~":
        inner = _slice_src(
            src.a, low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        if inner is not None and _expr_nodes(inner) <= _SLICE_INLINE_NODES:
            return UnaryOp("~", inner)
        return None
    if isinstance(src, BinOp) and src.op in {"&", "|", "^"}:
        left = _slice_src(
            src.a, low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        right = _slice_src(
            src.b, low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        if (
            left is not None
            and right is not None
            and _expr_nodes(left) + _expr_nodes(right) <= _SLICE_INLINE_NODES
        ):
            return BinOp(src.op, left, right)
        return None
    if isinstance(src, Ternary):
        left = _slice_src(
            src.a, low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        right = _slice_src(
            src.b, low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        if (
            left is not None
            and right is not None
            and _expr_nodes(left) + _expr_nodes(right) <= _SLICE_INLINE_NODES
        ):
            return Ternary(src.cond, left, right)
        return None
    if isinstance(src, Id) and src.name in assigns and src.name not in seen:
        key = (src.name, low, width)
        hit = cache.get(key)
        if hit is not None:
            return hit
        seen.add(src.name)
        sliced = _slice_src(
            assigns[src.name], low, width, assigns, sigs, uses, fresh, cache, seen, fuel - 1, width_cache
        )
        seen.remove(src.name)
        if sliced is None:
            return None
        if (
            isinstance(sliced, Extract)
            and isinstance(sliced.a, Id)
            and sliced.a.name == src.name
            and sliced.low == low
            and sliced.width == width
        ):
            return None
        original = assigns[src.name]
        # A compare or a constant is cheaper than the wide value it came from
        # even when the leaf set is unchanged (one-hot `bits(1 << i, k)`).
        cheaper = isinstance(sliced, (Const, Id)) or (
            isinstance(sliced, BinOp) and sliced.op in CMP_OPS
        )
        if (
            not cheaper
            and expr_ids(sliced) == expr_ids(original)
            and _expr_nodes(sliced) >= _expr_nodes(original)
        ):
            return None
        if _expr_nodes(sliced) <= _SLICE_INLINE_NODES or uses.get(src.name, 0) <= 1:
            return sliced
        name = f"{src.name}__b{low}_{width}"
        suffix = 0
        while name in assigns or name in fresh or name in sigs:
            suffix += 1
            name = f"{src.name}__b{low}_{width}_{suffix}"
        fresh[name] = sliced
        # Subsequent narrowing and reassociation ask for widths before the
        # final signal table is built. The slice's width is already known.
        sigs[name] = Signal(name, width, "wire")
        ref = Id(name)
        cache[key] = ref
        return ref
    return None


def _narrow_expr(
    expr: Expr,
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    uses: dict[str, int],
    fresh: dict[str, Expr],
    cache: dict[tuple[str, int, int], Expr],
    width_cache: dict[int, tuple[Expr, int]],
) -> Expr:
    def fn(node: Expr) -> Expr:
        if not isinstance(node, Extract):
            return node
        sliced = _slice_src(
            node.a, node.low, node.width, assigns, sigs, uses, fresh, cache, set(), 64, width_cache
        )
        return sliced if sliced is not None else node

    return map_expr(expr, fn)


def _narrow_bit_slices(
    assigns: dict[str, Expr],
    body: list[Stmt],
    mem_writes: list[MemWrite],
    sigs: dict[str, Signal],
) -> tuple[dict[str, Expr], list[Stmt], list[MemWrite]]:
    uses = _count_uses(assigns, body, mem_writes)
    fresh: dict[str, Expr] = {}
    cache: dict[tuple[str, int, int], Expr] = {}
    width_cache: dict[int, tuple[Expr, int]] = {}

    def narrow(expr: Expr) -> Expr:
        return _narrow_expr(expr, assigns, sigs, uses, fresh, cache, width_cache)

    narrowed = {name: narrow(expr) for name, expr in assigns.items()}
    body = _map_stmts(body, narrow)
    mem_writes = [
        MemWrite(w.mem, narrow(w.addr), narrow(w.data), narrow(w.enable))
        for w in mem_writes
    ]
    # Sequential and memory-write expressions can create new slices too.
    narrowed.update(fresh)
    return narrowed, body, mem_writes


def _is_reset_pred(cond: Expr) -> bool:
    if isinstance(cond, Id):
        return _RESET_NAME.search(cond.name) is not None
    return (
        isinstance(cond, UnaryOp)
        and cond.op == "!"
        and isinstance(cond.a, Id)
        and _RESET_NAME.search(cond.a.name) is not None
    )


def _const_arm(stmts: list[Stmt], sigs: dict[str, Signal]) -> list[NbAssign] | None:
    if not stmts:
        return None
    arm: list[NbAssign] = []
    for stmt in stmts:
        if not isinstance(stmt, NbAssign) or not isinstance(stmt.rhs, Const):
            return None
        sig = sigs.get(stmt.lhs)
        if sig is None or sig.depth or sig.kind == "mem":
            return None
        arm.append(stmt)
    if len({stmt.lhs for stmt in arm}) != len(arm):
        return None
    return arm


def _hoist_shared_resets(
    body: list[Stmt], sigs: dict[str, Signal]
) -> tuple[list[Stmt], list[Stmt]]:
    """Lift shared reset-to-constant arms into one epilogue check.

    GSIM assumes reset is false on the fast path and repairs the few reset
    nets afterwards. Only a predicate whose name is `reset` (or `*_reset` /
    `reset_*`) qualifies, and the data arm must not write a register the
    constant arm does not override.
    """
    if _RESET_HOIST_MIN < 2:
        return body, []
    classified: list[tuple[Expr, list[Stmt], list[NbAssign]] | None] = []
    for stmt in body:
        if not isinstance(stmt, If) or not _is_reset_pred(stmt.cond):
            classified.append(None)
            continue
        then_c = _const_arm(stmt.then_body, sigs)
        else_c = _const_arm(stmt.else_body, sigs) if stmt.else_body else None
        if then_c is not None and else_c is None:
            data: list[Stmt] = list(stmt.else_body)
            consts = then_c
            taken = stmt.cond
        elif else_c is not None and then_c is None:
            data = list(stmt.then_body)
            consts = else_c
            taken = _not_expr(stmt.cond)
        else:
            classified.append(None)
            continue
        covered = {assign.lhs for assign in consts}
        written: set[str] = set()
        for item in data:
            written |= stmt_writes(item)
        if not written <= covered:
            classified.append(None)
            continue
        classified.append((taken, data, consts))
    counts: dict[Expr, int] = {}
    for info in classified:
        if info is not None:
            counts[info[0]] = counts.get(info[0], 0) + 1
    out: list[Stmt] = []
    buckets: dict[Expr, list[NbAssign]] = {}
    order: list[Expr] = []
    for stmt, info in zip(body, classified):
        if info is None or counts[info[0]] < _RESET_HOIST_MIN:
            out.append(stmt)
            continue
        taken, data, consts = info
        out.extend(data)
        if taken not in buckets:
            order.append(taken)
            buckets[taken] = []
        buckets[taken].extend(consts)
    epilogue = [If(taken, list(buckets[taken]), []) for taken in order]
    return out, epilogue


def _lower_seq_mux(body: list[Stmt], assigns: dict[str, Expr], sigs: dict[str, Signal]) -> list[Stmt]:
    """Turn `reg <= mux(...)` back into `if` so mix cones are not always demanded."""
    out: list[Stmt] = []
    for stmt in body:
        if isinstance(stmt, NbAssign):
            out.extend(
                _nba_to_if(stmt.lhs, _follow(stmt.rhs, assigns, stmt.lhs, sigs), sigs)
            )
        elif isinstance(stmt, If):
            out.append(
                If(
                    stmt.cond,
                    _lower_seq_mux(stmt.then_body, assigns, sigs),
                    _lower_seq_mux(stmt.else_body, assigns, sigs),
                )
            )
        else:
            raise TypeError(stmt)
    return out


def _follow(
    expr: Expr, assigns: dict[str, Expr], lhs: str | None = None, sigs: dict[str, Signal] | None = None
) -> Expr:
    if isinstance(expr, Id) and expr.name in assigns:
        inner = assigns[expr.name]
        if isinstance(inner, Ternary):
            return inner
        if (
            lhs is not None
            and sigs is not None
            and lhs in sigs
            and sigs[lhs].width == 1
            and _hold_next(inner, lhs) is not None
        ):
            return inner
    return expr


def _flatten_op(expr: Expr, op: str) -> list[Expr]:
    if isinstance(expr, BinOp) and expr.op == op:
        return _flatten_op(expr.a, op) + _flatten_op(expr.b, op)
    return [expr]


def _absorb_and(expr: BinOp) -> Expr | None:
    """Simplify `v & (v | …)` / duplicate `v & v` in an and-chain.

    Instruction-cache holds gate each bit with a mega `(OR all valids) & valid_k`
    conjunct; absorbing the OR collapses multi-KB conditions to a few Ids.
    """
    parts = _flatten_op(expr, "&")
    if len(parts) < 2:
        return None
    id_names = {p.name for p in parts if isinstance(p, Id)}
    out: list[Expr] = []
    seen_ids: set[str] = set()
    changed = False
    for p in parts:
        if isinstance(p, Id):
            if p.name in seen_ids:
                changed = True
                continue
            seen_ids.add(p.name)
            out.append(p)
            continue
        if id_names and isinstance(p, BinOp) and p.op == "|":
            leaves = _flatten_op(p, "|")
            if any(isinstance(x, Id) and x.name in id_names for x in leaves):
                changed = True
                continue
        out.append(p)
    if not changed or not out:
        return None
    if len(out) == 1:
        return out[0]
    return _and_chain(out)


def _and_chain(parts: list[Expr]) -> Expr:
    expr = parts[0]
    for part in parts[1:]:
        expr = BinOp("&", expr, part)
    return expr


def _or_chain(parts: list[Expr]) -> Expr:
    expr = parts[0]
    for part in parts[1:]:
        expr = BinOp("|", expr, part)
    return expr


# Sibling L2 gates share huge &/| spines that differ only in a suffix.
# Factor the common prefix (by structural equality) into one SSA temp.
_CSE_CHAIN_NODES_MIN = 24
_CSE_OR_GROUP_MIN = 3


def _expr_eq(a: Expr, b: Expr) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, Id):
        return a.name == b.name
    if isinstance(a, Const):
        return a.value == b.value and a.width == b.width
    if isinstance(a, UnaryOp):
        return a.op == b.op and _expr_eq(a.a, b.a)
    if isinstance(a, BinOp):
        return a.op == b.op and _expr_eq(a.a, b.a) and _expr_eq(a.b, b.b)
    if isinstance(a, Ternary):
        return (
            _expr_eq(a.cond, b.cond)
            and _expr_eq(a.a, b.a)
            and _expr_eq(a.b, b.b)
        )
    if isinstance(a, Extract):
        return a.low == b.low and a.width == b.width and _expr_eq(a.a, b.a)
    if isinstance(a, Concat):
        return len(a.parts) == len(b.parts) and all(
            _expr_eq(x, y) for x, y in zip(a.parts, b.parts)
        )
    return False


def _expr_nodes(expr: Expr) -> int:
    if isinstance(expr, (Id, Const, ArrayZeros)):
        return 1
    if isinstance(expr, UnaryOp):
        return 1 + _expr_nodes(expr.a)
    if isinstance(expr, BinOp):
        return 1 + _expr_nodes(expr.a) + _expr_nodes(expr.b)
    if isinstance(expr, Ternary):
        return 1 + _expr_nodes(expr.cond) + _expr_nodes(expr.a) + _expr_nodes(expr.b)
    if isinstance(expr, Extract):
        return 1 + _expr_nodes(expr.a)
    if isinstance(expr, Concat):
        return 1 + sum(_expr_nodes(p) for p in expr.parts)
    if isinstance(expr, ArrayGet):
        return 1 + _expr_nodes(expr.arr) + _expr_nodes(expr.index)
    if isinstance(expr, ArrayInject):
        return (
            1
            + _expr_nodes(expr.arr)
            + _expr_nodes(expr.index)
            + _expr_nodes(expr.value)
        )
    if isinstance(expr, MemRead):
        return 1 + _expr_nodes(expr.addr)
    return 1


def _chain_op(op: str, parts: list[Expr]) -> Expr:
    expr = parts[0]
    for part in parts[1:]:
        expr = BinOp(op, expr, part)
    return expr


def _cse_or_prefix_run(
    stmts: list[Stmt], assigns: dict[str, Expr], counter: list[int],
    sigs: dict[str, Signal] | None = None,
) -> list[Stmt]:
    """Factor shared &/| prefixes across a flat list of sibling stmts."""
    out: list[Stmt] = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if not isinstance(stmt, If):
            out.append(stmt)
            i += 1
            continue
        best_op: str | None = None
        best_group: list[If] = []
        best_seqs: list[list[Expr]] = []
        best_pre = 0
        best_nodes = 0
        for op in ("&", "|"):
            first = _flatten_op(stmt.cond, op)
            if len(first) < 2 and _expr_nodes(stmt.cond) < _CSE_CHAIN_NODES_MIN:
                continue
            group = [stmt]
            seqs = [first]
            # The current group already shares `pre` terms. Comparing its
            # members again for every new sibling makes a long run quadratic.
            pre = len(first)
            prefix_nodes = [0]
            j = i + 1
            while j < len(stmts):
                nxt = stmts[j]
                if not isinstance(nxt, If):
                    break
                nl = _flatten_op(nxt.cond, op)
                limit = min(pre, len(nl))
                next_pre = 0
                while next_pre < limit and _expr_eq(nl[next_pre], first[next_pre]):
                    next_pre += 1
                if next_pre < 1:
                    break
                while len(prefix_nodes) <= next_pre:
                    prefix_nodes.append(prefix_nodes[-1] + _expr_nodes(first[len(prefix_nodes)-1]))
                nodes = prefix_nodes[next_pre] + next_pre - 1
                if nodes < _CSE_CHAIN_NODES_MIN:
                    break
                group.append(nxt)
                seqs.append(nl)
                pre = next_pre
                j += 1
            if len(group) < _CSE_OR_GROUP_MIN:
                continue
            nodes = prefix_nodes[pre] + pre - 1
            if nodes > best_nodes:
                best_op, best_group, best_seqs, best_pre, best_nodes = (
                    op,
                    group,
                    seqs,
                    pre,
                    nodes,
                )
        if best_op is None or best_nodes < _CSE_CHAIN_NODES_MIN:
            out.append(stmt)
            i += 1
            continue
        while True:
            cse = f"_fs_cse_t{counter[0]}"
            counter[0] += 1
            if cse not in assigns and (sigs is None or cse not in sigs):
                break
        prefix = best_seqs[0][:best_pre]
        assigns[cse] = prefix[0] if len(prefix) == 1 else _chain_op(best_op, prefix)
        if sigs is not None:
            try:
                sigs[cse] = Signal(cse, expr_width(assigns[cse], sigs), "wire")
            except (KeyError, TypeError):
                # Later validation reports genuinely unresolved wires.
                pass
        for ifs, terms in zip(best_group, best_seqs):
            suffix = terms[best_pre:]
            if suffix:
                new_cond = _chain_op(best_op, [Id(cse)] + suffix)
            else:
                new_cond = Id(cse)
            out.append(If(new_cond, ifs.then_body, ifs.else_body))
        i = i + len(best_group)
    return out


def _cse_sibling_or_prefixes(
    body: list[Stmt], assigns: dict[str, Expr], sigs: dict[str, Signal] | None = None
) -> list[Stmt]:
    """CSE long shared &/| prefixes among sibling hold/if conditions (L2)."""
    counter = [0]

    def walk(stmts: list[Stmt]) -> list[Stmt]:
        mapped: list[Stmt] = []
        for stmt in stmts:
            if isinstance(stmt, If):
                mapped.append(
                    If(stmt.cond, walk(stmt.then_body), walk(stmt.else_body))
                )
            else:
                mapped.append(stmt)
        return _cse_or_prefix_run(mapped, assigns, counter, sigs)

    return walk(body)


def _hold_next(expr: Expr, lhs: str) -> tuple[str, Expr, Expr] | None:
    """Recognize 1-bit `enable & (set | x)` or `enable & (toggle ^ x)`."""
    if not isinstance(expr, BinOp):
        return None
    ands = _flatten_op(expr, "&")
    for i, part in enumerate(ands):
        ors = _flatten_op(part, "|") if isinstance(part, BinOp) and part.op == "|" else [part]
        if any(isinstance(p, Id) and p.name == lhs for p in ors):
            rest = [p for p in ors if not (isinstance(p, Id) and p.name == lhs)]
            enables = [p for j, p in enumerate(ands) if j != i]
            if not enables:
                return None
            set_expr = Const(0, 1) if not rest else rest[0] if len(rest) == 1 else _or_chain(rest)
            return ("or", _and_chain(enables), set_expr)
        if isinstance(part, BinOp) and part.op == "^":
            toggle = None
            if isinstance(part.a, Id) and part.a.name == lhs:
                toggle = part.b
            elif isinstance(part.b, Id) and part.b.name == lhs:
                toggle = part.a
            if toggle is None:
                continue
            enables = [p for j, p in enumerate(ands) if j != i]
            if not enables:
                return None
            return ("xor", _and_chain(enables), toggle)
    return None


def _nba_to_if(lhs: str, expr: Expr, sigs: dict[str, Signal]) -> list[Stmt]:
    # A hold is a self-reference: CIRCT emits `when (en) r <= v` as
    # mux(en, v, r), so only a self-referencing arm may be dropped. An arm
    # that is a constant — zero included — is real data the design drives,
    # never an undriven else, so it must be kept.
    #
    # Ids are not expanded into the statement tree: that duplicates shared
    # SSA cones and explodes code size. Only Ternary nodes already present
    # in the expression get lowered.
    if isinstance(expr, Ternary):
        then_body = _nba_to_if(lhs, expr.a, sigs)
        else_body = _nba_to_if(lhs, expr.b, sigs)
        if _is_hold(lhs, else_body):
            return [If(expr.cond, then_body)]
        if _is_hold(lhs, then_body):
            return [If(UnaryOp("!", expr.cond), else_body)]
        return [If(expr.cond, then_body, else_body)]
    if lhs in sigs and sigs[lhs].width == 1:
        hold = _hold_next(expr, lhs)
        if hold is not None:
            kind, enable, extra = hold
            reset_body = [NbAssign(lhs, Const(0, 1))]
            if kind == "or":
                taken = [NbAssign(lhs, Const(1, 1))]
            else:
                taken = [NbAssign(lhs, UnaryOp("~", Id(lhs)))]
            return [
                If(
                    UnaryOp("!", enable),
                    reset_body,
                    [If(extra, taken)],
                )
            ]
    return [NbAssign(lhs, expr)]


def _is_hold(lhs: str, body: list[Stmt]) -> bool:
    return (
        len(body) == 1
        and isinstance(body[0], NbAssign)
        and body[0].lhs == lhs
        and isinstance(body[0].rhs, Id)
        and body[0].rhs.name == lhs
    )


def _is_hold_if(stmt: Stmt) -> bool:
    return isinstance(stmt, If) and not stmt.else_body


def _all_holds(stmts: list[Stmt]) -> bool:
    return bool(stmts) and all(_is_hold_if(s) for s in stmts)


def _not_expr(expr: Expr) -> Expr:
    if isinstance(expr, UnaryOp) and expr.op in {"!", "~"}:
        return expr.a
    return UnaryOp("!", expr)


_FLATTEN_ELSE_MAX = int(os.environ.get("FLASHSIM_FLATTEN_ELSE_MAX", "4"))
_MERGE_SOP = os.environ.get("FLASHSIM_MERGE_SOP", "1") != "0"
_SOP_TERMS_MAX = 256
_SOP_LITS_MAX = 64
_SOP_FALSE = frozenset([("<false>", True)])


def _merge_sop_terms(
    assigns: dict[str, Expr], sigs: dict[str, Signal]
) -> dict[str, Expr]:
    """Collapse 1-bit ORs whose AND terms re-merge complementary branches.

    CIRCT's control-flow removal gives the block after an if/else the
    condition `(p & c & …) | (p & !c & …) | …`, one term per path, and each
    later statement's block condition is built on the previous one. For an
    L2 slice's 256 `sharers(way)(idx) :=` trees that is a 256-deep chain of
    52-term ORs that is really just the entry condition. Flatten each OR to
    literal sets, then apply `A.x | A.!x = A`, `A | A.!x.B = A | A.B` and
    absorption until nothing changes.
    """

    def bit(e: Expr) -> bool:
        try:
            return expr_width(e, sigs) == 1
        except (KeyError, TypeError):
            return False

    def neg_of(e: Expr) -> Expr | None:
        if isinstance(e, UnaryOp) and e.op in {"!", "~"} and bit(e.a):
            return e.a
        if isinstance(e, BinOp) and e.op == "^" and bit(e.a):
            if _is_const(e.b, 1):
                return e.a
            if _is_const(e.a, 1):
                return e.b
        return None

    def resolve(e: Expr) -> Expr:
        seen = 0
        while isinstance(e, Id) and e.name in assigns and seen < 64:
            inner = assigns[e.name]
            if isinstance(inner, Id) or neg_of(inner) is not None:
                e = inner
                seen += 1
                continue
            break
        return e

    def atom(e: Expr) -> tuple[Expr, bool]:
        pol = True
        for _ in range(64):
            e = resolve(e)
            n = neg_of(e)
            if n is None:
                break
            e, pol = n, not pol
        return e, pol

    def is_and(e: Expr) -> Expr | None:
        if isinstance(e, BinOp) and e.op == "&":
            return e
        if isinstance(e, Id) and e.name in assigns:
            inner = assigns[e.name]
            if isinstance(inner, BinOp) and inner.op == "&":
                return inner
        return None

    def term_lits(e: Expr, opaque: set[Expr]) -> frozenset | None:
        lits: set[tuple[Expr, bool]] = set()
        stack = [e]
        while stack:
            k, pol = atom(stack.pop())
            conj = is_and(k) if pol and k not in opaque else None
            if conj is not None and bit(conj.a) and bit(conj.b):
                stack.append(conj.b)
                stack.append(conj.a)
                continue
            if (k, not pol) in lits:
                return _SOP_FALSE
            lits.add((k, pol))
            if len(lits) > _SOP_LITS_MAX:
                return None
        return frozenset(lits)

    @lru_cache(maxsize=8192)
    def simple_literal_key(literal) -> str:
        return repr(literal)

    def literal_key(literal) -> str:
        # Cache the common small atoms without retaining potentially large
        # expression trees and their representations across all cones.
        if isinstance(literal[0], (Id, Const)):
            return simple_literal_key(literal)
        return repr(literal)

    @lru_cache(maxsize=2048)
    def simple_term_key(term: frozenset) -> tuple[str, ...]:
        return tuple(sorted(map(literal_key, term)))

    def term_key(term: frozenset) -> tuple[str, ...]:
        # repr(frozenset) preserves its hash-dependent iteration order.
        # Canonicalize literals as well as the outer collection: otherwise
        # identical RTL needlessly invalidates large compiler-cache shards.
        if all(isinstance(literal[0], (Id, Const)) for literal in term):
            return simple_term_key(term)
        return tuple(sorted(map(literal_key, term)))

    def simplify(terms: set[frozenset]) -> set[frozenset]:
        # Large CIRCT mux cones commonly contain pairs that differ only in
        # one literal polarity.  Looking for those pairs by scanning every
        # term against every other term made the 256-entry L2 sharer cones
        # quadratic in the number of paths.  An exact complement candidate is
        # determined by each literal and can be found in the hash set.
        # Keep the exhaustive absorption pass for small cones, where it is
        # cheap and catches the more general subset case.
        if len(terms) > 64:
            while True:
                merged = False
                snapshot = sorted(terms, key=term_key)
                for t in snapshot:
                    if t not in terms:
                        continue
                    for k, pol in sorted(t, key=literal_key):
                        candidate = (t - {(k, pol)}) | {(k, not pol)}
                        if candidate in terms:
                            terms.discard(t)
                            terms.discard(candidate)
                            terms.add(t - {(k, pol)})
                            merged = True
                            break
                    if merged:
                        break
                if not merged:
                    return terms
        changed = True
        while changed:
            changed = False
            by_size = [(len(t), t) for t in sorted(terms, key=lambda t: (len(t), term_key(t)))]
            for t_size, t in by_size:
                if t not in terms:
                    continue
                for s_size, s in by_size:
                    # Later terms cannot be subsets or one-literal reductions
                    # once their size exceeds this term's size.
                    if s_size > t_size:
                        break
                    if s is t or s not in terms:
                        continue
                    if s <= t:
                        terms.discard(t)
                        changed = True
                        break
                    diff = s - t
                    if len(diff) == 1:
                        (k, pol), = diff
                        if (k, not pol) in t:
                            nt = t - {(k, not pol)}
                            terms.discard(t)
                            if s_size == t_size:
                                terms.discard(s)
                            terms.add(nt)
                            changed = True
                            break
                if changed:
                    break
        return terms

    def build(lits: frozenset, negs: dict[Expr, Expr]) -> Expr:
        parts: list[Expr] = []
        for k, pol in sorted(lits, key=literal_key):
            if pol:
                parts.append(k)
            else:
                parts.append(negs.get(k) or UnaryOp("!", k))
        return _and_chain(parts)

    negs: dict[Expr, Expr] = {}
    for name, e in assigns.items():
        n = neg_of(e)
        if n is not None and name in sigs and not sigs[name].depth:
            negs.setdefault(resolve(n), Id(name))

    order: list[str] = []
    state: dict[str, int] = {}
    for root in assigns:
        stack = [(root, False)]
        while stack:
            name, done = stack.pop()
            if done:
                order.append(name)
                continue
            if state.get(name):
                continue
            state[name] = 1
            stack.append((name, True))
            for dep in expr_ids(assigns[name]):
                if dep in assigns and not state.get(dep):
                    stack.append((dep, False))

    out = assigns
    merged = 0
    for name in order:
        e = out[name]
        sig = sigs.get(name)
        if sig is None or sig.depth or sig.width != 1:
            continue
        if not (isinstance(e, BinOp) and e.op == "|"):
            continue
        parts = _flatten_op(e, "|")
        if len(parts) < 2 or len(parts) > _SOP_TERMS_MAX or not all(bit(p) for p in parts):
            continue
        opaque: set[Expr] = set()
        terms: list[frozenset] | None = None
        for _ in range(4):
            terms = []
            for p in parts:
                t = term_lits(p, opaque)
                if t is None:
                    terms = None
                    break
                terms.append(t)
            if terms is None:
                break
            neg_keys = {k for t in terms if t is not _SOP_FALSE for k, pol in t if not pol}
            grow = neg_keys - opaque
            if not grow:
                break
            opaque |= grow
        if terms is None:
            continue
        orig: dict[frozenset, Expr] = {}
        for p, t in zip(parts, terms):
            orig.setdefault(t, p)
        live = {t for t in terms if t is not _SOP_FALSE}
        n_before = len(set(terms))
        live = simplify(live)
        if len(live) >= n_before:
            continue
        merged += 1
        if not live:
            out[name] = Const(0, 1)
        elif frozenset() in live:
            out[name] = Const(1, 1)
        else:
            out[name] = _or_chain(
                [orig.get(t) or build(t, negs) for t in sorted(live, key=term_key)]
            )
    if merged:
        print(f"[sop] merged {merged} or-chains", file=sys.stderr)
    return out


_REASSOC_AND = os.environ.get("FLASHSIM_REASSOC_AND", "1") != "0"
_REASSOC_AND_MAX = 48


def _reassoc_and_chains(
    assigns: dict[str, Expr], sigs: dict[str, Signal]
) -> dict[str, Expr]:
    """Rebuild 1-bit AND chains so operands shared by many wires come first.

    CIRCT nests `a & (b & (c & e_i))` with the per-entry term innermost, so
    each of an L2 slice's per-entry enables recomputes the same 7-term global
    product. Flattening every chain to its operand set and rebuilding it
    left-deep in descending share count turns common prefixes into one shared
    wire, leaving one or two ANDs per entry.
    """

    def bit(e: Expr) -> bool:
        try:
            return expr_width(e, sigs) == 1
        except (KeyError, TypeError):
            # Leave an unknown-width cone untouched; final IR validation
            # distinguishes an unresolved signal from a valid temporary.
            return False

    flat: dict[str, frozenset[Expr] | None] = {}

    def literals(e: Expr) -> frozenset[Expr] | None:
        if isinstance(e, BinOp) and e.op == "&" and bit(e.a) and bit(e.b):
            a = literals(e.a)
            b = literals(e.b) if a is not None else None
            if a is None or b is None or len(a) + len(b) > _REASSOC_AND_MAX:
                return None
            return a | b
        if isinstance(e, Id) and e.name in flat:
            s = flat[e.name]
            return s if s is not None and len(s) > 1 else frozenset([e])
        return frozenset([e])

    order: list[str] = []
    state: dict[str, int] = {}
    for root in assigns:
        stack = [(root, False)]
        while stack:
            name, done = stack.pop()
            if done:
                order.append(name)
                continue
            if state.get(name):
                continue
            state[name] = 1
            stack.append((name, True))
            for dep in expr_ids(assigns[name]):
                if dep in assigns and not state.get(dep):
                    stack.append((dep, False))
    for name in order:
        e = assigns[name]
        sig = sigs.get(name)
        if (
            sig is None
            or sig.depth
            or sig.width != 1
            or not (isinstance(e, BinOp) and e.op == "&")
        ):
            continue
        try:
            flat[name] = literals(e)
        except RecursionError:
            # Deep opaque expressions need not be hashed or reassociated.
            # Retaining the original chain is safe and keeps large DUTs
            # buildable when Python's recursive dataclass hash reaches its
            # recursion limit.
            flat[name] = None

    chains = {n: s for n, s in flat.items() if s is not None and len(s) >= 3}
    if not chains:
        return assigns
    freq: Counter = Counter()
    for s in chains.values():
        freq.update(s)
    by_set: dict[frozenset[Expr], str] = {}
    for n in order:
        s = chains.get(n)
        if s is not None:
            by_set.setdefault(s, n)
    out = dict(assigns)
    counter = [0]

    def prefix(lits: list[Expr], stem: str) -> Expr:
        if len(lits) == 1:
            return lits[0]
        key = frozenset(lits)
        name = by_set.get(key)
        if name is None:
            while True:
                counter[0] += 1
                name = f"{stem}_t9{counter[0]}"
                if name not in sigs and name not in out:
                    break
            sigs[name] = Signal(name, 1, "wire")
            by_set[key] = name
            out[name] = BinOp("&", prefix(lits[:-1], stem), lits[-1])
        return Id(name)

    for n in order:
        s = chains.get(n)
        if s is None:
            continue
        try:
            lits = sorted(s, key=lambda e: (-freq[e], repr(e)))
        except RecursionError:
            # The deterministic tie-breaker's repr can reach its recursion
            # limit before hashing does. This chain is safe to leave as-is.
            continue
        stem = n[: n.rfind("_t")] if re.search(r"_t\d+$", n) else n
        out[n] = BinOp("&", prefix(lits[:-1], stem), lits[-1])
    return out


def _flatten_nested_holds(body: list[Stmt]) -> list[Stmt]:
    """Lift `if (c) then else if (e) hold` into skippable empty-else holds.

    `_nba_to_if` emits 1-bit enables as `if (!en) reset else if (extra) taken`.
    On GPU-sized nets those stay in `_nba_live` and run every cycle. Flattening
    puts both sides into hold buckets that GSIM wake can skip.

    An else arm is lifted only while it holds at most `_FLATTEN_ELSE_MAX`
    holds. Each lifted hold is prefixed with `!c`, so flattening a whole
    priority chain `if (c1) .. else if (c2) .. else if (cn)` makes arm k test
    k-1 negations: O(n^2) guard terms. An L2 slice's per-entry registers
    have ~50-arm chains, which put ~36 KB of guards into every entry's hold.
    """
    out: list[Stmt] = []
    for stmt in body:
        if not isinstance(stmt, If):
            out.append(stmt)
            continue
        then = _flatten_nested_holds(stmt.then_body)
        els = _flatten_nested_holds(stmt.else_body)
        if _all_holds(els) and len(els) <= _FLATTEN_ELSE_MAX:
            if _all_holds(then):
                for inner in then:
                    assert isinstance(inner, If)
                    out.append(If(BinOp("&", stmt.cond, inner.cond), list(inner.then_body), []))
            else:
                out.append(If(stmt.cond, then, []))
            ncond = _not_expr(stmt.cond)
            for inner in els:
                assert isinstance(inner, If)
                out.append(If(BinOp("&", ncond, inner.cond), list(inner.then_body), []))
        elif not els and _all_holds(then):
            for inner in then:
                assert isinstance(inner, If)
                out.append(If(BinOp("&", stmt.cond, inner.cond), list(inner.then_body), []))
        else:
            out.append(If(stmt.cond, then, els))
    return out


def _merge_hold_key(stmt: If) -> tuple[Expr, str] | None:
    """Identity for foldable holds: same cond and same write-partition bucket.

    Cross-partition merges would glue unrelated CU cones into one mega hold and
    wake them together. Mirror emit's write-target bucketing.
    """
    from flashsim.emit import skip_partition_key

    writes = stmt_writes(stmt)
    if not writes:
        return None
    part = max((skip_partition_key(w) for w in writes), key=len)
    return (stmt.cond, part)


def _merge_hold_ifs(body: list[Stmt]) -> list[Stmt]:
    """Collapse same-cond holds, even when non-adjacent and both have else.

    Chisel/GPU always_ff emits many `if (en) a <= ... else a <= ...` for each
    field. Adjacent-only / empty-else-only merge left vectorCoalescer cones
    duplicated under repeated `if (arbiter_t8)`. Folding every `if (c)` into the
    first matching `if (c)` (same write partition), concatenating then and else
    arms, is safe for NBA: arms only write `__n` and do not observe siblings.
    """
    out: list[Stmt] = []
    first: dict[tuple[Expr, str], int] = {}
    for stmt in body:
        if isinstance(stmt, If):
            stmt = If(
                stmt.cond,
                _merge_hold_ifs(stmt.then_body),
                _merge_hold_ifs(stmt.else_body),
            )
        if isinstance(stmt, If):
            key = _merge_hold_key(stmt)
            if key is not None and key in first:
                i = first[key]
                prev = out[i]
                assert isinstance(prev, If)
                out[i] = If(
                    prev.cond,
                    prev.then_body + stmt.then_body,
                    prev.else_body + stmt.else_body,
                )
                continue
            if key is not None:
                first[key] = len(out)
        out.append(stmt)
    return out


def _is_scalar_const_nba(stmt: Stmt, sigs: dict[str, Signal]) -> bool:
    return (
        isinstance(stmt, NbAssign)
        and isinstance(stmt.rhs, Const)
        and stmt.lhs in sigs
        and sigs[stmt.lhs].depth == 0
    )


def _value_differs(lhs: str, value: int, sigs: dict[str, Signal]) -> Expr:
    width = sigs[lhs].width
    if width == 1:
        return UnaryOp("!", Id(lhs)) if value else Id(lhs)
    return BinOp("!=", Id(lhs), Const(value, width))


def _self_gate_const_holds(body: list[Stmt], sigs: dict[str, Signal]) -> list[Stmt]:
    """`if (en) x <= c` is a no-op once x already holds c; skip it while idle."""
    out: list[Stmt] = []
    for stmt in body:
        if not isinstance(stmt, If):
            out.append(stmt)
            continue
        then = _self_gate_const_holds(stmt.then_body, sigs)
        els = _self_gate_const_holds(stmt.else_body, sigs)
        if not els and then and all(_is_scalar_const_nba(s, sigs) for s in then):
            diffs = [
                _value_differs(s.lhs, s.rhs.value, sigs)
                for s in then
                if isinstance(s, NbAssign) and isinstance(s.rhs, Const)
            ]
            nz = diffs[0]
            for part in diffs[1:]:
                nz = BinOp("|", nz, part)
            out.append(If(BinOp("&", stmt.cond, nz), then, []))
        else:
            out.append(If(stmt.cond, then, els))
    return out


def _sticky_rhs(lhs: str, rhs: Expr, sigs: dict[str, Signal]) -> bool:
    """True when `lhs <= rhs` can stay unchanged while the cone is quiet.

    Only wrap const and scalar-id copies. Complex next-state expressions can
    be wider than the C compare emit supports.
    """
    if lhs not in sigs or sigs[lhs].depth or sigs[lhs].width > 128:
        return False
    if isinstance(rhs, Const):
        return True
    if isinstance(rhs, Id):
        src = sigs.get(rhs.name)
        if src is not None and (src.depth or src.width > 128):
            return False
        return rhs.name != lhs
    return False


def _differs(lhs: str, rhs: Expr, sigs: dict[str, Signal]) -> Expr:
    if isinstance(rhs, Const):
        return _value_differs(lhs, rhs.value, sigs)
    if sigs[lhs].width == 1:
        return BinOp("^", Id(lhs), rhs)
    return BinOp("!=", Id(lhs), rhs)


def _wrap_sticky_nbas(body: list[Stmt], sigs: dict[str, Signal]) -> list[Stmt]:
    """Turn always-run `x <= y` into `if (x != y) x <= y` so idle copies skip.

    Also wraps const/id else-branches so `_flatten_nested_holds` can lift
    scoreboard resets and muxed copies out of `_nba_live`. Leave `x <= x+1`
    counters alone: those fire every cycle and must stay live.
    """
    out: list[Stmt] = []
    for stmt in body:
        if isinstance(stmt, NbAssign):
            if isinstance(stmt.rhs, Id) and stmt.rhs.name == stmt.lhs:
                continue
            if _sticky_rhs(stmt.lhs, stmt.rhs, sigs):
                out.append(If(_differs(stmt.lhs, stmt.rhs, sigs), [stmt], []))
            else:
                out.append(stmt)
        elif isinstance(stmt, If):
            out.append(
                If(
                    stmt.cond,
                    _wrap_sticky_nbas(stmt.then_body, sigs),
                    _wrap_sticky_nbas(stmt.else_body, sigs),
                )
            )
        else:
            out.append(stmt)
    return out


def _map_stmts(stmts: list[Stmt], fn) -> list[Stmt]:
    out: list[Stmt] = []
    for stmt in stmts:
        if isinstance(stmt, NbAssign):
            out.append(NbAssign(stmt.lhs, fn(stmt.rhs)))
        elif isinstance(stmt, If):
            out.append(
                If(fn(stmt.cond), _map_stmts(stmt.then_body, fn), _map_stmts(stmt.else_body, fn))
            )
        else:
            raise TypeError(stmt)
    return out


def _alias_expr(expr: Expr, alias: dict[str, str]) -> Expr:
    if not alias:
        return expr

    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name in alias:
            return Id(alias[node.name])
        return node

    return map_expr(expr, fn)


def _dedup_assigns(
    assigns: dict[str, Expr],
    body: list[Stmt],
    reset_epilogue: list[Stmt],
    mem_writes: list[MemWrite],
    assertions: list[ClockedAssertion | ClockedPrint],
    clocked_processes: list[ClockedProcess],
    sigs: dict[str, Signal],
) -> tuple[dict[str, Expr], list[Stmt], list[Stmt], list[MemWrite], list[ClockedAssertion | ClockedPrint], list[ClockedProcess]]:
    """Merge structurally identical wires (GSIM commonExpr / arcilator dedup).
    Only scalar wires are merged; array, reg, input, output and mem values stay.
    The smallest name is canonical so the result is PYTHONHASHSEED-independent.
    """
    if _DEDUP_MIN_NODES < 1:
        return assigns, body, reset_epilogue, mem_writes, assertions, clocked_processes
    groups: dict[Expr, list[str]] = {}
    for lhs, rhs in assigns.items():
        sig = sigs.get(lhs)
        if sig is None or sig.kind != "wire" or sig.depth:
            continue
        if isinstance(rhs, (Id, Const, ArrayZeros)):
            continue
        try:
            if _expr_nodes(rhs) < _DEDUP_MIN_NODES:
                continue
            groups.setdefault(rhs, []).append(lhs)
        except RecursionError:
            # Dataclass hashing can overflow the Python/C stack on very deep
            # expressions. Deduplication is optional; keep that wire intact.
            continue
    alias: dict[str, str] = {}
    for rhs, names in groups.items():
        if len(names) < 2:
            continue
        names = sorted(names)
        canonical = names[0]
        for dup in names[1:]:
            alias[dup] = canonical
    if not alias:
        return assigns, body, reset_epilogue, mem_writes, assertions, clocked_processes
    try:
        new_assigns = {n: _alias_expr(e, alias) for n, e in assigns.items() if n not in alias}
        new_body = _map_stmts(body, lambda e: _alias_expr(e, alias))
        new_reset = _map_stmts(reset_epilogue, lambda e: _alias_expr(e, alias))
        new_mem = [
            MemWrite(w.mem, _alias_expr(w.addr, alias), _alias_expr(w.data, alias), _alias_expr(w.enable, alias))
            for w in mem_writes
        ]
        new_assert = [map_observation(a, lambda expr: _alias_expr(expr, alias))
                      for a in assertions]
        new_procs = [ClockedProcess(_alias_expr(p.clock, alias), p.scope) for p in clocked_processes]
    except RecursionError:
        # Publishing only part of an alias rewrite would leave dangling wires.
        # Retain the complete original IR if any observer or use is too deep.
        return assigns, body, reset_epilogue, mem_writes, assertions, clocked_processes
    return new_assigns, new_body, new_reset, new_mem, new_assert, new_procs


def _expr_demands(expr: Expr, demand: int, sigs: dict[str, Signal]) -> dict[str, int]:
    """Leaf demands for the low `demand` bits of `expr` (GSIM usedBits backwards)."""
    out: dict[str, int] = {}
    stack: list[tuple[Expr, int]] = [(expr, demand)]
    while stack:
        node, need = stack.pop()
        if need <= 0:
            continue
        if isinstance(node, Id):
            prev = out.get(node.name, 0)
            if need > prev:
                out[node.name] = need
            continue
        if isinstance(node, Const) or isinstance(node, ArrayZeros):
            continue
        if isinstance(node, UnaryOp):
            stack.append((node.a, need))
            continue
        if isinstance(node, Extract):
            stack.append((node.a, node.low + need))
            continue
        if isinstance(node, Concat):
            widths: list[int] = []
            ok = True
            for part in node.parts:
                try:
                    widths.append(expr_width(part, sigs))
                except (KeyError, TypeError, ValueError):
                    ok = False
                    break
            if not ok:
                for part in node.parts:
                    for name in expr_ids(part):
                        out[name] = max(out.get(name, 0), sigs[name].width if name in sigs else 32)
                continue
            remain = need
            for idx in range(len(node.parts) - 1, -1, -1):
                if remain <= 0:
                    break
                take = min(widths[idx], remain)
                stack.append((node.parts[idx], take))
                remain -= take
            continue
        if isinstance(node, BinOp):
            if node.op in CMP_OPS:
                try:
                    stack.append((node.a, expr_width(node.a, sigs)))
                except (KeyError, TypeError, ValueError):
                    stack.append((node.a, 32))
                try:
                    stack.append((node.b, expr_width(node.b, sigs)))
                except (KeyError, TypeError, ValueError):
                    stack.append((node.b, 32))
                continue
            if node.op in {"+", "-", "&", "|", "^", "*"}:
                stack.append((node.a, need))
                stack.append((node.b, need))
                continue
            if node.op == "<<":
                if isinstance(node.b, Const):
                    stack.append((node.a, max(0, need - node.b.value)))
                else:
                    try:
                        stack.append((node.a, expr_width(node.a, sigs)))
                    except (KeyError, TypeError, ValueError):
                        stack.append((node.a, 32))
                try:
                    stack.append((node.b, expr_width(node.b, sigs)))
                except (KeyError, TypeError, ValueError):
                    stack.append((node.b, 32))
                continue
            try:
                stack.append((node.a, expr_width(node.a, sigs)))
            except (KeyError, TypeError, ValueError):
                stack.append((node.a, 32))
            try:
                stack.append((node.b, expr_width(node.b, sigs)))
            except (KeyError, TypeError, ValueError):
                stack.append((node.b, 32))
            continue
        if isinstance(node, Ternary):
            stack.append((node.cond, 1))
            stack.append((node.a, need))
            stack.append((node.b, need))
            continue
        if isinstance(node, MemRead):
            try:
                stack.append((node.addr, expr_width(node.addr, sigs)))
            except (KeyError, TypeError, ValueError):
                stack.append((node.addr, 32))
            continue
        if isinstance(node, ArrayGet):
            if isinstance(node.arr, Id) and node.arr.name in sigs:
                out[node.arr.name] = max(out.get(node.arr.name, 0), sigs[node.arr.name].width)
            else:
                try:
                    stack.append((node.arr, expr_width(node.arr, sigs)))
                except (KeyError, TypeError, ValueError):
                    stack.append((node.arr, 32))
            try:
                stack.append((node.index, expr_width(node.index, sigs)))
            except (KeyError, TypeError, ValueError):
                stack.append((node.index, 32))
            continue
        if isinstance(node, ArrayInject):
            if isinstance(node.arr, Id) and node.arr.name in sigs:
                out[node.arr.name] = max(out.get(node.arr.name, 0), sigs[node.arr.name].width)
            else:
                try:
                    stack.append((node.arr, expr_width(node.arr, sigs)))
                except (KeyError, TypeError, ValueError):
                    stack.append((node.arr, 32))
            try:
                stack.append((node.index, expr_width(node.index, sigs)))
            except (KeyError, TypeError, ValueError):
                stack.append((node.index, 32))
            stack.append((node.value, need))
            continue
        raise TypeError(type(node))
    return out


def _truncate_to_used(expr: Expr, need: int, sigs: dict[str, Signal]) -> Expr:
    """Expression computing only the low `need` bits of `expr` (GSIM usedBits)."""
    if need <= 0:
        return Const(0, 1)
    try:
        width = expr_width(expr, sigs)
    except (KeyError, TypeError, ValueError):
        return expr
    if need >= width:
        return expr
    if isinstance(expr, Const):
        return Const(expr.value & ((1 << need) - 1), need)
    if isinstance(expr, Id):
        sig = sigs.get(expr.name)
        if sig is not None and sig.depth:
            return expr
        return Extract(expr, 0, need)
    if isinstance(expr, UnaryOp):
        if expr.op in {"~", "!"}:
            return UnaryOp(expr.op, _truncate_to_used(expr.a, need, sigs))
        return Extract(expr, 0, need)
    if isinstance(expr, Extract):
        eff_need = min(need, expr.width)
        if isinstance(expr.a, Extract):
            inner = expr.a
            eff_low = inner.low + expr.low
            return Extract(_truncate_to_used(inner.a, eff_low + eff_need, sigs), eff_low, eff_need)
        return Extract(_truncate_to_used(expr.a, expr.low + eff_need, sigs), expr.low, eff_need)
    if isinstance(expr, Concat):
        widths: list[int] = []
        for part in expr.parts:
            try:
                widths.append(expr_width(part, sigs))
            except (KeyError, TypeError, ValueError):
                return expr
        parts: list[Expr] = []
        remain = need
        for idx in range(len(expr.parts) - 1, -1, -1):
            if remain <= 0:
                break
            take = min(widths[idx], remain)
            part = expr.parts[idx]
            parts.append(part if take == widths[idx] else _truncate_to_used(part, take, sigs))
            remain -= take
        if not parts:
            return Const(0, need)
        parts.reverse()
        if len(parts) == 1:
            return parts[0]
        return Concat(tuple(parts))
    if isinstance(expr, BinOp):
        if expr.op in CMP_OPS:
            return expr
        if expr.op in {"+", "-", "&", "|", "^", "*"}:
            return BinOp(expr.op, _truncate_to_used(expr.a, need, sigs), _truncate_to_used(expr.b, need, sigs))
        if expr.op == "<<":
            if isinstance(expr.b, Const):
                shift = expr.b.value
                if need <= shift:
                    return Const(0, need)
                return BinOp("<<", _truncate_to_used(expr.a, need - shift, sigs), expr.b)
            return Extract(expr, 0, need)
        if expr.op == ">>":
            if isinstance(expr.b, Const):
                shift = expr.b.value
                if shift + need <= expr_width(expr.a, sigs):
                    return Extract(_truncate_to_used(expr.a, shift + need, sigs), shift, need)
            return Extract(expr, 0, need)
        return Extract(expr, 0, need)
    if isinstance(expr, Ternary):
        return Ternary(expr.cond, _truncate_to_used(expr.a, need, sigs), _truncate_to_used(expr.b, need, sigs))
    if isinstance(expr, (MemRead, ArrayGet)):
        return Extract(expr, 0, need)
    return expr


def _apply_used_bits(
    assigns: dict[str, Expr],
    body: list[Stmt],
    reset_epilogue: list[Stmt],
    mem_writes: list[MemWrite],
    sigs: dict[str, Signal],
    observed_exprs: list[Expr] = (),
) -> tuple[dict[str, Expr], dict[str, Signal]]:
    """Shrink internal wires to their consumed bits (GSIM usedBits)."""
    used: dict[str, int] = {name: 0 for name in sigs}
    for name, sig in sigs.items():
        if sig.kind in {"output", "reg", "mem"}:
            used[name] = sig.width
    body_pairs: list[tuple[Expr, int]] = []

    def walk(stmts: list[Stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, NbAssign):
                target = sigs.get(stmt.lhs)
                body_pairs.append((stmt.rhs, target.width if target is not None else 32))
            elif isinstance(stmt, If):
                body_pairs.append((stmt.cond, 1))
                walk(stmt.then_body)
                walk(stmt.else_body)

    walk(body)
    walk(reset_epilogue)
    for expr in observed_exprs:
        body_pairs.append((expr, expr_width(expr, sigs)))
    for wr in mem_writes:
        try:
            body_pairs.append((wr.addr, expr_width(wr.addr, sigs)))
        except (KeyError, TypeError, ValueError):
            body_pairs.append((wr.addr, 32))
        mem = sigs.get(wr.mem)
        body_pairs.append((wr.data, mem.width if mem is not None else 32))
        body_pairs.append((wr.enable, 1))
    queue: deque[str] = deque()
    queued: set[str] = set()

    def enqueue(name: str) -> None:
        if name in assigns and name not in queued:
            queue.append(name)
            queued.add(name)

    def bump(name: str, need: int) -> None:
        if name not in used:
            return
        sig = sigs.get(name)
        if sig is not None:
            need = min(need, sig.width)
        if need > used[name]:
            used[name] = need
            enqueue(name)

    for expr, need in body_pairs:
        for name, leaf_need in _expr_demands(expr, need, sigs).items():
            bump(name, leaf_need)
    for name, need in used.items():
        if need > 0:
            enqueue(name)
    while queue:
        name = queue.popleft()
        queued.discard(name)
        rhs = assigns.get(name)
        if rhs is None:
            continue
        for leaf, leaf_need in _expr_demands(rhs, used[name], sigs).items():
            bump(leaf, leaf_need)
    new_sigs: dict[str, Signal] = {}
    for name, sig in sigs.items():
        need = used.get(name, 0)
        if sig.kind == "wire" and not sig.depth and need > 0 and need < sig.width:
            new_sigs[name] = Signal(name, need, "wire", sig.depth)
        else:
            new_sigs[name] = sig
    new_assigns: dict[str, Expr] = {}
    for name, rhs in assigns.items():
        need = used.get(name, 0)
        if name in new_sigs and new_sigs[name].kind == "wire" and not new_sigs[name].depth and need > 0 and need < sigs[name].width:
            new_assigns[name] = _truncate_to_used(rhs, need, sigs)
        else:
            new_assigns[name] = rhs
    return new_assigns, new_sigs


def _eval_lut_expr(expr: Expr, leaf_vals: dict[str, int], sigs: dict[str, Signal]) -> int:
    if isinstance(expr, Id):
        return leaf_vals[expr.name]
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, UnaryOp):
        a = _eval_lut_expr(expr.a, leaf_vals, sigs)
        if expr.op in {"~", "!"}:
            width = expr_width(expr.a, sigs)
            return (~a) & ((1 << width) - 1) if width else 0
        raise TypeError(expr.op)
    if isinstance(expr, BinOp):
        a = _eval_lut_expr(expr.a, leaf_vals, sigs)
        b = _eval_lut_expr(expr.b, leaf_vals, sigs)
        if expr.op == "&":
            return a & b
        if expr.op == "|":
            return a | b
        if expr.op == "^":
            return a ^ b
        if expr.op in CMP_OPS:
            if expr.op in {"s<", "s<=", "s>", "s>="}:
                width = expr_width(expr.a, sigs)
                sa, sb = _as_signed(a, width), _as_signed(b, width)
                return int({"s<": sa < sb, "s<=": sa <= sb, "s>": sa > sb, "s>=": sa >= sb}[expr.op])
            return int({"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[expr.op])
        if expr.op in {"+", "-", "*"}:
            width = expr_width(expr, sigs)
            result = {"+": a + b, "-": a - b, "*": a * b}[expr.op]
            return result & ((1 << width) - 1)
        if expr.op in {"<<", ">>"}:
            width = expr_width(expr, sigs)
            shifted = (a << b) if expr.op == "<<" else (a >> b)
            return shifted & ((1 << width) - 1)
        raise TypeError(expr.op)
    if isinstance(expr, Extract):
        return (_eval_lut_expr(expr.a, leaf_vals, sigs) >> expr.low) & ((1 << expr.width) - 1)
    if isinstance(expr, Concat):
        acc = 0
        for part in expr.parts:
            width = expr_width(part, sigs)
            acc = (acc << width) | (_eval_lut_expr(part, leaf_vals, sigs) & ((1 << width) - 1))
        return acc
    if isinstance(expr, Ternary):
        return _eval_lut_expr(expr.a if _eval_lut_expr(expr.cond, leaf_vals, sigs) else expr.b, leaf_vals, sigs)
    raise TypeError(type(expr))


def _lut_eligible(expr: Expr, sigs: dict[str, Signal]) -> list[str] | None:
    leaves = sorted(expr_ids(expr))
    total = 0
    for name in leaves:
        sig = sigs.get(name)
        if sig is None or sig.depth:
            return None
        total += sig.width
        if total > _LUT_MAX_BITS:
            return None
    if total < 3:
        return None
    if _expr_nodes(expr) < _LUT_MIN_NODES:
        return None
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, (MemRead, ArrayGet, ArrayInject, ArrayZeros)):
            return None
        if isinstance(node, BinOp) and node.op in {"/", "%", "s/", "s%"}:
            return None
        if isinstance(node, Extract):
            if isinstance(node.a, Id):
                sig = sigs.get(node.a.name)
                if sig is not None and not (node.low == 0 and node.width == sig.width):
                    return None
            stack.append(node.a)
        elif isinstance(node, (UnaryOp,)):
            stack.append(node.a)
        elif isinstance(node, BinOp):
            stack.append(node.a)
            stack.append(node.b)
        elif isinstance(node, Ternary):
            stack.append(node.cond)
            stack.append(node.a)
            stack.append(node.b)
        elif isinstance(node, Concat):
            stack.extend(node.parts)
    return leaves


def _apply_lut(assigns: dict[str, Expr], sigs: dict[str, Signal]) -> dict[str, Expr]:
    out: dict[str, Expr] = {}
    for lhs, rhs in assigns.items():
        sig = sigs.get(lhs)
        replaced = rhs
        if sig is not None and sig.kind == "wire" and not sig.depth:
            try:
                width = expr_width(rhs, sigs)
            except (KeyError, TypeError, ValueError):
                width = 0
            if width == 1:
                leaves = _lut_eligible(rhs, sigs)
                if leaves is not None:
                    total = sum(sigs[name].width for name in leaves)
                    table = 0
                    for idx in range(1 << total):
                        leaf_vals: dict[str, int] = {}
                        remain = total
                        for name in leaves:
                            w = sigs[name].width
                            remain -= w
                            leaf_vals[name] = (idx >> remain) & ((1 << w) - 1)
                        if _eval_lut_expr(rhs, leaf_vals, sigs) & 1:
                            table |= 1 << idx
                    entries = 1 << total
                    if table == 0:
                        replaced = Const(0, 1)
                    elif table == (1 << entries) - 1:
                        replaced = Const(1, 1)
                    else:
                        bits_width = 1 << total
                        table_const = Const(table, bits_width)
                        idx_expr: Expr = Id(leaves[0]) if len(leaves) == 1 else Concat(tuple(Id(name) for name in leaves))
                        replaced = Extract(BinOp(">>", table_const, idx_expr), 0, 1)
        out[lhs] = replaced
    return out
