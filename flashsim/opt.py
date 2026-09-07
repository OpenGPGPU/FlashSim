"""Simplify FlashSim IR so activity skipping sees muxes, not SSA soup.

CIRCT lowers `x << n` to extract/concat and gives every op an SSA name.
Emitting each name as a struct field + `eval_*` kills both skip and the busy
path. Fold constants, restore shifts, and inline single-use ops into users.
Keep values that feed `always_ff` named so `__ok` can still skip a sticky cone.
"""

from __future__ import annotations

from flashsim.ir import (
    AlwaysFF,
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    Assign,
    BinOp,
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
    replace_id,
    stmt_reads,
    stmt_writes,
)


def optimize(mod: Module) -> Module:
    assigns = {a.lhs: a.rhs for a in mod.assigns}
    sigs = dict(mod.signals)
    body = mod.always.body
    outputs = {n for n, s in sigs.items() if s.kind == "output"}
    regs = {n for n, s in sigs.items() if s.kind == "reg"}
    inputs = {n for n, s in sigs.items() if s.kind == "input"}
    mems = {n for n, s in sigs.items() if s.kind == "mem"}
    keep = outputs | regs | inputs | mems | {mod.always.clock}

    assigns = _subst_consts(assigns)
    assigns = {k: _rewrite_expr(v, assigns, sigs) for k, v in assigns.items()}
    body = _map_stmts(body, lambda e: _rewrite_expr(e, assigns, sigs))
    mem_writes = [
        MemWrite(
            w.mem,
            _rewrite_expr(w.addr, assigns, sigs),
            _rewrite_expr(w.data, assigns, sigs),
            _rewrite_expr(w.enable, assigns, sigs),
        )
        for w in mod.mem_writes
    ]
    assigns, body = _inline_single_use(assigns, body, outputs, mem_writes)
    assigns = {k: _fold_expr(v) for k, v in assigns.items()}
    body = _map_stmts(body, _fold_expr)
    mem_writes = [
        MemWrite(w.mem, _fold_expr(w.addr), _fold_expr(w.data), _fold_expr(w.enable))
        for w in mem_writes
    ]
    body = _lower_seq_mux(body, assigns, sigs)
    body = _flatten_nested_holds(body)
    body = _merge_hold_ifs(body)
    body = _self_gate_const_holds(body, sigs)
    n_writes = 0
    seen_w: set[str] = set()
    for stmt in body:
        for name in stmt_writes(stmt):
            if name not in seen_w:
                seen_w.add(name)
                n_writes += 1
    if n_writes >= 256:
        body = _wrap_sticky_nbas(body, sigs)
        body = _flatten_nested_holds(body)
        body = _merge_hold_ifs(body)
        body = _self_gate_const_holds(body, sigs)
    assigns, body = _dce(assigns, body, outputs, mem_writes)

    referred: set[str] = set(assigns)
    for expr in assigns.values():
        referred |= expr_ids(expr)
    for stmt in body:
        referred |= stmt_reads(stmt) | stmt_writes(stmt)
    for wr in mem_writes:
        referred |= expr_ids(wr.addr) | expr_ids(wr.data) | expr_ids(wr.enable)
        referred.add(wr.mem)
    new_sigs: dict[str, Signal] = {
        n: sigs[n] for n in (keep | referred) if n in sigs
    }
    for name, expr in assigns.items():
        if name in new_sigs:
            continue
        width = expr_width(expr, {**sigs, **new_sigs})
        depth = sigs[name].depth if name in sigs else 0
        new_sigs[name] = Signal(name, width, "wire", depth)
    new_assigns = [Assign(lhs, rhs) for lhs, rhs in assigns.items()]
    for name, expr in assigns.items():
        for dep in expr_ids(expr):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used by {name}")
    for stmt in body:
        for dep in stmt_reads(stmt):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used in sequential body")
    for wr in mem_writes:
        for dep in expr_ids(wr.addr) | expr_ids(wr.data) | expr_ids(wr.enable):
            if dep not in assigns and dep not in keep:
                raise ValueError(f"dangling wire {dep} used by mem write")
    return Module(
        mod.name, new_sigs, new_assigns, AlwaysFF(mod.always.clock, body), mod.ports, mem_writes
    )


def _subst_consts(assigns: dict[str, Expr]) -> dict[str, Expr]:
    consts = {n: e for n, e in assigns.items() if isinstance(e, Const)}

    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name in consts:
            return consts[node.name]
        return node

    out = {n: map_expr(e, fn) for n, e in assigns.items() if n not in consts}
    return out


def _unwrap(expr: Expr, assigns: dict[str, Expr], seen: set[str] | None = None) -> Expr:
    if seen is None:
        seen = set()
    if isinstance(expr, Id) and expr.name in assigns and expr.name not in seen:
        seen.add(expr.name)
        return _unwrap(assigns[expr.name], assigns, seen)
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


def _inline_single_use(
    assigns: dict[str, Expr],
    body: list[Stmt],
    outputs: set[str],
    mem_writes: list[MemWrite] | None = None,
) -> tuple[dict[str, Expr], list[Stmt]]:
    writes = mem_writes or []
    while True:
        uses = _count_uses(assigns, body, writes)
        users: dict[str, list[str]] = {}
        for lhs, rhs in assigns.items():
            for name in expr_ids(rhs):
                if name in assigns:
                    users.setdefault(name, []).append(lhs)
        changed = False
        # Use current RHS when substituting: inlining A into B then B into C in
        # the same pass must copy B's updated body, not the snapshot that still
        # names A after A was deleted.
        for name in list(assigns):
            if name not in assigns or name in outputs:
                continue
            n = uses.get(name, 0)
            if n == 0:
                del assigns[name]
                changed = True
                continue
            if n != 1:
                continue
            loc = users.get(name, [])
            if len(loc) != 1:
                continue
            user = loc[0]
            if user not in assigns or user == name:
                continue
            assigns[user] = replace_id(assigns[user], name, assigns[name])
            del assigns[name]
            changed = True
        if not changed:
            return assigns, body


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
    changed = True
    while changed:
        changed = False
        for name in list(live):
            if name not in assigns:
                continue
            for dep in expr_ids(assigns[name]):
                if dep not in live:
                    live.add(dep)
                    changed = True
    assigns = {n: e for n, e in assigns.items() if n in live}
    return assigns, body


def _lower_seq_mux(body: list[Stmt], assigns: dict[str, Expr], sigs: dict[str, Signal]) -> list[Stmt]:
    """Turn `reg <= mux(...)` back into `if` so mix cones are not always demanded."""
    out: list[Stmt] = []
    for stmt in body:
        if isinstance(stmt, NbAssign):
            out.extend(_nba_to_if(stmt.lhs, _follow(stmt.rhs, assigns, stmt.lhs, sigs), sigs))
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
    if isinstance(expr, Ternary):
        then_body = _nba_to_if(lhs, expr.a, sigs)
        else_body = _nba_to_if(lhs, expr.b, sigs)
        if _is_hold(lhs, else_body):
            else_body = []
        elif _is_hold(lhs, then_body):
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


def _flatten_nested_holds(body: list[Stmt]) -> list[Stmt]:
    """Lift `if (c) then else if (e) hold` into skippable empty-else holds.

    `_nba_to_if` emits 1-bit enables as `if (!en) reset else if (extra) taken`.
    On GPU-sized nets those stay in `_nba_live` and run every cycle. Flattening
    puts both sides into hold buckets that GSIM wake can skip.
    """
    out: list[Stmt] = []
    for stmt in body:
        if not isinstance(stmt, If):
            out.append(stmt)
            continue
        then = _flatten_nested_holds(stmt.then_body)
        els = _flatten_nested_holds(stmt.else_body)
        if _all_holds(els):
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


def _merge_hold_ifs(body: list[Stmt]) -> list[Stmt]:
    """Collapse adjacent `if (en) a <= ...; if (en) b <= ...` into one branch."""
    out: list[Stmt] = []
    for stmt in body:
        if isinstance(stmt, If):
            stmt = If(
                stmt.cond,
                _merge_hold_ifs(stmt.then_body),
                _merge_hold_ifs(stmt.else_body),
            )
        if (
            out
            and isinstance(out[-1], If)
            and isinstance(stmt, If)
            and not out[-1].else_body
            and not stmt.else_body
            and out[-1].cond == stmt.cond
        ):
            out[-1] = If(stmt.cond, out[-1].then_body + stmt.then_body, [])
        else:
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
