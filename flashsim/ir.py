from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Expr:
    pass


@dataclass(frozen=True, slots=True)
class Id(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: int
    width: int = 32


@dataclass(frozen=True, slots=True)
class UnaryOp(Expr):
    op: str
    a: Expr


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    op: str
    a: Expr
    b: Expr


@dataclass(frozen=True, slots=True)
class Ternary(Expr):
    cond: Expr
    a: Expr
    b: Expr


@dataclass(frozen=True, slots=True)
class Extract(Expr):
    a: Expr
    low: int
    width: int


@dataclass(frozen=True, slots=True)
class Concat(Expr):
    parts: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class ArrayGet(Expr):
    arr: Expr
    index: Expr


@dataclass(frozen=True, slots=True)
class ArrayInject(Expr):
    arr: Expr
    index: Expr
    value: Expr


@dataclass(frozen=True, slots=True)
class ArrayZeros(Expr):
    depth: int
    width: int


@dataclass(frozen=True, slots=True)
class ArrayConst(Expr):
    """Array elements in index order, starting at index zero."""
    values: tuple[int, ...]
    width: int

    @property
    def depth(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class MemRead(Expr):
    mem: str
    addr: Expr


@dataclass(slots=True)
class NbAssign:
    lhs: str
    rhs: Expr


@dataclass(slots=True)
class If:
    cond: Expr
    then_body: list[Stmt]
    else_body: list[Stmt] = field(default_factory=list)


Stmt = NbAssign | If


@dataclass(slots=True)
class Signal:
    name: str
    width: int
    kind: str  # input, output, reg, wire, mem
    depth: int = 0  # 0 = scalar; otherwise unpacked array length


@dataclass(slots=True)
class Assign:
    lhs: str
    rhs: Expr


@dataclass(slots=True)
class MemWrite:
    mem: str
    addr: Expr
    data: Expr
    enable: Expr


@dataclass(slots=True)
class AlwaysFF:
    clock: str
    body: list[Stmt]


@dataclass(slots=True)
class ClockedAssertion:
    clock: Expr
    failure: Expr
    message: str
    scope: str


@dataclass(frozen=True, slots=True)
class DecimalFormat:
    value: Expr
    width: int
    field_width: int | None = None


@dataclass(slots=True)
class ClockedPrint:
    clock: Expr
    failure: Expr  # The guard for this observable effect.
    pieces: tuple[str | DecimalFormat, ...]
    scope: str
    stream: str


def observation_exprs(observation: ClockedAssertion | ClockedPrint) -> list[Expr]:
    values = [observation.clock, observation.failure]
    if isinstance(observation, ClockedPrint):
        values.extend(piece.value for piece in observation.pieces
                      if isinstance(piece, DecimalFormat))
    return values


def map_observation(observation: ClockedAssertion | ClockedPrint,
                    transform: Callable[[Expr], Expr]) -> ClockedAssertion | ClockedPrint:
    if isinstance(observation, ClockedPrint):
        pieces = tuple(DecimalFormat(transform(piece.value), piece.width, piece.field_width)
                       if isinstance(piece, DecimalFormat) else piece
                       for piece in observation.pieces)
        return ClockedPrint(transform(observation.clock), transform(observation.failure),
                            pieces, observation.scope, observation.stream)
    return ClockedAssertion(transform(observation.clock), transform(observation.failure),
                            observation.message, observation.scope)


@dataclass(slots=True)
class ClockedProcess:
    clock: Expr
    scope: str


@dataclass
class Module:
    name: str
    signals: dict[str, Signal]
    assigns: list[Assign]
    always: AlwaysFF
    ports: list[str]
    mem_writes: list[MemWrite] = field(default_factory=list)
    assertions: list[ClockedAssertion | ClockedPrint] = field(default_factory=list)
    clocked_processes: list[ClockedProcess] = field(default_factory=list)
    # Constant reset arms lifted off the per-register fast path. Emitted after
    # hold dispatch so a later write still wins when buckets are not in
    # source order.
    reset_epilogue: list[Stmt] = field(default_factory=list)


def signal_map(mod: Module) -> dict[str, Signal]:
    return mod.signals


class IdSet(set):
    """A set of signal names that iterates in a canonical (sorted) order.

    Plain `set` iteration over strings follows PYTHONHASHSEED, and the order
    leaks into the emitted C++: dep lists drive the order of `eval_*` calls
    inside a method and the order of struct fields. Two runs on identical
    input then produce different C++, which guarantees a ccache miss on every
    ARTI rebuild and makes a binary unreproducible from source. Sorting here
    fixes every current and future call site at once; set semantics are
    unchanged. Note that set *operators* return a plain `set`, so callers that
    combine two IdSets and then iterate still need their own `sorted(...)`.
    """

    def __iter__(self):
        return iter(sorted(set.__iter__(self)))


def expr_ids(expr: Expr) -> IdSet:
    return IdSet(expr_id_counts(expr))


def expr_id_counts(expr: Expr) -> dict[str, int]:
    counts: dict[str, int] = {}
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Id):
            counts[node.name] = counts.get(node.name, 0) + 1
            continue
        if isinstance(node, (Const, ArrayZeros, ArrayConst)):
            continue
        if isinstance(node, UnaryOp) or isinstance(node, Extract):
            stack.append(node.a)
            continue
        if isinstance(node, BinOp):
            stack.append(node.b)
            stack.append(node.a)
            continue
        if isinstance(node, Ternary):
            stack.append(node.b)
            stack.append(node.a)
            stack.append(node.cond)
            continue
        if isinstance(node, Concat):
            stack.extend(node.parts)
            continue
        if isinstance(node, ArrayGet):
            stack.append(node.index)
            stack.append(node.arr)
            continue
        if isinstance(node, ArrayInject):
            stack.append(node.value)
            stack.append(node.index)
            stack.append(node.arr)
            continue
        if isinstance(node, MemRead):
            counts[node.mem] = counts.get(node.mem, 0) + 1
            stack.append(node.addr)
            continue
        raise TypeError(type(node))
    return counts


def map_expr(expr: Expr, fn, *, memo: dict[int, tuple[Expr, Expr]] | None = None) -> Expr:
    """Rewrite each occurrence, or memoize a pure rewrite when explicitly asked.

    The default still invokes the visitor for every occurrence. A supplied
    memo retains original nodes as well as results to prevent identity reuse.
    Use one memo only with the same immutable transformation and input graph.
    """
    if memo is not None:
        cached = memo.get(id(expr))
        if cached is not None:
            return cached[1]
    if isinstance(expr, Id) or isinstance(expr, Const):
        result = fn(expr)
    elif isinstance(expr, UnaryOp):
        a = map_expr(expr.a, fn, memo=memo)
        result = fn(expr if a is expr.a else UnaryOp(expr.op, a))
    elif isinstance(expr, BinOp):
        a, b = map_expr(expr.a, fn, memo=memo), map_expr(expr.b, fn, memo=memo)
        result = fn(expr if a is expr.a and b is expr.b else BinOp(expr.op, a, b))
    elif isinstance(expr, Ternary):
        cond = map_expr(expr.cond, fn, memo=memo)
        a, b = map_expr(expr.a, fn, memo=memo), map_expr(expr.b, fn, memo=memo)
        unchanged = cond is expr.cond and a is expr.a and b is expr.b
        result = fn(expr if unchanged else Ternary(cond, a, b))
    elif isinstance(expr, Extract):
        a = map_expr(expr.a, fn, memo=memo)
        result = fn(expr if a is expr.a else Extract(a, expr.low, expr.width))
    elif isinstance(expr, Concat):
        parts = tuple(map_expr(p, fn, memo=memo) for p in expr.parts)
        unchanged = all(a is b for a, b in zip(parts, expr.parts))
        result = fn(expr if unchanged else Concat(parts))
    elif isinstance(expr, ArrayGet):
        arr = map_expr(expr.arr, fn, memo=memo)
        index = map_expr(expr.index, fn, memo=memo)
        result = fn(expr if arr is expr.arr and index is expr.index else ArrayGet(arr, index))
    elif isinstance(expr, ArrayInject):
        arr = map_expr(expr.arr, fn, memo=memo)
        index = map_expr(expr.index, fn, memo=memo)
        value = map_expr(expr.value, fn, memo=memo)
        unchanged = arr is expr.arr and index is expr.index and value is expr.value
        result = fn(expr if unchanged else ArrayInject(arr, index, value))
    elif isinstance(expr, (ArrayZeros, ArrayConst)):
        result = fn(expr)
    elif isinstance(expr, MemRead):
        addr = map_expr(expr.addr, fn, memo=memo)
        result = fn(expr if addr is expr.addr else MemRead(expr.mem, addr))
    else:
        raise TypeError(type(expr))
    if memo is not None:
        memo[id(expr)] = (expr, result)
    return result


def replace_id(expr: Expr, name: str, repl: Expr) -> Expr:
    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name == name:
            return repl
        return node

    return map_expr(expr, fn)


CMP_OPS = {"==", "!=", "<", "<=", ">", ">=", "s<", "s<=", "s>", "s>="}


def expr_width(expr: Expr, sigs: Mapping[str, Signal]) -> int:
    if isinstance(expr, Const):
        return expr.width
    if isinstance(expr, Id):
        return sigs[expr.name].width
    if isinstance(expr, Extract):
        return expr.width
    if isinstance(expr, Concat):
        return sum(expr_width(part, sigs) for part in expr.parts)
    if isinstance(expr, UnaryOp):
        return expr_width(expr.a, sigs)
    if isinstance(expr, BinOp):
        if expr.op in CMP_OPS:
            return 1
        if expr.op in {"<<", ">>"}:
            return expr_width(expr.a, sigs)
        return max(expr_width(expr.a, sigs), expr_width(expr.b, sigs))
    if isinstance(expr, Ternary):
        return expr_width(expr.a, sigs)
    if isinstance(expr, ArrayGet):
        return expr_width(expr.arr, sigs) if not isinstance(expr.arr, Id) else sigs[expr.arr.name].width
    if isinstance(expr, ArrayInject):
        return expr_width(expr.arr, sigs)
    if isinstance(expr, (ArrayZeros, ArrayConst)):
        return expr.width
    if isinstance(expr, MemRead):
        return sigs[expr.mem].width
    raise TypeError(type(expr))


def stmt_reads(stmt: Stmt) -> set[str]:
    if isinstance(stmt, NbAssign):
        return expr_ids(stmt.rhs)
    reads = expr_ids(stmt.cond)
    for s in stmt.then_body + stmt.else_body:
        reads |= stmt_reads(s)
    return reads


def stmt_writes(stmt: Stmt) -> set[str]:
    if isinstance(stmt, NbAssign):
        return {stmt.lhs}
    writes: set[str] = set()
    for s in stmt.then_body + stmt.else_body:
        writes |= stmt_writes(s)
    return writes
