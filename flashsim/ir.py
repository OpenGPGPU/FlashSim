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


def _expr_children(node: Expr) -> tuple[Expr, ...]:
    if isinstance(node, (Id, Const, ArrayZeros, ArrayConst)):
        return ()
    if isinstance(node, (UnaryOp, Extract)):
        return (node.a,)
    if isinstance(node, BinOp):
        return (node.a, node.b)
    if isinstance(node, Ternary):
        return (node.cond, node.a, node.b)
    if isinstance(node, Concat):
        return node.parts
    if isinstance(node, ArrayGet):
        return (node.arr, node.index)
    if isinstance(node, ArrayInject):
        return (node.arr, node.index, node.value)
    if isinstance(node, MemRead):
        return (node.addr,)
    raise TypeError(type(node))


def expr_ids(expr: Expr) -> IdSet:
    names = IdSet()
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Id):
            names.add(node.name)
        elif isinstance(node, MemRead):
            names.add(node.mem)
        stack.extend(_expr_children(node))
    return names


def expr_id_counts(expr: Expr) -> dict[str, int]:
    # Propagate path multiplicities through distinct DAG nodes. Deduplicating
    # visits alone would undercount a shared input used by both operands.
    postorder: list[Expr] = []
    map_expr_memoized(expr, lambda node: postorder.append(node) or node)
    weights = {id(expr): 1}
    counts: dict[str, int] = {}
    for node in reversed(postorder):
        weight = weights.pop(id(node))
        if isinstance(node, (Id, MemRead)):
            name = node.name if isinstance(node, Id) else node.mem
            counts[name] = counts.get(name, 0) + weight
        for child in _expr_children(node):
            key = id(child)
            weights[key] = weights.get(key, 0) + weight
    return counts


def map_expr(expr: Expr, fn) -> Expr:
    if isinstance(expr, Id) or isinstance(expr, Const):
        return fn(expr)
    if isinstance(expr, UnaryOp):
        a = map_expr(expr.a, fn)
        return fn(expr if a is expr.a else UnaryOp(expr.op, a))
    if isinstance(expr, BinOp):
        a, b = map_expr(expr.a, fn), map_expr(expr.b, fn)
        return fn(expr if a is expr.a and b is expr.b else BinOp(expr.op, a, b))
    if isinstance(expr, Ternary):
        cond = map_expr(expr.cond, fn)
        a, b = map_expr(expr.a, fn), map_expr(expr.b, fn)
        unchanged = cond is expr.cond and a is expr.a and b is expr.b
        return fn(expr if unchanged else Ternary(cond, a, b))
    if isinstance(expr, Extract):
        a = map_expr(expr.a, fn)
        return fn(expr if a is expr.a else Extract(a, expr.low, expr.width))
    if isinstance(expr, Concat):
        parts = tuple(map_expr(p, fn) for p in expr.parts)
        unchanged = all(a is b for a, b in zip(parts, expr.parts))
        return fn(expr if unchanged else Concat(parts))
    if isinstance(expr, ArrayGet):
        arr, index = map_expr(expr.arr, fn), map_expr(expr.index, fn)
        return fn(expr if arr is expr.arr and index is expr.index else ArrayGet(arr, index))
    if isinstance(expr, ArrayInject):
        arr, index = map_expr(expr.arr, fn), map_expr(expr.index, fn)
        value = map_expr(expr.value, fn)
        unchanged = arr is expr.arr and index is expr.index and value is expr.value
        return fn(expr if unchanged else ArrayInject(arr, index, value))
    if isinstance(expr, (ArrayZeros, ArrayConst)):
        return fn(expr)
    if isinstance(expr, MemRead):
        addr = map_expr(expr.addr, fn)
        return fn(expr if addr is expr.addr else MemRead(expr.mem, addr))
    raise TypeError(type(expr))


def replace_id(expr: Expr, name: str, repl: Expr) -> Expr:
    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name == name:
            return repl
        return node

    return map_expr(expr, fn)


def map_expr_memoized(
    expr: Expr, fn, memo: dict[int, tuple[Expr, Expr]] | None = None
) -> Expr:
    """Map an immutable DAG, invoking the visitor once per distinct input node.

    A supplied memo belongs to one unchanging visitor context. Holding the
    original nodes prevents recycled object IDs from aliasing later roots.
    Each result must remain valid for every use of its input in that context.
    `map_expr` retains its per-occurrence visitor semantics.
    """
    if memo is None:
        memo = {}
    active: set[int] = set()
    stack = [(expr, False)]
    while stack:
        node, ready = stack.pop()
        key = id(node)
        if key in memo:
            if memo[key][0] is not node:
                raise ValueError("expression memo contains a foreign identity")
            continue
        children = _expr_children(node)
        if not ready:
            if key in active:
                raise ValueError("cyclic expression graph")
            active.add(key)
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children))
            continue
        active.remove(key)
        mapped = tuple(memo[id(child)][1] for child in children)
        rebuilt = node
        if any(a is not b for a, b in zip(children, mapped)):
            if isinstance(node, UnaryOp):
                rebuilt = UnaryOp(node.op, mapped[0])
            elif isinstance(node, BinOp):
                rebuilt = BinOp(node.op, *mapped)
            elif isinstance(node, Ternary):
                rebuilt = Ternary(*mapped)
            elif isinstance(node, Extract):
                rebuilt = Extract(mapped[0], node.low, node.width)
            elif isinstance(node, Concat):
                rebuilt = Concat(mapped)
            elif isinstance(node, ArrayGet):
                rebuilt = ArrayGet(*mapped)
            elif isinstance(node, ArrayInject):
                rebuilt = ArrayInject(*mapped)
            elif isinstance(node, MemRead):
                rebuilt = MemRead(node.mem, mapped[0])
        memo[key] = (node, fn(rebuilt))
    return memo[id(expr)][1]


CMP_OPS = {"==", "!=", "<", "<=", ">", ">=", "s<", "s<=", "s>", "s>="}


def expr_width(expr: Expr, sigs: Mapping[str, Signal]) -> int:
    widths: dict[int, int] = {}
    stack = [(expr, False)]
    while stack:
        node, ready = stack.pop()
        key = id(node)
        if key in widths:
            continue
        if isinstance(node, (Const, Extract, ArrayZeros, ArrayConst)):
            widths[key] = node.width
            continue
        if isinstance(node, Id):
            widths[key] = sigs[node.name].width
            continue
        if isinstance(node, MemRead):
            widths[key] = sigs[node.mem].width
            continue
        if isinstance(node, BinOp) and node.op in CMP_OPS:
            widths[key] = 1
            continue
        if isinstance(node, Concat):
            children = node.parts
        elif isinstance(node, BinOp):
            children = (node.a,) if node.op in {"<<", ">>"} else (node.a, node.b)
        elif isinstance(node, (UnaryOp, Ternary)):
            children = (node.a,)
        elif isinstance(node, (ArrayGet, ArrayInject)):
            children = (node.arr,)
        else:
            raise TypeError(type(node))
        if not ready:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children))
        elif isinstance(node, Concat):
            widths[key] = sum(widths[id(child)] for child in children)
        else:
            widths[key] = max(widths[id(child)] for child in children)
    return widths[id(expr)]


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
