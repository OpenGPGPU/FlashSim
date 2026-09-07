from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Id(Expr):
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: int
    width: int = 32


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    a: Expr


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    a: Expr
    b: Expr


@dataclass(frozen=True)
class Ternary(Expr):
    cond: Expr
    a: Expr
    b: Expr


@dataclass(frozen=True)
class Extract(Expr):
    a: Expr
    low: int
    width: int


@dataclass(frozen=True)
class Concat(Expr):
    parts: tuple[Expr, ...]


@dataclass(frozen=True)
class ArrayGet(Expr):
    arr: Expr
    index: Expr


@dataclass(frozen=True)
class ArrayInject(Expr):
    arr: Expr
    index: Expr
    value: Expr


@dataclass(frozen=True)
class ArrayZeros(Expr):
    depth: int
    width: int


@dataclass(frozen=True)
class MemRead(Expr):
    mem: str
    addr: Expr


@dataclass
class NbAssign:
    lhs: str
    rhs: Expr


@dataclass
class If:
    cond: Expr
    then_body: list[Stmt]
    else_body: list[Stmt] = field(default_factory=list)


Stmt = NbAssign | If


@dataclass
class Signal:
    name: str
    width: int
    kind: str  # input, output, reg, wire, mem
    depth: int = 0  # 0 = scalar; otherwise unpacked array length


@dataclass
class Assign:
    lhs: str
    rhs: Expr


@dataclass
class MemWrite:
    mem: str
    addr: Expr
    data: Expr
    enable: Expr


@dataclass
class AlwaysFF:
    clock: str
    body: list[Stmt]


@dataclass
class Module:
    name: str
    signals: dict[str, Signal]
    assigns: list[Assign]
    always: AlwaysFF
    ports: list[str]
    mem_writes: list[MemWrite] = field(default_factory=list)


def signal_map(mod: Module) -> dict[str, Signal]:
    return mod.signals


def expr_ids(expr: Expr) -> set[str]:
    return set(expr_id_counts(expr))


def expr_id_counts(expr: Expr) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(node: Expr) -> None:
        if isinstance(node, Id):
            counts[node.name] = counts.get(node.name, 0) + 1
            return
        if isinstance(node, Const):
            return
        if isinstance(node, UnaryOp):
            add(node.a)
            return
        if isinstance(node, BinOp):
            add(node.a)
            add(node.b)
            return
        if isinstance(node, Ternary):
            add(node.cond)
            add(node.a)
            add(node.b)
            return
        if isinstance(node, Extract):
            add(node.a)
            return
        if isinstance(node, Concat):
            for part in node.parts:
                add(part)
            return
        if isinstance(node, ArrayGet):
            add(node.arr)
            add(node.index)
            return
        if isinstance(node, ArrayInject):
            add(node.arr)
            add(node.index)
            add(node.value)
            return
        if isinstance(node, ArrayZeros):
            return
        if isinstance(node, MemRead):
            counts[node.mem] = counts.get(node.mem, 0) + 1
            add(node.addr)
            return
        raise TypeError(type(node))

    add(expr)
    return counts


def map_expr(expr: Expr, fn) -> Expr:
    if isinstance(expr, Id) or isinstance(expr, Const):
        return fn(expr)
    if isinstance(expr, UnaryOp):
        return fn(UnaryOp(expr.op, map_expr(expr.a, fn)))
    if isinstance(expr, BinOp):
        return fn(BinOp(expr.op, map_expr(expr.a, fn), map_expr(expr.b, fn)))
    if isinstance(expr, Ternary):
        return fn(
            Ternary(map_expr(expr.cond, fn), map_expr(expr.a, fn), map_expr(expr.b, fn))
        )
    if isinstance(expr, Extract):
        return fn(Extract(map_expr(expr.a, fn), expr.low, expr.width))
    if isinstance(expr, Concat):
        return fn(Concat(tuple(map_expr(p, fn) for p in expr.parts)))
    if isinstance(expr, ArrayGet):
        return fn(ArrayGet(map_expr(expr.arr, fn), map_expr(expr.index, fn)))
    if isinstance(expr, ArrayInject):
        return fn(
            ArrayInject(
                map_expr(expr.arr, fn),
                map_expr(expr.index, fn),
                map_expr(expr.value, fn),
            )
        )
    if isinstance(expr, ArrayZeros):
        return fn(expr)
    if isinstance(expr, MemRead):
        return fn(MemRead(expr.mem, map_expr(expr.addr, fn)))
    raise TypeError(type(expr))


def replace_id(expr: Expr, name: str, repl: Expr) -> Expr:
    def fn(node: Expr) -> Expr:
        if isinstance(node, Id) and node.name == name:
            return repl
        return node

    return map_expr(expr, fn)


CMP_OPS = {"==", "!=", "<", "<=", ">", ">=", "s<", "s<=", "s>", "s>="}


def expr_width(expr: Expr, sigs: dict[str, Signal]) -> int:
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
    if isinstance(expr, ArrayZeros):
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
