from __future__ import annotations

import re
from dataclasses import dataclass

from flashsim.ir import (
    AlwaysFF,
    Assign,
    BinOp,
    Const,
    Expr,
    Id,
    If,
    Module,
    NbAssign,
    Signal,
    Stmt,
    Ternary,
    UnaryOp,
)

KEYWORDS = {
    "module",
    "endmodule",
    "input",
    "output",
    "wire",
    "reg",
    "assign",
    "always",
    "begin",
    "end",
    "if",
    "else",
    "posedge",
    "negedge",
}


@dataclass
class Tok:
    kind: str
    value: str
    pos: int


class ParseError(Exception):
    pass


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r\n]+)
    |(?P<comment>//[^\n]*)
    |(?P<block>/\*.*?\*/)
    |(?P<based>\d+'\s*[sS]?[dDhHbB]\s*[0-9a-fA-FxX_]+)
    |(?P<num>\d+)
    |(?P<id>[a-zA-Z_][a-zA-Z0-9_$]*)
    |(?P<eq>==)
    |(?P<ne>!=)
    |(?P<land>&&)
    |(?P<lor>\|\|)
    |(?P<lshift><<)
    |(?P<rshift>>>)
    |(?P<le><=)
    |(?P<ge>>=)
    |(?P<sym>[][(){}:;,?=<>!~&|^+\-*@])
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(src: str) -> list[Tok]:
    tokens: list[Tok] = []
    pos = 0
    for m in _TOKEN_RE.finditer(src):
        if m.start() != pos:
            raise ParseError(f"unexpected character {src[pos]!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind in {"ws", "comment", "block"}:
            continue
        value = re.sub(r"\s+", "", m.group()) if kind == "based" else m.group()
        if kind == "id" and value in KEYWORDS:
            tokens.append(Tok("kw", value, m.start()))
        elif kind == "sym":
            tokens.append(Tok(value, value, m.start()))
        elif kind in {"le", "ge", "eq", "ne", "land", "lor", "lshift", "rshift"}:
            tokens.append(Tok(value, value, m.start()))
        else:
            tokens.append(Tok(kind, value, m.start()))
    if pos != len(src):
        raise ParseError(f"unexpected character {src[pos]!r} at {pos}")
    tokens.append(Tok("eof", "", pos))
    return tokens


def parse_based(text: str) -> Const:
    m = re.match(r"(\d+)'[sS]?([dDhHbB])([0-9a-fA-FxX_]+)$", text.replace("_", ""))
    if not m:
        raise ParseError(f"bad based number {text}")
    width = int(m.group(1))
    base_ch = m.group(2).lower()
    digits = m.group(3).replace("x", "0").replace("X", "0")
    base = {"d": 10, "h": 16, "b": 2}.get(base_ch)
    if base is None:
        raise ParseError(f"unsupported base in {text}")
    return Const(int(digits, base), width)


class Parser:
    def __init__(self, src: str):
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def accept(self, *kinds: str) -> Tok | None:
        tok = self.peek()
        if tok.kind in kinds or tok.value in kinds:
            self.i += 1
            return tok
        return None

    def expect(self, *kinds: str) -> Tok:
        tok = self.accept(*kinds)
        if tok is None:
            got = self.peek()
            raise ParseError(f"expected {kinds} got {got.kind}:{got.value} at {got.pos}")
        return tok

    def parse_module(self) -> Module:
        self.expect("kw")  # module
        name = self.expect("id").value
        self.expect("(")
        signals: dict[str, Signal] = {}
        ports: list[str] = []
        while self.peek().kind != ")":
            kind = self.expect("kw").value
            if kind not in {"input", "output"}:
                raise ParseError(f"expected port direction, got {kind}")
            width = self.parse_optional_width()
            ident = self.expect("id").value
            signals[ident] = Signal(ident, width, kind)
            ports.append(ident)
            self.accept(",")
        self.expect(")")
        self.expect(";")
        assigns: list[Assign] = []
        always: AlwaysFF | None = None
        while self.peek().value != "endmodule":
            tok = self.peek()
            if tok.value in {"wire", "reg"}:
                self.parse_decl(signals)
            elif tok.value == "assign":
                assigns.append(self.parse_assign())
            elif tok.value == "always":
                if always is not None:
                    raise ParseError("only one always_ff block is supported")
                always = self.parse_always()
            else:
                raise ParseError(f"unexpected token {tok.value} at {tok.pos}")
        self.expect("kw")  # endmodule
        if always is None:
            raise ParseError("missing always @(posedge clk) block")
        return Module(name, signals, assigns, always, ports)

    def parse_optional_width(self) -> int:
        if not self.accept("["):
            return 1
        msb = int(self.expect("num").value)
        self.expect(":")
        lsb = int(self.expect("num").value)
        self.expect("]")
        if lsb != 0:
            raise ParseError("only [msb:0] widths are supported")
        return msb - lsb + 1

    def parse_decl(self, signals: dict[str, Signal]) -> None:
        kind = self.expect("kw").value
        width = self.parse_optional_width()
        while True:
            ident = self.expect("id").value
            if ident not in signals:
                signals[ident] = Signal(ident, width, kind)
            else:
                signals[ident].kind = kind if signals[ident].kind in {"input", "output"} else kind
                signals[ident].width = width
            if self.accept(";"):
                return
            self.expect(",")

    def parse_assign(self) -> Assign:
        self.expect("kw")
        lhs = self.expect("id").value
        self.expect("=")
        rhs = self.parse_expr()
        self.expect(";")
        return Assign(lhs, rhs)

    def parse_always(self) -> AlwaysFF:
        self.expect("kw")
        self.expect("@")
        self.expect("(")
        self.expect("kw")  # posedge
        clock = self.expect("id").value
        self.expect(")")
        body = self.parse_stmt_or_block()
        if not isinstance(body, list):
            body = [body]
        return AlwaysFF(clock, body)

    def parse_stmt_or_block(self) -> list[Stmt] | Stmt:
        if self.peek().value == "begin":
            self.expect("kw")
            stmts: list[Stmt] = []
            while self.peek().value != "end":
                stmts.append(self.parse_stmt())
            self.expect("kw")
            return stmts
        return self.parse_stmt()

    def parse_stmt(self) -> Stmt:
        if self.peek().value == "if":
            return self.parse_if()
        lhs = self.expect("id").value
        self.expect("<=")
        rhs = self.parse_expr()
        self.expect(";")
        return NbAssign(lhs, rhs)

    def parse_if(self) -> If:
        self.expect("kw")  # if
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        then = self.parse_stmt_or_block()
        then_body = then if isinstance(then, list) else [then]
        else_body: list[Stmt] = []
        if self.peek().value == "else":
            self.expect("kw")
            if self.peek().value == "if":
                else_body = [self.parse_if()]
            else:
                els = self.parse_stmt_or_block()
                else_body = els if isinstance(els, list) else [els]
        return If(cond, then_body, else_body)

    def parse_expr(self, min_prec: int = 0) -> Expr:
        expr = self.parse_unary()
        while True:
            tok = self.peek()
            prec = BIN_PREC.get(tok.kind)
            if prec is None or prec < min_prec:
                if tok.kind == "?" and min_prec <= 1:
                    self.expect("?")
                    a = self.parse_expr()
                    self.expect(":")
                    b = self.parse_expr()
                    expr = Ternary(expr, a, b)
                    continue
                break
            op = tok.kind if tok.kind in BIN_PREC else tok.value
            self.i += 1
            rhs = self.parse_expr(prec + 1)
            expr = BinOp(op, expr, rhs)
        return expr

    def parse_unary(self) -> Expr:
        if self.accept("~"):
            return UnaryOp("~", self.parse_unary())
        if self.accept("!"):
            return UnaryOp("!", self.parse_unary())
        if self.accept("-"):
            return UnaryOp("-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        if tok := self.accept("id"):
            return Id(tok.value)
        if tok := self.accept("num"):
            return Const(int(tok.value), 32)
        if tok := self.accept("based"):
            return parse_based(tok.value)
        if self.accept("("):
            expr = self.parse_expr()
            self.expect(")")
            return expr
        tok = self.peek()
        raise ParseError(f"expected expression at {tok.pos}, got {tok.kind}:{tok.value}")


BIN_PREC = {
    "||": 2,
    "&&": 3,
    "|": 4,
    "^": 5,
    "&": 6,
    "==": 7,
    "!=": 7,
    "<": 8,
    ">": 8,
    "<=": 8,
    ">=": 8,
    "<<": 9,
    ">>": 9,
    "+": 10,
    "-": 10,
    "*": 11,
}


def parse_verilog(src: str) -> Module:
    return Parser(src).parse_module()
