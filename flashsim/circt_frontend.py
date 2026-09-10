from __future__ import annotations

import re
import subprocess
from pathlib import Path

from flashsim.ir import (
    AlwaysFF,
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    Assign,
    BinOp,
    Concat,
    Const,
    Expr,
    Extract,
    Id,
    MemRead,
    MemWrite,
    Module,
    NbAssign,
    Signal,
    Ternary,
)
from flashsim.toolchain import find_bin

_PORT = re.compile(r"(in|out)\s+%?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*i(\d+)")
_MODULE = re.compile(
    r"hw\.module(?:\s+private)?\s+@([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)"
)
_CONST = re.compile(r"%(\S+)\s*=\s*hw\.constant\s+(\S+?)(?:\s*:\s*i(\d+))?\s*$")
_EXTRACT = re.compile(
    r"%(\S+)\s*=\s*comb\.extract\s+%(\S+)\s+from\s+(\d+)\s*:\s*\(i\d+\)\s*->\s*i(\d+)\s*$"
)
_CONCAT = re.compile(r"%(\S+)\s*=\s*comb\.concat\s+(.+?)\s*:\s*(.+)$")
_BIN = re.compile(
    r"%(\S+)\s*=\s*comb\.(add|xor|and|or|sub|mul|divu|divs|modu|mods|shl|shru|shrs)\s+(.+?)\s*:\s*i(\d+)\s*$"
)
_ICMP = re.compile(
    r"%(\S+)\s*=\s*comb\.icmp\s+([A-Za-z]+)\s+(.+?)\s*:\s*i(\d+)\s*$"
)
_REPLICATE = re.compile(
    r"%(\S+)\s*=\s*comb\.replicate\s+%(\S+)\s*:\s*\(i(\d+)\)\s*->\s*i(\d+)\s*$"
)
_MUX = re.compile(r"%(\S+)\s*=\s*comb\.mux(?:\s+bin)?\s+(.+?)\s*:\s*(.+)$")
_FIRREG = re.compile(
    r"%(\S+)\s*=\s*seq\.firreg\s+%(\S+)\s+clock\s+%(\S+)\s*:\s*(.+)$"
)
_FIRMEM = re.compile(
    r"%(\S+)\s*=\s*seq\.firmem\s+(\d+)\s*,\s*(\d+)\s*,.*<\s*(\d+)\s*x\s*(\d+)"
)
_READ = re.compile(
    r"%(\S+)\s*=\s*seq\.firmem\.read_port\s+%(\S+)\[%(\S+)\],\s*clock\s+%(\S+)"
)
_WRITE = re.compile(
    r"seq\.firmem\.write_port\s+%(\S+)\[%(\S+)\]\s*=\s*%(\S+),\s*clock\s+%(\S+)(?:\s+enable\s+%(\S+))?"
)
_ARRAY_GET = re.compile(
    r"%(\S+)\s*=\s*hw\.array_get\s+%(\S+)\[%(\S+)\]\s*:\s*!hw\.array<(\d+)xi(\d+)>"
)
_ARRAY_INJECT = re.compile(
    r"%(\S+)\s*=\s*hw\.array_inject\s+%(\S+)\[%(\S+)\],\s*%(\S+)\s*:\s*!hw\.array<(\d+)xi(\d+)>"
)
_AGG = re.compile(
    r"%(\S+)\s*=\s*hw\.aggregate_constant\s+\[.*\]\s*:\s*!hw\.array<(\d+)xi(\d+)>\s*$"
)
_INSTANCE = re.compile(
    r"((?:%\S+\s*,\s*)*%\S+)\s*=\s*hw\.instance\s+\"([^\"]+)\"\s+@"
    r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*->\s*\((.*)\)"
)
_OUTPUT = re.compile(r"hw\.output\s+(.+)$")
_TO_CLOCK = re.compile(r"%(\S+)\s*=\s*seq\.to_clock\s+%(\S+)")
_SSA = re.compile(r"%([A-Za-z0-9_.:-]+)")
_ARRAY_TY = re.compile(r"!hw\.array<(\d+)xi(\d+)>")
_INT_TY = re.compile(r"i(\d+)$")

_BIN_OPS = {
    "add": "+",
    "xor": "^",
    "and": "&",
    "or": "|",
    "sub": "-",
    "mul": "*",
    "divu": "/",
    "divs": "s/",
    "modu": "%",
    "mods": "s%",
    "shl": "<<",
    "shru": ">>",
    "shrs": ">>",
}
_ICMP_OPS = {
    "eq": "==",
    "ceq": "==",
    "weq": "==",
    "ne": "!=",
    "cne": "!=",
    "wne": "!=",
    "ult": "<",
    "ule": "<=",
    "ugt": ">",
    "uge": ">=",
    "slt": "s<",
    "sle": "s<=",
    "sgt": "s>",
    "sge": "s>=",
}


class CirctError(Exception):
    pass


def ssa_name(raw: str) -> str:
    raw = raw.lstrip("%")
    ident = re.sub(r"[^0-9A-Za-z_]", "_", raw)
    if ident[0].isdigit():
        ident = "t" + ident
    if ident in {"true", "false"}:
        return f"c_{ident}"
    return ident


def _ref(raw: str) -> Id:
    return Id(ssa_name(raw if raw.startswith("%") else "%" + raw))


def _const_value(text: str, width: int) -> int:
    if text == "true":
        return 1
    if text == "false":
        return 0
    value = int(text, 0)
    return value & ((1 << width) - 1)


def _parse_type(text: str) -> tuple[int, int]:
    text = text.strip()
    if m := _INT_TY.match(text):
        return int(m.group(1)), 0
    if m := _ARRAY_TY.search(text):
        return int(m.group(2)), int(m.group(1))
    raise CirctError(f"unsupported type {text}")


def _replicate_expr(src: Id, in_w: int, out_w: int) -> Expr:
    if in_w < 1 or out_w < 1 or out_w % in_w != 0:
        raise CirctError(f"unsupported replicate {in_w} -> {out_w}")
    if in_w == 1:
        return Ternary(src, Const((1 << out_w) - 1, out_w), Const(0, out_w))
    copies = out_w // in_w
    return Concat(tuple(src for _ in range(copies)))


class _ParsedMod:
    def __init__(self, name: str, private: bool, ports_src: str, body: str):
        self.name = name
        self.private = private
        self.ports_src = ports_src
        self.body = body


def _split_modules(src: str) -> list[_ParsedMod]:
    mods: list[_ParsedMod] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "hw.module" in line and "@" in line:
            header = line.strip()
            private = "private" in header.split("@", 1)[0]
            m = _MODULE.search(header)
            if not m:
                raise CirctError(f"bad hw.module header: {header}")
            depth = line.count("{") - line.count("}")
            body_lines: list[str] = []
            i += 1
            while i < len(lines) and depth > 0:
                cur = lines[i]
                depth += cur.count("{") - cur.count("}")
                if depth > 0:
                    body_lines.append(cur)
                i += 1
            mods.append(_ParsedMod(m.group(1), private, m.group(2), "\n".join(body_lines)))
            continue
        i += 1
    return mods


def parse_hw_mlir(src: str, top: str | None = None) -> Module:
    src = re.sub(r"\{sv\.[^}]*\}", "", src)
    parsed = _split_modules(src)
    if not parsed:
        raise CirctError("no hw.module found in CIRCT output")
    by_name = {m.name: _lower_module(m) for m in parsed}
    if top and top in by_name:
        chosen = by_name[top]
    else:
        public = [m for m in parsed if not m.private]
        chosen = by_name[(public[-1] if public else parsed[-1]).name]
    return _flatten(chosen, by_name)


def _lower_module(raw: _ParsedMod) -> Module:
    signals: dict[str, Signal] = {}
    ports: list[str] = []
    for direction, port, width in _PORT.findall(raw.ports_src):
        kind = "input" if direction == "in" else "output"
        signals[port] = Signal(port, int(width), kind)
        ports.append(port)

    assigns: list[Assign] = []
    regs: list[tuple[str, str, int, int]] = []
    mem_writes: list[MemWrite] = []
    instances: list[tuple[str, str, dict[str, str], str]] = []
    clock = next(
        (
            n
            for n in ("clk", "clock", "io_s_axi_aclk", "aclk")
            if n in signals and signals[n].kind == "input"
        ),
        "clk",
    )
    output_srcs: list[str] = []

    for raw_line in raw.body.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or line.startswith("//") or line in {"}", "{"}:
            continue
        if _TO_CLOCK.match(line):
            continue
        if m := _CONST.match(line):
            width = int(m.group(3) or 1)
            ident = ssa_name(m.group(1))
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, Const(_const_value(m.group(2), width), width)))
            continue
        if m := _EXTRACT.match(line):
            ident = ssa_name(m.group(1))
            width = int(m.group(4))
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, Extract(_ref(m.group(2)), int(m.group(3)), width)))
            continue
        if m := _CONCAT.match(line):
            ident = ssa_name(m.group(1))
            args = [_ref(tok) for tok in _SSA.findall(m.group(2))]
            widths = [int(w) for w in re.findall(r"i(\d+)", m.group(3))]
            width = sum(widths) if widths else 32
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, Concat(tuple(args))))
            continue
        if m := _ICMP.match(line):
            ident = ssa_name(m.group(1))
            pred = m.group(2)
            if pred not in _ICMP_OPS:
                raise CirctError(f"unsupported icmp predicate {pred}: {line}")
            op = _ICMP_OPS[pred]
            args = [_ref(tok) for tok in _SSA.findall(m.group(3))]
            if len(args) != 2:
                raise CirctError(f"icmp expects 2 args: {line}")
            signals[ident] = Signal(ident, 1, "wire")
            assigns.append(Assign(ident, BinOp(op, args[0], args[1])))
            continue
        if m := _REPLICATE.match(line):
            ident = ssa_name(m.group(1))
            src = _ref(m.group(2))
            in_w = int(m.group(3))
            out_w = int(m.group(4))
            signals[ident] = Signal(ident, out_w, "wire")
            assigns.append(Assign(ident, _replicate_expr(src, in_w, out_w)))
            continue
        if m := _MUX.match(line):
            ident = ssa_name(m.group(1))
            args = [_ref(tok) for tok in _SSA.findall(m.group(2))]
            if len(args) != 3:
                raise CirctError(f"mux expects 3 args: {line}")
            width, depth = _parse_type(m.group(3))
            signals[ident] = Signal(ident, width, "wire", depth)
            assigns.append(Assign(ident, Ternary(args[0], args[1], args[2])))
            continue
        if m := _BIN.match(line):
            ident = ssa_name(m.group(1))
            op = _BIN_OPS[m.group(2)]
            args = [_ref(tok) for tok in _SSA.findall(m.group(3))]
            width = int(m.group(4))
            expr: Expr = args[0]
            for rhs in args[1:]:
                expr = BinOp(op, expr, rhs)
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, expr))
            continue
        if m := _FIRREG.match(line):
            ident = ssa_name(m.group(1))
            nxt = ssa_name(m.group(2))
            width, depth = _parse_type(m.group(4))
            signals[ident] = Signal(ident, width, "reg", depth)
            regs.append((ident, nxt, width, depth))
            continue
        if m := _FIRMEM.match(line):
            ident = ssa_name(m.group(1))
            read_lat = int(m.group(2))
            write_lat = int(m.group(3))
            if read_lat != 0 or write_lat != 1:
                raise CirctError(
                    f"unsupported firmem latencies read={read_lat} write={write_lat}"
                )
            depth = int(m.group(4))
            width = int(m.group(5))
            signals[ident] = Signal(ident, width, "mem", depth)
            continue
        if m := _READ.match(line):
            ident = ssa_name(m.group(1))
            mem = ssa_name(m.group(2))
            addr = _ref(m.group(3))
            width = signals[mem].width if mem in signals else 32
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, MemRead(mem, addr)))
            continue
        if m := _WRITE.match(line):
            mem = ssa_name(m.group(1))
            addr = _ref(m.group(2))
            data = _ref(m.group(3))
            enable = _ref(m.group(5)) if m.group(5) else Const(1, 1)
            mem_writes.append(MemWrite(mem, addr, data, enable))
            continue
        if m := _ARRAY_GET.match(line):
            ident = ssa_name(m.group(1))
            width = int(m.group(5))
            signals[ident] = Signal(ident, width, "wire")
            assigns.append(Assign(ident, ArrayGet(_ref(m.group(2)), _ref(m.group(3)))))
            continue
        if m := _ARRAY_INJECT.match(line):
            ident = ssa_name(m.group(1))
            depth = int(m.group(5))
            width = int(m.group(6))
            signals[ident] = Signal(ident, width, "wire", depth)
            assigns.append(
                Assign(
                    ident,
                    ArrayInject(_ref(m.group(2)), _ref(m.group(3)), _ref(m.group(4))),
                )
            )
            continue
        if m := _AGG.match(line):
            ident = ssa_name(m.group(1))
            depth = int(m.group(2))
            width = int(m.group(3))
            signals[ident] = Signal(ident, width, "wire", depth)
            assigns.append(Assign(ident, ArrayZeros(depth, width)))
            continue
        if m := _INSTANCE.match(line):
            results = [ssa_name(tok) for tok in _SSA.findall(m.group(1))]
            inst = m.group(2)
            callee = m.group(3)
            inputs: dict[str, str] = {}
            for pm in re.finditer(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*%(\S+?)\s*:\s*i\d+", m.group(4)
            ):
                inputs[pm.group(1)] = ssa_name(pm.group(2))
            instances.append((results, inst, callee, inputs))
            continue
        if m := _OUTPUT.match(line):
            output_srcs = [ssa_name(tok) for tok in _SSA.findall(m.group(1))]
            continue
        raise CirctError(f"unsupported CIRCT op: {line}")

    body = [NbAssign(reg, Id(nxt)) for reg, nxt, _, _ in regs]
    out_ports = [p for p in ports if signals[p].kind == "output"]
    if output_srcs:
        if len(output_srcs) != len(out_ports):
            raise CirctError(
                f"hw.output arity {len(output_srcs)} != output ports {len(out_ports)}"
            )
        for port, src in zip(out_ports, output_srcs):
            if src != port:
                assigns.append(Assign(port, Id(src)))
    mod = Module(raw.name, signals, assigns, AlwaysFF(clock, body), ports, mem_writes)
    mod.instances = instances  # type: ignore[attr-defined]
    return mod


def _flatten(mod: Module, lib: dict[str, Module]) -> Module:
    instances = getattr(mod, "instances", [])
    if not instances:
        return mod
    assigns = list(mod.assigns)
    signals = dict(mod.signals)
    body = list(mod.always.body)
    mem_writes = list(mod.mem_writes)
    for results, inst, callee_name, inputs in instances:
        if callee_name not in lib:
            raise CirctError(f"missing callee @{callee_name}")
        callee = _flatten(lib[callee_name], lib)
        lib[callee_name] = callee
        prefix = re.sub(r"[^0-9A-Za-z_]", "_", inst) + "_"
        out_ports = [p for p in callee.ports if callee.signals[p].kind == "output"]
        if len(results) != len(out_ports):
            raise CirctError(
                f"instance {inst} @{callee_name}: {len(results)} results, "
                f"{len(out_ports)} outputs"
            )
        out_map = dict(zip(out_ports, results))
        rename: dict[str, str] = {}
        for n, sig in callee.signals.items():
            if n in inputs:
                mapped = inputs[n]
            elif n in out_map:
                mapped = out_map[n]
            else:
                mapped = prefix + n
            rename[n] = mapped
            if n in inputs:
                continue
            if mapped not in signals:
                kind = "wire" if sig.kind in {"input", "output"} else sig.kind
                signals[mapped] = Signal(mapped, sig.width, kind, sig.depth)
        for a in callee.assigns:
            assigns.append(Assign(rename.get(a.lhs, a.lhs), _rename_expr(a.rhs, rename)))
        for stmt in callee.always.body:
            body.append(_rename_stmt(stmt, rename))
        for wr in callee.mem_writes:
            mem_writes.append(
                MemWrite(
                    rename.get(wr.mem, wr.mem),
                    _rename_expr(wr.addr, rename),
                    _rename_expr(wr.data, rename),
                    _rename_expr(wr.enable, rename),
                )
            )
    return Module(
        mod.name, signals, assigns, AlwaysFF(mod.always.clock, body), mod.ports, mem_writes
    )


def _rename_expr(expr: Expr, rename: dict[str, str]) -> Expr:
    from flashsim.ir import Id as IrId
    from flashsim.ir import map_expr

    def fn(node: Expr) -> Expr:
        if isinstance(node, IrId) and node.name in rename:
            return IrId(rename[node.name])
        if isinstance(node, MemRead):
            return MemRead(rename.get(node.mem, node.mem), node.addr)
        return node

    return map_expr(expr, fn)


def _rename_stmt(stmt, rename: dict[str, str]):
    from flashsim.ir import If

    if isinstance(stmt, NbAssign):
        lhs = rename.get(stmt.lhs, stmt.lhs)
        return NbAssign(lhs, _rename_expr(stmt.rhs, rename))
    if isinstance(stmt, If):
        return If(
            _rename_expr(stmt.cond, rename),
            [_rename_stmt(s, rename) for s in stmt.then_body],
            [_rename_stmt(s, rename) for s in stmt.else_body],
        )
    raise TypeError(stmt)


def verilog_sources(path: Path) -> list[Path]:
    """Synthesizable SystemVerilog for `path`.

    Prefer a sibling `filelist.f` (the Chisel emit list). Otherwise take
    `*.sv` next to the top. Skip `verification/` layers in both cases.
    """
    path = path.resolve()
    listed = path.parent / "filelist.f"
    files: list[Path] = []
    if listed.is_file():
        seen: set[Path] = set()
        for raw in listed.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "verification/" in line.replace("\\", "/"):
                continue
            candidate = (path.parent / line).resolve()
            if not candidate.is_file() or candidate in seen:
                continue
            seen.add(candidate)
            files.append(candidate)
        if path not in seen and path.is_file():
            files.append(path)
        if files:
            return files
    files = sorted(
        p for p in path.parent.glob("*.sv") if p.is_file() and p.name != "filelist.f"
    )
    return files if files else [path]


def circt_verilog_to_mlir(path: Path) -> str:
    tool = find_bin("circt-verilog")
    if tool is None:
        raise CirctError("circt-verilog not found; run python3 -m flashsim setup-circt")
    srcs = [str(p) for p in verilog_sources(path)]
    proc = subprocess.run(
        [str(tool), *srcs, "-o", "-"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise CirctError(proc.stderr or proc.stdout or "circt-verilog failed")
    return proc.stdout


def import_verilog_file(path: Path) -> Module:
    return parse_hw_mlir(circt_verilog_to_mlir(path), top=path.stem)
