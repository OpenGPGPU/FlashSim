from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

from flashsim.ir import (
    AlwaysFF,
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    Assign,
    BinOp,
    ClockedAssertion,
    ClockedProcess,
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

_PORT = re.compile(
    r"(in|out)\s+%?([A-Za-z_][A-Za-z0-9_.:]*)\s*:\s*"
    r"(!llhd\.ref<i\d+>|i\d+)"
)
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
    r"%(\S+)\s*=\s*seq\.firreg\s+%(\S+)\s+clock\s+%(\S+)\s*"
    r"(?:\{[^}]*\}\s*)?:\s*(.+)$"
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
_ARRAY_CREATE = re.compile(
    r"%(\S+)\s*=\s*hw\.array_create\s+(.+?)\s*:\s*i(\d+)\s*$"
)
_ARRAY_INJECT = re.compile(
    r"%(\S+)\s*=\s*hw\.array_inject\s+%(\S+)\[%(\S+)\],\s*%(\S+)\s*:\s*!hw\.array<(\d+)xi(\d+)>"
)
_AGG = re.compile(
    r"%(\S+)\s*=\s*hw\.aggregate_constant\s+\[.*\]\s*:\s*!hw\.array<(\d+)xi(\d+)>\s*$"
)
_INSTANCE = re.compile(
    r"(?:((?:%\S+\s*,\s*)*%\S+)\s*=\s*)?hw\.instance\s+\"([^\"]+)\"\s+@"
    r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*->\s*\((.*)\)"
)
# A module with no results still ends in `hw.output`. Operand-less is legal.
_OUTPUT = re.compile(r"hw\.output(?:\s+(.+))?$")
_TO_CLOCK = re.compile(r"%(\S+)\s*=\s*seq\.to_clock\s+%(\S+)")
_SSA = re.compile(r"%([A-Za-z0-9_.:#-]+)")
_ARRAY_TY = re.compile(r"!hw\.array<(\d+)xi(\d+)>")
_INT_TY = re.compile(r"i(\d+)$")
_REF_TY = re.compile(r"!llhd\.ref<(.*)>$")
_TIME = re.compile(r"%(\S+)\s*=\s*llhd\.constant_time\s+<([^>]+)>$")
_SIG = re.compile(r'%(\S+)\s*=\s*llhd\.sig(?:\s+name\s+"[^"]*")?\s+%(\S+)\s*:\s*(.*)$')
_PROBE = re.compile(r"%(\S+)\s*=\s*llhd\.prb\s+%(\S+)\s*:\s*(.*)$")
_DRIVE = re.compile(r"llhd\.drv\s+%(\S+),\s*%(\S+)\s+after\s+%(\S+)"
                    r"(?:\s+if\s+%(\S+))?\s*:\s*(.*)$")
_FORMAT = re.compile(r'%(\S+)\s*=\s*sim\.fmt\.literal\s+("(?:\\.|[^"\\])*")$')

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
    if m := _REF_TY.fullmatch(text):
        text = m.group(1)
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


def _brace_delta(line: str) -> int:
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', "", line)
    return without_strings.count("{") - without_strings.count("}")


def _split_modules(src: str) -> list[_ParsedMod]:
    mods: list[_ParsedMod] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*hw\.module\b", line):
            header = line.strip()
            private = "private" in header.split("@", 1)[0]
            m = _MODULE.search(header)
            if not m:
                raise CirctError(f"bad hw.module header: {header}")
            depth = _brace_delta(line)
            body_lines: list[str] = []
            i += 1
            while i < len(lines) and depth > 0:
                cur = lines[i]
                depth += _brace_delta(cur)
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
    remaining, reachable = _reachable_instances(chosen, by_name)
    # Unreachable modules cannot contribute to this top. Releasing their IR and
    # completed children reduces the peak cost of flattening large hierarchies.
    lib = {name: by_name[name] for name in reachable}
    del by_name
    return _flatten(chosen, lib, remaining)


def _lower_module(raw: _ParsedMod) -> Module:
    signals: dict[str, Signal] = {}
    ports: list[str] = []
    for direction, port, type_src in _PORT.findall(raw.ports_src):
        port = ssa_name(port)
        width, depth = _parse_type(type_src)
        kind = "input" if direction == "in" else "output"
        if port in signals:
            raise CirctError("colliding normalized port name " + port)
        signals[port] = Signal(port, width, kind, depth)
        ports.append(port)

    # A result's SSA name can equal the name of an output port (for example a
    # registered SRAM read). Keep the output port and internal value distinct.
    body_text = raw.body
    definitions = set(re.findall(r"(?m)^\s*%([A-Za-z0-9_.:-]+)\s*=", body_text))
    used = set(signals) | {ssa_name(n) for n in _SSA.findall(body_text)}
    rename: dict[str, str] = {}
    for name in sorted(definitions):
        ident = ssa_name(name)
        if ident not in signals or signals[ident].kind != "output":
            continue
        candidate = ident + "__flashsim_value"
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        rename[name] = candidate
    if rename:
        body_text = _SSA.sub(
            lambda match: "%" + rename.get(match.group(1), match.group(1)), body_text
        )

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
    saw_output = False
    processes = []
    combinational = []
    ordinary_lines = []
    lines = iter(body_text.splitlines())
    for raw_line in lines:
        is_process = bool(re.search(r"(?:^|=\s*)llhd\.process\b", raw_line.strip()))
        is_comb = bool(re.search(r"(?:^|=\s*)llhd\.combinational\b", raw_line.strip()))
        if not (is_process or is_comb):
            ordinary_lines.append(raw_line)
            continue
        process = [raw_line.strip()]
        depth = _brace_delta(raw_line)
        if depth != 1:
            raise CirctError("unsupported LLHD process header: " + raw_line)
        for process_line in lines:
            process.append(process_line.strip())
            depth += _brace_delta(process_line)
            if depth == 0:
                break
        if depth:
            raise CirctError("unterminated LLHD process")
        (processes if is_process else combinational).append(process)
    formats = {}
    times = {}
    references = set()
    driven = set()
    initializers = {}
    nba_drives = []

    for raw_line in ordinary_lines:
        line = raw_line.strip().rstrip(",")
        if not line or line.startswith("//") or line in {"}", "{"}:
            continue
        if _TO_CLOCK.match(line):
            continue
        if m := _FORMAT.fullmatch(line):
            from flashsim.circt_assertions import decode_mlir_string
            formats[ssa_name(m.group(1))] = decode_mlir_string(m.group(2))
            continue
        if m := _TIME.fullmatch(line):
            times[ssa_name(m.group(1))] = m.group(2)
            continue
        if m := _SIG.fullmatch(line):
            ident = ssa_name(m.group(1))
            width, depth = _parse_type(m.group(3))
            signals[ident] = Signal(ident, width, "wire", depth)
            references.add(ident)
            initializers[ident] = ssa_name(m.group(2))
            continue
        if m := _PROBE.fullmatch(line):
            ident = ssa_name(m.group(1))
            width, depth = _parse_type(m.group(3))
            signals[ident] = Signal(ident, width, "wire", depth)
            assigns.append(Assign(ident, _ref(m.group(2))))
            continue
        if m := _DRIVE.fullmatch(line):
            ident = ssa_name(m.group(1))
            if ident in driven:
                raise CirctError("multiple reference drivers: " + ident)
            driven.add(ident)
            delay = times.get(ssa_name(m.group(3)), "").replace(" ", "")
            if delay == "0ns,0d,1e" and not m.group(4):
                assigns.append(Assign(ident, _ref(m.group(2))))
            elif delay == "0ns,1d,0e":
                nba_drives.append((ident,m.group(2),m.group(4)))
            else:
                raise CirctError("unsupported procedural reference drive: " + line)
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
        if m := _ARRAY_CREATE.match(line):
            ident = ssa_name(m.group(1))
            values = [_ref(tok) for tok in _SSA.findall(m.group(2))]
            if not values:
                raise CirctError(f"array_create requires an operand: {line}")
            width, depth = int(m.group(3)), len(values)
            signals[ident] = Signal(ident, width, "wire", depth)
            # CIRCT places its last operand at index zero.
            expr: Expr = ArrayZeros(depth, width)
            index_width = max(1, (depth - 1).bit_length())
            for index, value in enumerate(reversed(values)):
                expr = ArrayInject(expr, Const(index, index_width), value)
            assigns.append(Assign(ident, expr))
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
            results = [ssa_name(tok) for tok in _SSA.findall(m.group(1) or "")]
            result_ports = [port.strip() for port in m.group(5).split(",") if port.strip()]
            if len(results) != len(result_ports):
                raise CirctError("instance result type arity mismatch: " + line)
            # Clocked checks can observe a child output before hierarchy is
            # flattened. Retain the declared type now, just as for local SSA
            # values; the child's assignments supply its driver in _flatten.
            for ident, port in zip(results, result_ports):
                _, separator, type_src = port.partition(":")
                if not separator:
                    raise CirctError("missing instance result type: " + port)
                width, depth = _parse_type(type_src)
                signals[ident] = Signal(ident, width, "wire", depth)
            inst = m.group(2)
            callee = m.group(3)
            inputs: dict[str, str] = {}
            for pm in re.finditer(
                r"([A-Za-z_][A-Za-z0-9_.:]*)\s*:\s*%(\S+?)\s*:\s*"
                r"(?:!llhd\.ref<i\d+>|i\d+)", m.group(4)
            ):
                inputs[ssa_name(pm.group(1))] = ssa_name(pm.group(2))
            instances.append((results, inst, callee, inputs))
            continue
        if m := _OUTPUT.match(line):
            saw_output = True
            payload = m.group(1)
            output_srcs = [ssa_name(tok) for tok in _SSA.findall(payload)] if payload else []
            continue
        raise CirctError(f"unsupported CIRCT op: {line}")

    body = [NbAssign(reg, Id(nxt)) for reg, nxt, _, _ in regs]
    out_ports = [p for p in ports if signals[p].kind == "output"]
    if saw_output:
        if len(output_srcs) != len(out_ports):
            raise CirctError(
                f"hw.output arity {len(output_srcs)} != output ports {len(out_ports)}"
            )
        for port, src in zip(out_ports, output_srcs):
            if src != port:
                assigns.append(Assign(port, Id(src)))
    if references - driven:
        raise CirctError("undriven procedural references: " + ", ".join(sorted(references-driven)))
    assertions = []
    clocked_processes = []
    if processes or combinational:
        from flashsim.circt_assertions import (AssertionLoweringError, lower_process,
                                               lower_combinational, process_results)
        result_types = {}
        for process in processes:
            name,types = process_results(process[0])
            if name:
                result_types[ssa_name(name)] = types
        drives_by_process = {}
        constants = {a.lhs:a.rhs for a in assigns if isinstance(a.rhs,Const)}
        def result_operand(token):
            parts = token.rsplit("#",1)
            name = ssa_name(parts[0])
            index = int(parts[1]) if len(parts)==2 else 0
            if name not in result_types or not 0<=index<len(result_types[name]):
                raise CirctError("unowned LLHD process result: %"+token)
            return name,index
        for target,data,enabled in nba_drives:
            name,index = result_operand(data)
            if target not in references or _parse_type(result_types[name][index]) != (
                    signals[target].width,signals[target].depth):
                raise CirctError("LLHD state driver type mismatch: "+target)
            if constants.get(initializers[target]) != Const(0,signals[target].width):
                raise CirctError("nonzero or unknown LLHD state initializer: "+target)
            signals[target].kind = "reg"
            enable_index = None
            if enabled:
                if ssa_name(enabled) in constants:
                    enable_index = constants[ssa_name(enabled)]
                else:
                    enable_name,enable_index = result_operand(enabled)
                    if enable_name!=name or _parse_type(result_types[name][enable_index])!=(1,0):
                        raise CirctError("LLHD write enable requires the same clock process")
            drives_by_process.setdefault(name,[]).append((target,index,enable_index))
        comb_results = []
        for region in combinational:
            name,types = process_results(region[0].replace("llhd.combinational","llhd.process",1))
            names = [ssa_name(name+("#"+str(i) if len(types)>1 else ""))
                     for i in range(len(types))]
            for ident,type_src in zip(names,types):
                width,depth = _parse_type(type_src)
                signals[ident] = Signal(ident,width,"wire",depth)
            comb_results.append((names,types))
        global_values = {n: Id(n) for n in signals}
        global_values.update({a.lhs: a.rhs for a in assigns if isinstance(a.rhs, (Const, Id))})

        def lower_pure(line):
            lowered = _lower_module(_ParsedMod("assertion_expr", False, "", line))
            if (len(lowered.assigns) != 1 or lowered.always.body or lowered.mem_writes
                    or lowered.assertions or lowered.clocked_processes
                    or getattr(lowered, "instances", [])):
                raise CirctError("unsupported assertion expression: " + line)
            assign = lowered.assigns[0]
            sig = lowered.signals[assign.lhs]
            return assign.lhs, assign.rhs, (sig.width,sig.depth)

        def make_materializer(prefix):
            def materialize(label,expr,width,depth):
                if isinstance(expr,Const):
                    return expr
                proposed = prefix+ssa_name(label)
                while proposed in signals:
                    proposed += "_"
                signals[proposed] = Signal(proposed,width,"wire",depth)
                assigns.append(Assign(proposed,expr))
                return Id(proposed)
            return materialize

        for index,region in enumerate(combinational):
            try:
                values = lower_combinational(region,globals=global_values,
                    lower_pure=lower_pure,normalize=ssa_name,dimensions=_parse_type,
                    materialize=make_materializer("_flashsim_comb_"+str(index)+"_"),
                    scope=raw.name+".comb"+str(index))
                names,_ = comb_results[index]
                for ident,value in zip(names,values):
                    assigns.append(Assign(ident,value))
                    global_values[ident] = value if isinstance(value,(Const,Id)) else Id(ident)
            except AssertionLoweringError as error:
                raise CirctError(raw.name+": "+str(error)) from error

        for index, process in enumerate(processes):
            try:
                name,_ = process_results(process[0])
                checks,updates,clocks = lower_process(process, globals=global_values,
                    formats=formats, scope=raw.name+".process"+str(index),
                    lower_pure=lower_pure, normalize=ssa_name,
                    drives=drives_by_process.get(ssa_name(name),[]) if name else [],
                    materialize=make_materializer("_flashsim_process_"+str(index)+"_"),
                    dimensions=_parse_type)
                assertions.extend(checks)
                body.extend(updates)
                clocked_processes.extend(clocks)
            except AssertionLoweringError as error:
                raise CirctError(raw.name + ": " + str(error)) from error
    elif nba_drives:
        raise CirctError("nonblocking reference driver has no clock process")
    mod = Module(raw.name, signals, assigns, AlwaysFF(clock, body), ports, mem_writes,
                 assertions,clocked_processes)
    mod.instances = instances  # type: ignore[attr-defined]
    return mod


def _reachable_instances(
    mod: Module, lib: dict[str, Module]
) -> tuple[Counter[str], set[str]]:
    remaining: Counter[str] = Counter()
    reachable: set[str] = set()
    pending = [mod.name]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        if name not in lib:
            raise CirctError(f"missing callee @{name}")
        reachable.add(name)
        for _, _, callee_name, _ in getattr(lib[name], "instances", []):
            remaining[callee_name] += 1
            pending.append(callee_name)
    return remaining, reachable


def _flatten(mod: Module, lib: dict[str, Module], remaining: Counter[str]) -> Module:
    instances = getattr(mod, "instances", [])
    if not instances:
        return mod
    assigns = list(mod.assigns)
    signals = dict(mod.signals)
    body = list(mod.always.body)
    mem_writes = list(mod.mem_writes)
    assertions = list(mod.assertions)
    clocked_processes = list(mod.clocked_processes)
    for results, inst, callee_name, inputs in instances:
        if callee_name not in lib:
            raise CirctError(f"missing callee @{callee_name}")
        callee = _flatten(lib[callee_name], lib, remaining)
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
        reserved = set(signals) | set(out_map.values()) | set(inputs.values())
        for n, sig in callee.signals.items():
            if n in inputs:
                mapped = inputs[n]
            elif n in out_map:
                mapped = out_map[n]
            else:
                mapped = prefix + n
                # A parent result such as left.q can normalize to the same
                # name as a prefixed private child wire. They are different
                # values, including when a bind makes the output a reference.
                while mapped in reserved:
                    mapped += "__flashsim_local"
                reserved.add(mapped)
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
        for assertion in callee.assertions:
            assertions.append(ClockedAssertion(
                _rename_expr(assertion.clock, rename),
                _rename_expr(assertion.failure, rename),
                assertion.message, inst+"."+assertion.scope))
        for process in callee.clocked_processes:
            clocked_processes.append(ClockedProcess(
                _rename_expr(process.clock,rename),inst+"."+process.scope))
        remaining[callee_name] -= 1
        if remaining[callee_name] == 0:
            del lib[callee_name]
    return Module(
        mod.name, signals, assigns, AlwaysFF(mod.always.clock, body), mod.ports,
        mem_writes, assertions, clocked_processes
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


def _is_assertion_only(path: Path) -> bool:
    """Assertion-binding modules, not design logic.

    The gpu build drops a `<Module>_Verification_Assert.sv` beside each
    elaborated module. They only contain `bind`-style hierarchical references
    (`SomeModule.io_...`), so circt-verilog cannot resolve them without the
    whole design and Verilator's build never passes them either. Some output
    trees include them (63 in arti-work/.../generated/rtl) and some do not, so
    filter rather than rely on the caller.
    """
    return path.name.endswith("_Verification_Assert.sv")


def verilog_sources(path: Path, *, assertions: bool = False) -> list[Path]:
    """Synthesizable SystemVerilog for `path`.

    Prefer a sibling `filelist.f` (the Chisel emit list). Otherwise take
    `*.sv` next to the top. Skip `verification/` layers and assertion-only
    modules in both cases unless bound assertions are explicitly requested.
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
            normalized = line.replace("\\", "/")
            if "verification/" in normalized and not assertions:
                continue
            candidate = (path.parent / line).resolve()
            if not candidate.is_file():
                if assertions:
                    raise CirctError("missing requested simulation source: " + str(candidate))
                continue
            if candidate in seen:
                continue
            if _is_assertion_only(candidate) and not assertions:
                continue
            seen.add(candidate)
            files.append(candidate)
        if path not in seen and path.is_file():
            files.append(path)
        if files:
            if assertions:
                # Chisel's emit list names the check modules but omits the
                # generated bind-layer files. Include those layers as well;
                # otherwise hierarchical observers are unbound top modules.
                seen = set(files)
                files.extend(p.resolve() for p in sorted(
                    (path.parent / "verification/assert").glob("layers-*.sv"))
                    if p.resolve() not in seen)
            return files
    files = sorted(
        p
        for p in path.parent.glob("*.sv")
        if p.is_file() and p.name != "filelist.f"
        and (assertions or not _is_assertion_only(p))
    )
    if assertions:
        files.extend(sorted((path.parent / "verification/assert").glob("*.sv")))
    return files if files else [path]


def circt_verilog_command(path: Path, *, assertions: bool = False) -> list[str]:
    tool = find_bin("circt-verilog")
    if tool is None:
        raise CirctError("circt-verilog not found; run python3 -m flashsim setup-circt")
    files = verilog_sources(path, assertions=assertions)
    includes = verilog_include_dirs(path, files) if assertions else []
    srcs = [str(p) for p in files]
    # Bind layers recursively include one another. Their generated guards
    # must share a compilation unit rather than instantiate checks twice.
    return [str(tool), *(["--single-unit"] if assertions else []),
            *("-I"+directory for directory in includes), *srcs, "-o", "-"]


def circt_verilog_to_mlir(path: Path, *, assertions: bool = False) -> str:
    proc = subprocess.run(
        circt_verilog_command(path, assertions=assertions),
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise CirctError(proc.stderr or proc.stdout or "circt-verilog failed")
    return proc.stdout


def verilog_include_dirs(path: Path, files: list[Path]) -> list[str]:
    directories = {p.parent for p in files}
    base = path.resolve().parent
    for source in files:
        if source.is_relative_to(base):
            directories.update(p for p in source.parents if p.is_relative_to(base))
    return sorted(map(str, directories))


def verilog_dependencies(path: Path) -> list[Path]:
    """Conservatively include literal headers in assertion-cache provenance."""
    sources = verilog_sources(path, assertions=True)
    directories = list(map(Path, verilog_include_dirs(path, sources)))
    pending = list(sources)
    seen = set()
    while pending:
        source = pending.pop().resolve()
        if source in seen:
            continue
        seen.add(source)
        pattern = r'(?m)^\s*`include\s+"([^"]+)"'
        for header in re.findall(pattern, source.read_text()):
            for directory in [source.parent, *directories]:
                candidate = (directory / header).resolve()
                if candidate.is_file():
                    pending.append(candidate)
                    break
    return sorted(seen)


def import_verilog_file(path: Path, *, assertions: bool = False) -> Module:
    return parse_hw_mlir(circt_verilog_to_mlir(path, assertions=assertions), top=path.stem)
