"""Lower CIRCT's single-event clocked fatal checks without dropping effects.

The model's tick is a rising edge. Prove the LLHD wait/edge detector has that
meaning, then retain every fatal path and nonblocking state update. Acyclic
clock and combinational regions merge SSA values without enumerating their
execution paths. Other events and nonfatal procedural effects are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from flashsim.ir import (BinOp, ClockedAssertion, ClockedProcess, Const, Expr,
                         Id, If, NbAssign, Ternary, map_expr)


class AssertionLoweringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PreviousClock(Expr):
    pass


@dataclass
class _Block:
    arguments: list[str]
    lines: list[str]
    types: list[str] = field(default_factory=list)


_SSA = re.compile(r"%([A-Za-z0-9_.:#-]+)")
_LABEL = re.compile(r"^\^([A-Za-z0-9_.$-]+)(?:\((.*?)\))?:")
_TARGET = re.compile(r"^\^([A-Za-z0-9_.$-]+)(?:\((.*?)\))?$")
_PROBE = re.compile(r"^%(\S+)\s*=\s*llhd\.prb\s+%(\S+)\s*:")


def process_results(header):
    if header.strip() == "llhd.process {":
        return None, []
    match = re.fullmatch(r"%([A-Za-z0-9_.-]+)(?::(\d+))?\s*=\s*llhd\.process"
                         r"\s*->\s*(.*?)\s*\{", header.strip())
    if not match:
        raise AssertionLoweringError("unsupported LLHD process header: " + header)
    types = [value.strip() for value in match.group(3).split(",")]
    if len(types) != int(match.group(2) or 1):
        raise AssertionLoweringError("LLHD result arity mismatch")
    return match.group(1), types


def _target(text: str) -> tuple[str, list[str]]:
    match = _TARGET.fullmatch(text.strip())
    if not match:
        raise AssertionLoweringError("unsupported control-flow target: " + text)
    operands = (match.group(2) or "").split(":", 1)[0]
    return match.group(1), _SSA.findall(operands)


def _split_targets(text: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if char == "," and depth == 0:
            return text[:index], text[index+1:]
    raise AssertionLoweringError("conditional branch needs two targets: " + text)


def decode_mlir_string(text: str) -> str:
    """Decode MLIR's quoted UTF-8 bytes, including its hexadecimal escapes."""
    if not (text.startswith('"') and text.endswith('"')):
        raise AssertionLoweringError("expected a quoted diagnostic")
    data = bytearray()
    index = 1
    while index < len(text)-1:
        char = text[index]
        if char != "\\":
            data.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(text)-1:
            raise AssertionLoweringError("incomplete diagnostic escape")
        if text[index] in {'"', "\\"}:
            data.extend(text[index].encode())
            index += 1
        elif re.fullmatch(r"[0-9A-Fa-f]{2}", text[index:index+2]):
            data.append(int(text[index:index+2], 16))
            index += 2
        else:
            raise AssertionLoweringError("unsupported diagnostic escape")
    return data.decode("utf-8")


def lower_process(lines, *, globals, formats, scope, lower_pure, normalize,
                  drives=(), materialize=None, dimensions=None):
    result_name, result_types = process_results(lines[0])
    blocks = {"entry": _Block([], [])}
    current = blocks["entry"]
    for raw in lines[1:-1]:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if match := _LABEL.match(line):
            name = match.group(1)
            if name in blocks:
                raise AssertionLoweringError("duplicate process block " + name)
            arguments = [normalize(n) for n in re.findall(
                r"%([A-Za-z0-9_.:-]+)\s*:", match.group(2) or "")]
            current = blocks[name] = _Block(arguments, [])
            current.types = [m.group(2).strip() for m in re.finditer(
                r"%([A-Za-z0-9_.-]+)\s*:\s*([^,]+)", match.group(2) or "")]
        else:
            current.lines.append(line)

    def resolve(name, env):
        name = normalize(name)
        if name in env:
            return env[name]
        if name in globals:
            return globals[name]
        raise AssertionLoweringError("unknown assertion operand %" + name)

    def pure(line, env):
        if match := _PROBE.match(line):
            env[normalize(match.group(1))] = resolve(match.group(2), env)
            return
        name, expr, *_ = lower_pure(line)
        env[name] = map_expr(expr, lambda node: (
            resolve(node.name, env) if isinstance(node, Id) else node
        ))

    def enter(target, env):
        name, operands = _target(target)
        if name not in blocks:
            raise AssertionLoweringError("missing process block " + name)
        block = blocks[name]
        if len(operands) != len(block.arguments):
            raise AssertionLoweringError("block argument arity mismatch at " + name)
        values = [resolve(operand, env) for operand in operands]
        updated = dict(env)
        updated.update(zip(block.arguments, values))
        return name, updated

    # Follow the initial prologue to the single event wait.
    name, env, visited = "entry", {}, set()
    while True:
        if name in visited:
            raise AssertionLoweringError("process loops before its event wait")
        visited.add(name)
        block = blocks[name]
        if not block.lines:
            raise AssertionLoweringError("empty process block " + name)
        for line in block.lines[:-1]:
            pure(line, env)
        terminal = block.lines[-1]
        if terminal.startswith("llhd.wait "):
            wait_name = name
            break
        if not terminal.startswith("cf.br "):
            raise AssertionLoweringError("unsupported process prologue: " + terminal)
        name, env = enter(terminal[len("cf.br "):], env)

    match = re.fullmatch(r"llhd\.wait(?:\s+yield\s+\((.*?)\),)?"
                         r"\s+\((.*?)\),\s*(\^.*)", terminal)
    if not match:
        raise AssertionLoweringError("unsupported event wait: " + terminal)
    yield_values = _SSA.findall((match.group(1) or "").split(":", 1)[0])
    if len(yield_values) != len(result_types):
        raise AssertionLoweringError("LLHD clock wait/result arity mismatch")
    watches = _SSA.findall(match.group(2))
    if len(watches) != 1:
        raise AssertionLoweringError("a clocked check must wait on exactly one clock")
    clock = resolve(watches[0], env)
    edge_name, edge_operands = _target(match.group(3))
    if edge_name not in blocks:
        raise AssertionLoweringError("missing edge detector " + edge_name)
    edge = blocks[edge_name]
    if (len(edge.arguments) != 1 or len(edge_operands) != 1
            or resolve(edge_operands[0], env) != clock):
        raise AssertionLoweringError("edge detector must retain the previous clock")
    env[edge.arguments[0]] = _PreviousClock()
    for line in edge.lines[:-1]:
        pure(line, env)
    match = re.fullmatch(r"cf\.cond_br\s+%(\S+),\s*(.*)", edge.lines[-1])
    if not match:
        raise AssertionLoweringError("missing conditional clock edge detector")
    condition = resolve(match.group(1), env)
    yes, no = _split_targets(match.group(2))
    no_name, no_arguments = _target(no)
    if no_name != wait_name or len(no_arguments) != len(blocks[wait_name].arguments):
        raise AssertionLoweringError("non-edge branch must return to the event wait")

    def previous_low(expr):
        return isinstance(expr, BinOp) and expr.op == "^" and (
            (expr.a == _PreviousClock() and expr.b == Const(1, 1))
            or (expr.b == _PreviousClock() and expr.a == Const(1, 1))
        )

    if not (isinstance(condition, BinOp) and condition.op == "&" and (
            (previous_low(condition.a) and condition.b == clock)
            or (previous_low(condition.b) and condition.a == clock))):
        raise AssertionLoweringError("only a proven posedge event is supported")

    # The proved event fixes the previous clock at zero. Do not carry the
    # temporary proof marker into a subsequently used SSA expression.
    env[edge.arguments[0]] = Const(0, 1)
    for line in edge.lines[:-1]:
        pure(line, env)
    first, first_env = enter(yes, env)
    if materialize is None or dimensions is None:
        raise AssertionLoweringError("LLHD clock regions need typed value lowering")
    checks, updates = _lower_value_region(
        blocks, first, first_env, wait_name, yield_values, drives,
        resolve, lower_pure, normalize, materialize, dimensions, formats, clock, scope)
    if not result_types and not checks:
        raise AssertionLoweringError("clocked process contains no fatal checks")
    return checks, updates, [ClockedProcess(clock,scope)] if result_types else []


def _lower_value_region(blocks, first, values, wait_name, yields, drives,
                        resolve, lower_pure, normalize, materialize, dimensions,
                        formats, clock, scope, combinational=False):
    """Merge a clock-region DAG once, rather than enumerate register paths."""
    def successors(name):
        lines = blocks[name].lines
        if any(line.startswith("sim.terminate ") for line in lines):
            return []
        if not lines:
            raise AssertionLoweringError("empty value-process block")
        terminal = lines[-1]
        if terminal.startswith("cf.br "):
            return [_target(terminal[len("cf.br "):])[0]]
        match = re.fullmatch(r"cf\.cond_br\s+%(\S+),\s*(.*)", terminal)
        if not match:
            raise AssertionLoweringError("unsupported value-process terminator: "+terminal)
        return [_target(t)[0] for t in _split_targets(match.group(2))]

    reachable, stack = set(), [first]
    while stack:
        name = stack.pop()
        if name == wait_name or name in reachable:
            continue
        if name not in blocks:
            raise AssertionLoweringError("missing value-process block "+name)
        reachable.add(name)
        stack.extend(successors(name))
    incoming_count = {name:0 for name in reachable}
    for name in reachable:
        for target in successors(name):
            if target in reachable:
                incoming_count[target] += 1
    ready = sorted(name for name,count in incoming_count.items() if count==0)
    order = []
    while ready:
        name = ready.pop()
        order.append(name)
        for target in successors(name):
            if target in reachable:
                incoming_count[target] -= 1
                if incoming_count[target]==0:
                    ready.append(target)
    if len(order)!=len(reachable):
        raise AssertionLoweringError("value-process clock region contains a loop")
    wait_arguments = set(blocks[wait_name].arguments)
    if any(normalize(operand) in wait_arguments for name in reachable
           for line in blocks[name].lines for operand in _SSA.findall(line)):
        raise AssertionLoweringError("persistent process-local wait state is unsupported")

    incoming = {first:[(Const(1,1),[values[arg] for arg in blocks[first].arguments])]}
    updates, assertions, returned_values = [], [], []
    def merge_entries(entries):
        # Large reference selectors can converge thousands of mutually
        # exclusive arms. Keep reach predicates and phi values logarithmic
        # in depth while preserving the last incoming arm's priority.
        while len(entries)>1:
            paired = []
            for index in range(0,len(entries),2):
                if index+1==len(entries):
                    paired.append(entries[index])
                    continue
                left,left_values = entries[index]
                right,right_values = entries[index+1]
                predicate = BinOp("|",left,right)
                values = [a if a==b else Ternary(right,b,a)
                          for a,b in zip(left_values,right_values)]
                paired.append((predicate,values))
            entries = paired
        return entries[0]
    def wire(label,expr,type_src="i1"):
        width,depth = dimensions(type_src)
        return materialize(label,expr,width,depth)
    def send(target_text,predicate):
        name,operands = _target(target_text)
        args = [resolve(operand,values) for operand in operands]
        if len(args)!=len(blocks[name].arguments):
            raise AssertionLoweringError("value-process branch arity mismatch")
        if name!=wait_name:
            incoming.setdefault(name,[]).append((predicate,args))
            return
        returned = dict(values)
        returned.update(zip(blocks[wait_name].arguments,args))
        results = [resolve(operand,returned) for operand in yields]
        if combinational:
            returned_values.append((predicate,results))
            return
        for target,data_index,enable_index in drives:
            enabled = (results[enable_index] if isinstance(enable_index,int)
                       else enable_index if isinstance(enable_index,Expr) else Const(1,1))
            guard = wire(name+"_"+target+"_write",BinOp("&",predicate,enabled))
            updates.append(If(guard,[NbAssign(target,results[data_index])]))

    for name in order:
        block = blocks[name]
        entries = incoming.get(name,[])
        if not entries:
            raise AssertionLoweringError("unreachable value-process block "+name)
        reach,merged_values = merge_entries(entries)
        reach = wire(name+"_reach",reach)
        for index,arg in enumerate(block.arguments):
            values[arg] = wire(name+"_"+arg,merged_values[index],block.types[index])
        prints = []
        for index,line in enumerate(block.lines):
            if line.startswith("cf.br "):
                if index != len(block.lines)-1:
                    raise AssertionLoweringError("instructions follow a branch")
                if prints:
                    raise AssertionLoweringError("nonfatal procedural printing is unsupported")
                send(line[len("cf.br "):],reach)
            elif match := re.fullmatch(r"cf\.cond_br\s+%(\S+),\s*(.*)",line):
                if index != len(block.lines)-1:
                    raise AssertionLoweringError("instructions follow a conditional branch")
                if prints:
                    raise AssertionLoweringError("nonfatal procedural printing is unsupported")
                guard = resolve(match.group(1),values)
                yes,no = _split_targets(match.group(2))
                send(yes,wire(name+"_yes",BinOp("&",reach,guard)))
                send(no,wire(name+"_no",BinOp("&",reach,BinOp("^",guard,Const(1,1)))))
            elif match := _PROBE.match(line):
                values[normalize(match.group(1))] = resolve(match.group(2),values)
            elif match := re.fullmatch(r"sim\.proc\.print\s+%(\S+)",line):
                ident = normalize(match.group(1))
                if ident not in formats:
                    raise AssertionLoweringError("unsupported formatted assertion diagnostic")
                prints.append(formats[ident])
            elif line in {"sim.terminate failure","sim.terminate failure, quiet"}:
                if any(trailing != "ub.unreachable" for trailing in block.lines[index+1:]):
                    raise AssertionLoweringError("unsupported effects after fatal termination")
                assertions.append(ClockedAssertion(clock,reach,
                    "".join(prints).strip() or "fatal clocked check",scope+"."+name))
                break
            else:
                ident,expr,type_info = lower_pure(line)
                expr = map_expr(expr,lambda node:(
                    resolve(node.name,values) if isinstance(node,Id) else node))
                values[ident] = materialize(name+"_"+ident,expr,*type_info)
    if combinational:
        if assertions or not returned_values:
            raise AssertionLoweringError("combinational region needs pure yielded values")
        _,result = merge_entries(returned_values)
        return assertions,updates,result
    return assertions,updates


def lower_combinational(lines, *, globals, lower_pure, normalize,
                        materialize, dimensions, scope):
    _,types = process_results(lines[0].replace("llhd.combinational","llhd.process",1))
    blocks = {"entry":_Block([],[])}
    current = blocks["entry"]
    return_name = "_flashsim_return"
    arguments = ["_flashsim_result_"+str(i) for i in range(len(types))]
    for raw in lines[1:-1]:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if match := _LABEL.match(line):
            name = match.group(1)
            if name==return_name or name in blocks:
                raise AssertionLoweringError("colliding combinational block "+name)
            declarations = list(re.finditer(r"%([A-Za-z0-9_.-]+)\s*:\s*([^,]+)",
                                            match.group(2) or ""))
            current = blocks[name] = _Block(
                [normalize(m.group(1)) for m in declarations],[],
                [m.group(2).strip() for m in declarations])
        elif line.startswith("llhd.yield"):
            operands = _SSA.findall(line.split(":",1)[0])
            if len(operands)!=len(types):
                raise AssertionLoweringError("combinational yield arity mismatch")
            current.lines.append("cf.br ^"+return_name+"("+
                ", ".join("%"+operand for operand in operands)+" : "+", ".join(types)+")")
        else:
            current.lines.append(line)
    blocks[return_name] = _Block(arguments,[],types)
    def resolve(name,values):
        name = normalize(name)
        if name in values:
            return values[name]
        if name in globals:
            return globals[name]
        raise AssertionLoweringError("unknown combinational operand %"+name)
    _,_,results = _lower_value_region(blocks,"entry",{},return_name,arguments,[],
        resolve,lower_pure,normalize,materialize,dimensions,{},None,scope,
        combinational=True)
    return results
