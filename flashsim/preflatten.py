"""Prune private combinational wires before copying a module's instances.

Keep every port, state element, array and observation boundary. In particular,
instance connections are roots: this pass never rewrites a child's bindings or
changes sequential/reset ordering. The ordinary optimizer still runs after
flattening, when cross-module uses are visible.
"""

from flashsim.ir import (
    AlwaysFF, Assign, Module, expr_ids, observation_exprs, stmt_reads, stmt_writes,
)
from flashsim.opt import _dce, _inline_single_use


def compact_module_wires(mod: Module) -> Module:
    instances = getattr(mod, 'instances', [])
    roots = {name for name, signal in mod.signals.items()
             if signal.kind != 'wire' or signal.depth}
    roots.add(mod.always.clock)
    for results, _, _, inputs in instances:
        roots.update(results)
        roots.update(inputs.values())
    for observation in mod.assertions:
        for expr in observation_exprs(observation):
            roots.update(expr_ids(expr))
    for process in mod.clocked_processes:
        roots.update(expr_ids(process.clock))

    # The inliner leaves statement and memory-write expressions named. Include
    # both sequential sections in its use counts without changing either list.
    body = mod.always.body + mod.reset_epilogue
    assigns, _ = _dce({a.lhs: a.rhs for a in mod.assigns}, body, roots, mod.mem_writes)
    assigns, _ = _inline_single_use(assigns, body, roots, mod.mem_writes)

    referred = roots | set(assigns)
    for expr in assigns.values():
        referred.update(expr_ids(expr))
    for stmt in body:
        referred.update(stmt_reads(stmt))
        referred.update(stmt_writes(stmt))
    for write in mod.mem_writes:
        referred.add(write.mem)
        for expr in (write.addr, write.data, write.enable):
            referred.update(expr_ids(expr))
    result = Module(mod.name,
        {name: signal for name, signal in mod.signals.items() if name in referred},
        [Assign(name, expr) for name, expr in assigns.items()],
        AlwaysFF(mod.always.clock, mod.always.body), mod.ports,
        mod.mem_writes, mod.assertions, mod.clocked_processes, mod.reset_epilogue)
    result.instances = instances
    return result
