#!/usr/bin/env python3
"""Measure original hierarchy flattening and compare the emitted C++ digest.

Reads existing CIRCT IR without writing models or results. Run once per source
checkout with the same Python, hash seed and externally bounded resources.
The selected subtree retains its original state, memories and live checks.
"""
import argparse
from collections import Counter
from functools import cache
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mlir', type=Path)
    parser.add_argument('--module', required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--preflatten-wires', action='store_true')
    parser.add_argument('--frontend-only', action='store_true')
    parser.add_argument('--flashsim', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('positive --repeat required')
    source = args.flashsim.resolve(strict=True)
    sys.path.insert(0, str(source))
    from flashsim import circt_frontend as frontend, emit, opt
    for module in (frontend, emit, opt):
        if not Path(module.__file__).resolve().is_relative_to(source / 'flashsim'):
            parser.error('compiler must come from the selected checkout')

    enabled = gc.isenabled()
    gc.disable()
    try:
        text = re.sub(r'\{sv\.[^}]*\}', '', args.mlir.read_text())
        parsed = {raw.name: raw for raw in frontend._split_modules(text)}
        del text
        if args.module not in parsed:
            parser.error('module not found: ' + args.module)
        library = {}
        pending = [args.module]
        while pending:
            name = pending.pop()
            if name in library:
                continue
            library[name] = frontend._lower_module(parsed[name])
            pending.extend(child for _, _, child, _ in library[name].instances)
        del parsed
        top = library[args.module]
        remaining, reachable = frontend._reachable_instances(top, library)
        @cache
        def state_shape(name):
            mod = library[name]
            shape = Counter((signal.kind, signal.width, signal.depth)
                for signal in mod.signals.values() if signal.kind in {'reg', 'mem'})
            shape['assertions'] = len(mod.assertions)
            shape['processes'] = len(mod.clocked_processes)
            shape['memory_writes'] = len(mod.mem_writes)
            for _, _, child, _ in mod.instances:
                shape.update(state_shape(child))
            return shape
        expected_shape = state_shape(args.module)
        state_shape.cache_clear()
        del state_shape
        del top
        times = []
        compact_times = []
        for repeat in range(args.repeat):
            # On the final pass, transfer the library just as the production
            # frontend does. Do not pin an extra raw hierarchy during expansion.
            working = {name: library[name] for name in reachable} if repeat + 1 < args.repeat else library
            if repeat + 1 == args.repeat:
                del library
            started = time.perf_counter()
            if args.preflatten_wires:
                from flashsim.preflatten import compact_module_wires
                working = {name: compact_module_wires(mod) for name, mod in working.items()}
            compact_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            result = frontend._flatten(working[args.module], working, remaining.copy())
            times.append(time.perf_counter() - started)
            del working
            if repeat + 1 != args.repeat:
                del result
        signals, assignments, assertions = (len(result.signals), len(result.assigns), len(result.assertions))
        shape = Counter((signal.kind, signal.width, signal.depth)
            for signal in result.signals.values() if signal.kind in {'reg', 'mem'})
        shape.update(assertions=len(result.assertions), processes=len(result.clocked_processes),
                     memory_writes=len(result.mem_writes))
        if shape != expected_shape:
            raise ValueError('flattened state, memory or observations changed')
        summary = dict(module=args.module, signals=signals, assignments=assignments,
            assertions=assertions, repeats=args.repeat,
            registers=sum(signal.kind == 'reg' for signal in result.signals.values()),
            memory_arrays=sum(signal.kind == 'mem' for signal in result.signals.values()),
            memory_elements=sum(signal.depth for signal in result.signals.values() if signal.kind == 'mem'),
            clocked_processes=len(result.clocked_processes),
            source_state_and_observation_counts_preserved=True,
            preflatten_wires=args.preflatten_wires,
            median_compact_seconds=statistics.median(compact_times),
            median_flatten_seconds=statistics.median(times),
            median_compact_and_flatten_seconds=statistics.median(
                a + b for a, b in zip(compact_times, times)),
            automatic_gc_during_benchmark=False)
        if args.frontend_only:
            print(json.dumps(summary), flush=True)
            return
        owned = [result]
        del result
        optimized = opt._optimize_owned(owned)
        cpp = emit.emit_cpp(optimized).encode()
        summary.update(emitted_cpp_bytes=len(cpp), emitted_cpp_sha256=hashlib.sha256(cpp).hexdigest())
        print(json.dumps(summary), flush=True)
    finally:
        if enabled:
            gc.enable()


if __name__ == '__main__':
    main()
