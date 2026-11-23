#!/usr/bin/env python3
"""Measure optimization of one real CIRCT module, including bit-slice narrowing.

Uses an existing hw.mlir and writes no model or result files. Run each source
checkout in a fresh, externally bounded process with the same Python and seed.
This does not measure complete-host generation, native builds or simulation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import re
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mlir', type=Path)
    parser.add_argument('--module', required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--hierarchy', action='store_true',
                        help='include the selected module\'s original child instances')
    parser.add_argument('--preflatten-wires', action='store_true',
                        help='apply compiler wire compaction before flattening')
    parser.add_argument('--emit-cpp', action='store_true',
                        help='also measure emission and hash the resulting C++')
    parser.add_argument('--flashsim', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('positive --repeat required')
    source = args.flashsim.resolve(strict=True)
    sys.path.insert(0, str(source))
    from flashsim import circt_frontend as frontend, emit, opt
    from flashsim.preflatten import compact_module_wires
    for module in (frontend, opt, emit):
        if not Path(module.__file__).resolve().is_relative_to(source / 'flashsim'):
            parser.error('compiler must come from the selected checkout')

    enabled = gc.isenabled()
    original = opt._narrow_bit_slices
    narrowed = []

    def measure(*values):
        started = time.perf_counter()
        result = original(*values)
        narrowed.append(time.perf_counter() - started)
        return result

    gc.disable()
    opt._narrow_bit_slices = measure
    try:
        text = re.sub(r'\{sv\.[^}]*\}', '', args.mlir.read_text())
        modules = {module.name: module for module in frontend._split_modules(text)}
        del text
        if args.module not in modules:
            parser.error('module not found: ' + args.module)
        library = {}
        pending = [args.module]
        while pending:
            name = pending.pop()
            if name in library:
                continue
            lowered = frontend._lower_module(modules[name])
            if args.preflatten_wires:
                lowered = compact_module_wires(lowered)
            library[name] = lowered
            if args.hierarchy:
                pending.extend(child for _, _, child, _ in library[name].instances)
        del modules
        leaf = library[args.module]
        if args.hierarchy:
            remaining, reachable = frontend._reachable_instances(leaf, library)
            library = {name: library[name] for name in reachable}
            leaf = frontend._flatten(leaf, library, remaining)
        del library
        if getattr(leaf, 'instances', []):
            parser.error('select a leaf module or pass --hierarchy')
        times = []
        for repeat in range(args.repeat):
            started = time.perf_counter()
            result = opt.optimize(leaf)
            times.append(time.perf_counter() - started)
            if repeat + 1 != args.repeat:
                del result
        if len(narrowed) != args.repeat:
            parser.error('expected one enabled narrowing pass per optimization')
        digest = hashlib.sha256(pickle.dumps(result, protocol=5)).hexdigest()
        summary = dict(module=leaf.name, input_signals=len(leaf.signals),
            signals=len(result.signals), assignments=len(result.assigns),
            assertions=len(result.assertions), repeats=args.repeat,
            median_optimize_seconds=statistics.median(times),
            median_narrow_seconds=statistics.median(narrowed),
            representation_sha256=digest, automatic_gc_during_benchmark=False)
        if args.emit_cpp:
            started = time.perf_counter()
            cpp = emit.emit_cpp(result).encode()
            summary.update(emit_seconds=time.perf_counter() - started,
                           emitted_cpp_bytes=len(cpp),
                           emitted_cpp_sha256=hashlib.sha256(cpp).hexdigest())
        print(json.dumps(summary), flush=True)
    finally:
        opt._narrow_bit_slices = original
        if enabled:
            gc.enable()


if __name__ == '__main__':
    main()
