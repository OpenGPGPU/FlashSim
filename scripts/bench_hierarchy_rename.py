#!/usr/bin/env python3
"""Measure original hierarchy flattening and compare the emitted C++ digest.

Reads existing CIRCT IR without writing models or results. Run once per source
checkout with the same Python, hash seed and externally bounded resources.
The selected subtree retains its original state, memories and live checks.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mlir', type=Path)
    parser.add_argument('--module', required=True)
    parser.add_argument('--repeat', type=int, default=3)
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
        parsed = {raw.name: raw for raw in frontend._split_modules(args.mlir.read_text())}
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
        times = []
        for repeat in range(args.repeat):
            working = {name: library[name] for name in reachable}
            started = time.perf_counter()
            result = frontend._flatten(top, working, remaining.copy())
            times.append(time.perf_counter() - started)
            del working
            if repeat + 1 != args.repeat:
                del result
        del library, top
        signals, assignments, assertions = (len(result.signals), len(result.assigns), len(result.assertions))
        owned = [result]
        del result
        optimized = opt._optimize_owned(owned)
        cpp = emit.emit_cpp(optimized).encode()
        print(json.dumps(dict(module=args.module, signals=signals, assignments=assignments,
            assertions=assertions, repeats=args.repeat,
            median_flatten_seconds=statistics.median(times),
            emitted_cpp_bytes=len(cpp), emitted_cpp_sha256=hashlib.sha256(cpp).hexdigest(),
            automatic_gc_during_benchmark=False)), flush=True)
    finally:
        if enabled:
            gc.enable()


if __name__ == '__main__':
    main()
