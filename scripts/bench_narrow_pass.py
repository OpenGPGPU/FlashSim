#!/usr/bin/env python3
"""Measure optimization of one real CIRCT leaf, including bit-slice narrowing.

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
    from flashsim import circt_frontend as frontend, opt
    for module in (frontend, opt):
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
        modules = frontend._split_modules(args.mlir.read_text())
        raw = next((module for module in modules if module.name == args.module), None)
        if raw is None:
            parser.error('module not found: ' + args.module)
        del modules
        leaf = frontend._lower_module(raw)
        del raw
        if leaf.instances:
            parser.error('--module must select a leaf without instances')
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
        print(json.dumps(dict(module=leaf.name, input_signals=len(leaf.signals),
            signals=len(result.signals), assignments=len(result.assigns),
            assertions=len(result.assertions), repeats=args.repeat,
            median_optimize_seconds=statistics.median(times),
            median_narrow_seconds=statistics.median(narrowed),
            representation_sha256=digest, automatic_gc_during_benchmark=False)), flush=True)
    finally:
        opt._narrow_bit_slices = original
        if enabled:
            gc.enable()


if __name__ == '__main__':
    main()
