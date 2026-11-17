#!/usr/bin/env python3
"""Measure one actual CIRCT leaf module's hierarchy renaming without outputs.

Uses an existing hw.mlir, retaining all checks and state. This measures one
flatten step, not complete GPU generation, C++ compilation or simulation.
Run each selected checkout in a separate, externally bounded process.
"""
import argparse
from collections import Counter
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
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--flashsim', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('positive --repeat required')
    source = args.flashsim.resolve(strict=True)
    sys.path.insert(0, str(source))
    from flashsim import circt_frontend as frontend
    from flashsim.ir import AlwaysFF, Module, Signal
    if not Path(frontend.__file__).resolve().is_relative_to(source / 'flashsim'):
        parser.error('frontend must come from the selected checkout')

    enabled = gc.isenabled()
    gc.disable()
    try:
        parsed = frontend._split_modules(args.mlir.read_text())
        raw = next((m for m in parsed if m.name == args.module), None)
        if raw is None:
            parser.error('module not found: ' + args.module)
        del parsed
        leaf = frontend._lower_module(raw)
        del raw
        if leaf.instances:
            parser.error('--module must select a leaf without instances')
        inputs = {n: 'in_' + n for n in leaf.ports if leaf.signals[n].kind == 'input'}
        outputs = [n for n in leaf.ports if leaf.signals[n].kind == 'output']
        results = ['out_' + n for n in outputs]
        signals = {}
        for kind, mapping in [('input', inputs.items()), ('output', zip(outputs, results))]:
            for name, mapped in mapping:
                sig = leaf.signals[name]
                signals[mapped] = Signal(mapped, sig.width, kind, sig.depth)
        parent = Module('Wrapper', signals, [], AlwaysFF('clock', []), list(signals))
        parent.instances = [(results, 'child', leaf.name, inputs)]
        times = []
        for repeat in range(args.repeat):
            started = time.perf_counter()
            result = frontend._flatten(parent, {leaf.name: leaf}, Counter({leaf.name: 1}))
            times.append(time.perf_counter() - started)
            if repeat + 1 != args.repeat:
                del result
        # Compare both values and sharing with the same Python and hash seed.
        digest = hashlib.sha256(pickle.dumps(result, protocol=5)).hexdigest()
        print(json.dumps(dict(module=leaf.name, signals=len(result.signals),
            assignments=len(result.assigns), assertions=len(result.assertions),
            repeats=args.repeat, median_seconds=statistics.median(times),
            representation_sha256=digest, automatic_gc_during_benchmark=False)), flush=True)
    finally:
        if enabled:
            gc.enable()


if __name__ == '__main__':
    main()
