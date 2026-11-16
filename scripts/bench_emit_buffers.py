#!/usr/bin/env python3
"""Measure text-emission allocations in one fresh process; emit no model files.

Run separately against each source checkout. This synthetic register bank
measures Python allocations, not full GPU memory or simulator throughput.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flashsim', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registers', type=int, default=150000)
    args = parser.parse_args()
    if args.registers < 1:
        parser.error('positive --registers required')
    source = args.flashsim.resolve(strict=True)
    sys.path.insert(0, str(source))
    from flashsim import emit
    from flashsim.ir import AlwaysFF, Assign, Id, Module, NbAssign, Signal
    if not Path(emit.__file__).resolve().is_relative_to(source / 'flashsim'):
        parser.error('emitter must come from the selected source checkout')
    signals = {name: Signal(name, width, kind) for name, width, kind in (
        ('clock', 1, 'input'), ('data', 32, 'input'), ('out', 32, 'output'))}
    body = []
    for index in range(args.registers):
        name = 'r' + str(index)
        signals[name] = Signal(name, 32, 'reg')
        body.append(NbAssign(name, Id('data')))
    module = Module('EmissionBuffer', signals, [Assign('out', Id('r0'))],
                    AlwaysFF('clock', body), ['clock', 'data', 'out'])
    tracemalloc.start()
    started = time.monotonic()
    cpp = emit.emit_cpp(module)
    seconds = time.monotonic() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    data = cpp.encode()
    print(json.dumps(dict(registers=args.registers, cpp_bytes=len(data),
        cpp_sha256=hashlib.sha256(data).hexdigest(), seconds=seconds,
        python_current_bytes=current, python_peak_bytes=peak,
        gc_enabled=gc.isenabled())), flush=True)


if __name__ == '__main__':
    main()
