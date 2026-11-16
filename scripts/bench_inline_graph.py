#!/usr/bin/env python3
"""Measure single-use graph allocations without writing model or result files.

Run once per source checkout. Independent alias/output pairs isolate inline
bookkeeping; this is not a complete GPU generation or simulation benchmark.
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
    parser.add_argument('--pairs', type=int, default=100000)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('positive --pairs required')
    source = args.flashsim.resolve(strict=True)
    sys.path.insert(0, str(source))
    from flashsim import opt
    from flashsim.ir import BinOp, Id
    if not Path(opt.__file__).resolve().is_relative_to(source / 'flashsim'):
        parser.error('optimizer must come from the selected source checkout')
    assigns, outputs = {}, set()
    for index in range(args.pairs):
        alias, output = f'a{index}', f'out{index}'
        assigns[alias] = Id(f'i{index}')
        assigns[output] = BinOp('^', Id(alias), Id('mask'))
        outputs.add(output)
    enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    started = time.monotonic()
    try:
        result, body = opt._inline_single_use(assigns, [], outputs)
        seconds = time.monotonic() - started
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        if enabled:
            gc.enable()
    assert body == [] and len(result) == args.pairs
    digest = hashlib.sha256()
    for index, (name, expr) in enumerate(result.items()):
        assert name == f'out{index}'
        assert expr == BinOp('^', Id(f'i{index}'), Id('mask'))
        digest.update((name + ':' + repr(expr) + '\n').encode())
    print(json.dumps(dict(pairs=args.pairs, assignments=len(result),
        result_sha256=digest.hexdigest(), seconds=seconds,
        python_current_bytes=current, python_peak_bytes=peak,
        automatic_gc_during_inline=False)), flush=True)


if __name__ == '__main__':
    main()
