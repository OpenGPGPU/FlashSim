"""Opt-in, bounded stderr phase observations for future compilations."""
from contextlib import contextmanager
from functools import wraps
import json
import os
import sys
import time


def _event(value):
    try:
        print('[flashsim-phase] ' + json.dumps(value), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # Optional diagnostics must not mask a compiler failure or make a
        # healthy compile fail because its diagnostic stream was closed.
        pass


@contextmanager
def phase(name):
    if os.environ.get('FLASHSIM_PHASE_TIMINGS') != '1':
        yield
        return
    start = time.perf_counter_ns()
    _event({'phase': name, 'event': 'begin'})
    completed = False
    try:
        yield
        completed = True
    finally:
        # Do not inspect arguments, IR, exceptions or caller frames. No
        # observations are retained once this scalar-only scope returns.
        _event({
            'phase': name, 'event': 'end',
            'seconds': (time.perf_counter_ns() - start) / 1e9,
            'status': 'ok' if completed else 'error',
        })


def timed_phase(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate
