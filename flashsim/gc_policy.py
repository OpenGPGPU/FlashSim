"""Optional cyclic-GC deferral for a single model-generation call."""
from __future__ import annotations

import gc
import os
from functools import wraps
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_T = TypeVar("_T")


def compilation_gc_policy(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        policy = os.environ.get("FLASHSIM_GC_POLICY", "normal")
        if policy not in {"normal", "defer"}:
            raise ValueError("FLASHSIM_GC_POLICY must be normal or defer")
        if policy == "normal":
            return function(*args, **kwargs)
        # Reference counting remains active. Only automatic cyclic collection
        # is deferred; do not force a collection or change caller thresholds.
        enabled = gc.isenabled()
        gc.disable()
        try:
            return function(*args, **kwargs)
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()

    return wrapped
