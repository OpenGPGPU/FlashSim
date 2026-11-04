"""Optional AND reassociation must leave deep opaque expressions intact."""

import sys

import pytest

from flashsim.ir import BinOp, Extract, Id, Signal, UnaryOp
from flashsim.opt import _reassoc_and_chains


def test_deep_literal_is_left_unchanged_when_its_hash_exceeds_recursion_limit():
    nested = Id("source")
    for _ in range(600):
        nested = UnaryOp("~", nested)
    opaque = Extract(nested, 0, 1)
    root = BinOp("&", BinOp("&", Id("a"), Id("b")), opaque)
    signals = {name: Signal(name, 1, "input") for name in ("a", "b", "source")}
    signals["result"] = Signal("result", 1, "output")
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        with pytest.raises(RecursionError):
            hash(opaque)
        result = _reassoc_and_chains({"result": root}, signals)
    finally:
        sys.setrecursionlimit(old_limit)
    assert result["result"] is root
