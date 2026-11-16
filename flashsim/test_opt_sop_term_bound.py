"""Enforce the existing SOP term limit before expanding a shared OR cone."""

import pytest

from flashsim import ir, opt


def signals(names):
    return {name: ir.Signal(name, 1, "output" if name == "out" else "input")
            for name in names}


def test_oversized_shared_or_stops_before_expansion():
    reads = 0

    class ObservedBinOp(ir.BinOp):
        def __getattribute__(self, name):
            nonlocal reads
            if name in {"a", "b"}:
                reads += 1
                if reads > 4096:
                    raise AssertionError("SOP expanded a shared OR beyond its existing term limit")
            return super().__getattribute__(name)

    root = ir.Id("p")
    for _ in range(20):
        root = ObservedBinOp("|", root, root)
    assigns = {"out": root}
    result = opt._merge_sop_terms(assigns, signals(["p", "out"]))
    assert result is assigns and result["out"] is root
    assert reads <= 4096


@pytest.mark.parametrize("terms", [256, 257])
def test_existing_sop_cutoff_keeps_its_exact_behavior(terms):
    p = ir.Id("p")
    parts = [p, ir.UnaryOp("!", p),
             *(ir.Id(f"term{i}") for i in range(terms - 2))]
    root = parts[0]
    for part in parts[1:]:
        root = ir.BinOp("|", root, part)
    sigs = signals(["out", "p", *(f"term{i}" for i in range(terms - 2))])
    result = opt._merge_sop_terms({"out": root}, sigs)["out"]
    if terms == 256:
        assert result == ir.Const(1, 1)
    else:
        assert result is root


def test_deep_oversized_or_is_retained_without_recursive_flattening():
    p = ir.Id("p")
    root = p
    for _ in range(6000):
        root = ir.BinOp("|", root, p)
    result = opt._merge_sop_terms({"out": root}, signals(["p", "out"]))
    assert result["out"] is root
