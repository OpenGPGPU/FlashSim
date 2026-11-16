"""Completed IR walks must release their inputs without waiting for cyclic GC."""
import gc
import weakref

import pytest

from flashsim import emit, opt
from flashsim.ir import BinOp, Id, If, NbAssign, Signal, Ternary
from flashsim import partition_clocked


class TrackedDict(dict):
    """An ordinary input map with an observable lifetime."""


class TrackedName(str):
    """An ordinary IR name with an observable lifetime."""


@pytest.fixture(autouse=True)
def no_cyclic_collection(monkeypatch):
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()

    def forbidden_collect(*args, **kwargs):
        raise AssertionError("an IR walk must not force cyclic collection")

    monkeypatch.setattr(gc, "collect", forbidden_collect)
    try:
        yield
        assert not gc.isenabled()
    finally:
        (gc.enable if enabled else gc.disable)()


def inputs():
    assigns = TrackedDict(t0=Id("a"), t1=BinOp("&", Id("t0"), Id("b")))
    sigs = TrackedDict({n: Signal(n, 1, "input") for n in ("a", "b", "c")})
    sigs.update({n: Signal(n, 1, "wire") for n in assigns})
    sigs["q"] = Signal("q", 1, "reg")
    body = [If(Id("c"), [If(Id("b"), [NbAssign("q", Id("t1"))], [])],
               [NbAssign("q", Id("a"))])]
    return assigns, sigs, body


@pytest.mark.parametrize("chain", [False, True])
def test_reassociation_releases_inputs_on_early_and_rebuilt_paths(chain):
    def run():
        assigns, sigs, _ = inputs()
        if chain:
            assigns["t1"] = BinOp("&", Id("a"), BinOp("&", Id("b"), Id("c")))
        refs = [weakref.ref(assigns), weakref.ref(sigs)]
        result = opt._reassoc_and_chains(assigns, sigs)
        assert set(assigns) <= set(result)
        return refs

    assert all(ref() is None for ref in run())


@pytest.mark.parametrize("visitor", ["cse", "used_bits", "order", "bulky",
                                      "deep_mux", "stop", "cut"])
def test_completed_walk_releases_ir_maps(visitor):
    def run():
        assigns, sigs, body = inputs()
        refs = [weakref.ref(assigns), weakref.ref(sigs)]
        if visitor == "cse":
            assert opt._cse_sibling_or_prefixes(body, assigns, sigs) == body
        elif visitor == "used_bits":
            opt._apply_used_bits(assigns, body, [], [], sigs)
        elif visitor == "order":
            assert emit._order_ssa_temps(set(assigns), assigns, set(), set()) == ["t0", "t1"]
        elif visitor == "bulky":
            emit._promote_bulky_gated_cones(body, assigns, set(), set())
        elif visitor == "deep_mux":
            assigns["mux"] = Ternary(Id("c"), Id("a"), Id("b"))
            emit._extract_deep_mux_chunks(assigns, sigs, set(), body)
        elif visitor == "stop":
            assert emit._stmt_stop_leaves(body[0], assigns, set(), {}) == {"a", "b", "c"}
        elif visitor == "cut":
            memo = TrackedDict()
            refs.append(weakref.ref(memo))
            assert emit._stmt_cut_leaves(body[0], assigns, set(), {"t1"}, memo) == (
                {"a", "b", "c"}, {"t1"})
        return refs

    assert all(ref() is None for ref in run()), visitor


def test_statement_name_collection_releases_names():
    def run():
        name = TrackedName("q")
        ref = weakref.ref(name)
        stmt = If(Id("a"), [If(Id("b"), [NbAssign(name, Id("c"))], [])], [])
        assert emit._stmt_names(stmt) == {"a", "b", "q"}
        return ref

    assert run()() is None


def test_partition_releases_target_mapping(monkeypatch):
    from flashsim.test_partition_clocked import RTL

    refs = []
    original = partition_clocked._Parser.parse

    def tracked_parse(parser):
        root = original(parser)
        names = [TrackedName(name) for name in parser.counts]
        refs.extend(weakref.ref(name) for name in names)
        parser.counts = type(parser.counts)({name: parser.counts[name] for name in names})
        return root

    monkeypatch.setattr(partition_clocked._Parser, "parse", tracked_parse)
    _, counts = partition_clocked.partition_clocked_process(
        RTL, groups=3, assume_stable_edge_inputs=True
    )
    assert counts == {"targets": 4, "assignments": 13, "groups": 3}
    assert len(refs) == 4
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("visitor", ["order", "stop", "reassociation"])
def test_failed_walk_also_releases_ir_maps(monkeypatch, visitor):
    def fail(*args, **kwargs):
        raise RuntimeError("injected visitor failure")

    def run():
        assigns, sigs, body = inputs()
        refs = [weakref.ref(assigns), weakref.ref(sigs)]
        try:
            if visitor == "order":
                monkeypatch.setattr(emit, "_uncached_needed", fail)
                emit._order_ssa_temps(set(assigns), assigns, set(), set())
            elif visitor == "stop":
                monkeypatch.setattr(emit, "_cond_stop_leaves", fail)
                emit._stmt_stop_leaves(body[0], assigns, set(), {})
            else:
                monkeypatch.setattr(opt, "expr_width", fail)
                opt._reassoc_and_chains(assigns, sigs)
        except RuntimeError as error:
            assert str(error) == "injected visitor failure"
        else:
            pytest.fail("injected failure did not propagate")
        return refs

    assert all(ref() is None for ref in run()), visitor
