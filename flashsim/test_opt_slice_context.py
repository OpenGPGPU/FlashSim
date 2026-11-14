"""Recursive slicing memoization preserves context and shared-wire effects."""
import gc
import weakref

from flashsim import opt
from flashsim.ir import BinOp, Const, Extract, Id, Signal, UnaryOp
from flashsim.test_opt_generated_slices import circuit


def query(src, signals, *, assigns=None, uses=None, fresh=None, cache=None,
          seen=None, fuel=64, memo=None):
    return opt._slice_src(src, 1, 1, assigns or {}, signals, uses or {},
                          fresh if fresh is not None else {},
                          cache if cache is not None else {},
                          seen if seen is not None else set(), fuel, {}, memo)


def test_shared_recursive_slice_does_not_expand_a_diamond(monkeypatch):
    signals = {"a": Signal("a", 8, "input")}
    src = Id("a")
    for _ in range(40):
        src = BinOp("&", src, src)
    original = opt._slice_src_uncached
    calls = 0

    def bounded(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls < 100, "slicing expanded a shared expression DAG"
        return original(*args, **kwargs)

    monkeypatch.setattr(opt, "_slice_src_uncached", bounded)
    assert query(src, signals) is None
    assert calls <= 41


def test_seen_and_fuel_are_part_of_the_cache_context():
    signals = {"a": Signal("a", 4, "wire")}
    assigns = {"a": Const(6, 4)}
    memo = opt._SliceMemo()
    assert query(Id("a"), signals, assigns=assigns, seen={"a"}, memo=memo) is None
    assert query(Id("a"), signals, assigns=assigns, fuel=1, memo=memo) is None
    assert query(Id("a"), signals, assigns=assigns, fuel=2, memo=memo) == Const(1, 1)
    assert query(Id("a"), signals, assigns=assigns, seen={"a"}, memo=memo) is None
    assert query(Id("a"), signals, assigns=assigns, fuel=1, memo=memo) is None


def test_new_shared_wire_invalidates_a_cached_parent_failure():
    signals, wide = circuit()
    # Also exercise collision handling without changing generation order.
    signals["wide__b1_1"] = Signal("wide__b1_1", 1, "wire")
    assigns, uses, fresh, cache = {"wide": wide}, {"wide": 2}, {}, {}
    memo = opt._SliceMemo()
    parent = UnaryOp("~", Id("wide"))

    def run(src, fuel):
        return query(src, signals, assigns=assigns, uses=uses, fresh=fresh,
                     cache=cache, fuel=fuel, memo=memo)

    assert run(parent, 2) is None
    assert memo.entries
    wire = run(Id("wide"), 64)
    assert wire == Id("wide__b1_1_1") and list(fresh) == [wire.name]
    assert run(parent, 2) == UnaryOp("~", wire)
    assert len(fresh) == 1


def test_unknown_width_failures_and_their_parents_are_not_cached():
    for wrap in (lambda x: x, lambda x: UnaryOp("~", x)):
        signals = {}
        memo = opt._SliceMemo()
        src = wrap(BinOp("<<", Const(1, 8), Id("later")))
        assert query(src, signals, memo=memo) is None
        assert memo.unknown_widths and not memo.entries
        signals["later"] = Signal("later", 3, "input")
        expected = wrap(BinOp("==", Id("later"), Const(1, 3)))
        assert query(src, signals, memo=memo) == expected


def test_cache_is_bounded_and_full_width_id_returns_the_current_object(monkeypatch):
    monkeypatch.setattr(opt, "_SLICE_RESULT_CACHE_MAX", 3)
    memo = opt._SliceMemo()
    signals = {"a": Signal("a", 1, "input")}
    first, second = Id("a"), Id("a")
    args = (0, 1, {}, signals, {}, {}, {}, set())
    assert opt._slice_src(first, *args, memo=memo) is first
    assert opt._slice_src(second, *args, memo=memo) is second
    for index in range(40):
        query(Const(index, 8), signals, memo=memo)
        assert len(memo.entries) <= 3


def test_pass_cache_does_not_retain_input_graphs_or_signal_tables():
    class Signals(dict):
        pass

    class Expression(BinOp):
        pass

    signals = Signals(a=Signal("a", 8, "input"))
    src = Expression("&", Id("a"), Const(3, 8))
    refs = weakref.ref(signals), weakref.ref(src)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = opt._narrow_bit_slices({"out": Extract(src, 1, 1)}, [], [], signals)
        del result, src, signals
        assert all(ref() is None for ref in refs)
    finally:
        if was_enabled:
            gc.enable()
