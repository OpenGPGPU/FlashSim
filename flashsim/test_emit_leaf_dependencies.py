"""Bounded leaf reuse must preserve the original graph and emitted model."""
import random

import pytest

from flashsim import emit
from flashsim.ir import BinOp, Const, Id


def _check(assigns, stop, roots=None, **limits):
    cache = emit._ConeLeafCache(assigns, stop, **limits)
    for name in roots if roots is not None else sorted(set(assigns) | stop | {'missing'}):
        assert cache.leaves(name) == emit.cone_leaves(name, assigns, stop)
        assert len(cache._cache) <= cache._max_entries
        assert cache._names == sum(map(len, cache._cache.values())) <= cache._max_names
        assert cache._large_entries == sum(len(answer) > 256 for answer in cache._cache.values()) <= cache._max_large_entries
        assert all(len(answer) <= cache._max_result_names for answer in cache._cache.values())
    return cache


def test_shared_chain_reuses_nodes_without_quadratic_traversals(monkeypatch):
    assigns = {'w0000': Id('state')}
    for n in range(1, 2048):
        assigns[f'w{n:04}'] = BinOp('^', Id(f'w{n-1:04}'), Id('state'))
    original = emit.expr_ids
    calls = 0
    def counted(expr):
        nonlocal calls
        calls += 1
        return original(expr)
    monkeypatch.setattr(emit, 'expr_ids', counted)
    actual = emit._cone_leaves_many(set(assigns), assigns, {'state'})
    assert actual == {name: {'state'} for name in assigns}
    assert calls <= 2 * len(assigns)  # Every named expression expanded once.


@pytest.mark.parametrize('limits', [dict(), dict(max_entries=1, max_names=1, max_result_names=1),
                                   dict(max_entries=0), dict(max_names=0), dict(max_result_names=0)])
def test_diamond_fanout_stops_constants_and_missing_names(limits):
    assigns = {'left': BinOp('+', Id('a'), Id('b')),
               'right': BinOp('+', Id('b'), Id('c')),
               'diamond': BinOp('^', Id('left'), Id('right')),
               'constant': Const(3), 'stop': Id('ignored')}
    for n in range(50):
        assigns[f'out{n}'] = BinOp('|', Id('diamond'), Id('stop'))
    _check(assigns, {'a', 'b', 'c', 'stop'}, **limits)


@pytest.mark.parametrize('limits', [dict(), dict(max_entries=1, max_names=2, max_result_names=2)])
def test_cycles_keep_exact_reachable_leaves(limits):
    assigns = {'a': BinOp('|', Id('b'), Id('left')),
               'b': BinOp('|', Id('c'), Id('right')),
               'c': Id('a'), 'self': Id('self'),
               'fanout': BinOp('&', Id('a'), Id('b')),
               'empty0': Id('empty1'), 'empty1': Id('empty0')}
    _check(assigns, set(), roots=['a', 'b', 'c', 'fanout', 'self', 'empty0', 'empty1'] * 2, **limits)
    _check(assigns, {'b'}, **limits)


def test_random_shared_and_cyclic_graphs_match_original_traversal():
    rng = random.Random(570341)
    for cyclic in (False, True):
        for _ in range(20):
            names = [f'n{n}' for n in range(30)]
            assigns = {}
            for n, name in enumerate(names):
                pool = ['x', 'y', 'unknown'] + (names if cyclic else names[:n])
                assigns[name] = BinOp('^', Id(rng.choice(pool)), Id(rng.choice(pool)))
            roots = names[:]
            rng.shuffle(roots)
            _check(assigns, {'n3', 'x'}, roots=roots * 2,
                   max_entries=7, max_names=15, max_result_names=4)


def test_epoch_snapshots_bindings_and_stop_set_and_answers_are_copies():
    assigns = {'out': Id('a'), 'a': Id('old'), 'untouched': Id('older')}
    stop = set()
    old = emit._ConeLeafCache(assigns, stop)
    assert old.leaves('out') == {'old'}
    assigns['a'] = Id('new')
    assigns['out'] = Id('extra')
    stop.add('a')
    stop.add('untouched')
    assigns['untouched'] = Id('newer')
    assert old.leaves('untouched') == {'older'}  # Not queried before the edit.
    assert old.leaves('out') == {'old'}
    assert old.leaves('a') == {'old'}
    assert emit._ConeLeafCache(assigns, stop).leaves('a') == {'a'}
    fresh = emit._ConeLeafCache(assigns, set())
    result = fresh.leaves('out')
    result.clear()
    assert fresh.leaves('out') == {'extra'}


def test_oversized_results_and_eviction_remain_exact_and_bounded():
    assigns = {'chain0': Id('input0')}
    for n in range(1, 100):
        assigns[f'chain{n}'] = BinOp('|', Id(f'chain{n-1}'), Id(f'input{n}'))
    cache = _check(assigns, set(), roots=list(reversed(list(assigns))) * 2,
                   max_entries=4, max_names=5, max_result_names=3)
    assert 'chain99' not in cache._cache
    for keyword in ('max_entries', 'max_names', 'max_result_names', 'max_large_entries'):
        with pytest.raises(ValueError):
            emit._ConeLeafCache(assigns, set(), **{keyword: -1})


@pytest.mark.parametrize('large_limit', [0, 1, 2])
def test_wide_shared_frontier_keeps_large_entry_and_total_name_bounds(large_limit):
    from flashsim.ir import Concat
    assigns = {'a-wide': Concat(tuple(Id(f'input{n}') for n in range(300))), 'z29': Id('a-wide')}
    for n in reversed(range(29)):
        assigns[f'z{n}'] = BinOp('|', Id('a-wide'), Id(f'z{n+1}'))
    _check(assigns, set(), roots=['z0', 'z10', 'z20'] * 2,
           max_entries=20, max_names=1500, max_result_names=400, max_large_entries=large_limit)


@pytest.mark.parametrize('model', ['memory', 'ready0', 'ready1'])
def test_existing_emitter_models_keep_byte_identical_cpp_and_runtime(tmp_path, monkeypatch, model):
    from flashsim.test_emit_non_power_two_mem import test_non_power_of_two_memory_read_write
    from flashsim.test_ready_group_safety import test_later_unscheduled_producer_updates_registers
    optimized = emit._cone_leaves_many
    headers = []
    for use_memo in (False, True):
        directory = tmp_path / str(use_memo)
        directory.mkdir()
        with monkeypatch.context() as patch:
            patch.setattr(emit, '_cone_leaves_many', optimized if use_memo else
                          lambda roots, assigns, stop: {w: emit.cone_leaves(w, assigns, stop) for w in sorted(roots)})
            if model == 'memory':
                test_non_power_of_two_memory_read_write(directory)
            else:
                test_later_unscheduled_producer_updates_registers(directory, patch, model[-1])
        headers.append((directory / 'dut.h').read_bytes())
    assert headers[0] == headers[1]
