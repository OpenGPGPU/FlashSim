"""Bounded leaf reuse must preserve the original graph and emitted model."""
import random

import pytest

from flashsim import emit
from flashsim.ir import BinOp, Concat, Const, Id


def _check(assigns, stop, roots=None, **limits):
    if roots is None:
        roots = set(assigns) | stop | {'missing'}
    roots = set(roots)
    assert emit._cone_leaves_many(roots, assigns, stop, **limits) == {
        name: emit.cone_leaves(name, assigns, stop) for name in roots
    }


def test_shared_chain_reuses_nodes_without_quadratic_traversals(monkeypatch):
    assigns = {'w0000': Id('state')}
    for n in range(1, 2048):
        assigns[f'w{n:04}'] = BinOp('^', Id(f'w{n-1:04}'), Id('state'))
    original = emit.expr_ids
    calls = 0
    def counted(expr, **options):
        nonlocal calls
        calls += 1
        return original(expr, **options)
    monkeypatch.setattr(emit, 'expr_ids', counted)
    actual = emit._cone_leaves_many(set(assigns), assigns, {'state'})
    assert actual == {name: {'state'} for name in assigns}
    assert calls <= 2 * len(assigns)  # Every named expression expanded once.


@pytest.mark.parametrize('limits', [dict(), dict(temporary_limit=1, large_leaf_budget=1),
                                   dict(temporary_limit=0), dict(large_leaf_budget=0)])
def test_diamond_fanout_stops_constants_and_missing_names(limits):
    assigns = {'left': BinOp('+', Id('a'), Id('b')),
               'right': BinOp('+', Id('b'), Id('c')),
               'diamond': BinOp('^', Id('left'), Id('right')),
               'constant': Const(3), 'stop': Id('ignored')}
    for n in range(50):
        assigns[f'out{n}'] = BinOp('|', Id('diamond'), Id('stop'))
    _check(assigns, {'a', 'b', 'c', 'stop'}, **limits)


@pytest.mark.parametrize('limits', [dict(), dict(temporary_limit=1, large_leaf_budget=2)])
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
                   temporary_limit=7, large_leaf_budget=15)


def test_rewritten_graph_uses_fresh_answers_and_preserves_previous_results():
    assigns = {'out': Id('a'), 'a': Id('old'), 'untouched': Id('older')}
    stop = set()
    roots = {'out', 'untouched', 'a'}
    old = emit._cone_leaves_many(roots, assigns, stop)
    assigns['a'] = Id('new')
    assigns['out'] = Id('extra')
    stop.update(('a', 'untouched'))
    assigns['untouched'] = Id('newer')
    assert old == {'out': {'old'}, 'untouched': {'older'}, 'a': {'old'}}
    current = emit._cone_leaves_many(roots, assigns, stop)
    assert current == {'out': {'extra'}, 'untouched': {'untouched'}, 'a': {'a'}}
    assert all(isinstance(value, frozenset) for value in current.values())


def test_oversized_results_and_eviction_remain_exact_and_bounded():
    assigns = {'chain0': Id('input0')}
    for n in range(1, 100):
        assigns[f'chain{n}'] = BinOp('|', Id(f'chain{n-1}'), Id(f'input{n}'))
    _check(assigns, set(), temporary_limit=4, large_leaf_budget=5)
    for limits in ((-1, 5), (4, -1)):
        with pytest.raises(ValueError):
            emit._LeafMemo(*limits)


def test_small_and_large_answers_share_the_same_name_budget():
    memo = emit._LeafMemo(20, 500)
    small = frozenset(f'small{n}' for n in range(128))
    large = frozenset(f'large{n}' for n in range(300))
    memo.remember('retired', small)
    memo.remember('kept', small)
    memo.get('kept')
    memo.remember('wide', large)
    assert list(memo.answers) == ['kept', 'wide']
    assert memo.names == sum(map(len, memo.answers.values())) == 428
    memo.remember('oversize', small | large | frozenset(f'other{n}' for n in range(128)))
    assert memo.get('oversize') is None


def test_large_entry_limit_uses_the_large_lru():
    answer = frozenset(f'leaf{n}' for n in range(257))
    memo = emit._LeafMemo(32768, 1048576)
    memo.remember('small', frozenset({'one'}))
    for n in range(1024):
        memo.remember(f'large{n}', answer)
    assert memo.get('large0') is answer
    memo.remember('new', answer)
    assert len(memo.large) == 1024
    assert memo.get('large1') is None
    assert memo.get('large0') is answer
    assert memo.get('small') == {'one'}


def test_wide_frontier_falls_back_without_changing_its_root(monkeypatch):
    shared = {f'input{n}' for n in range(48)}
    assigns = {'a-wide': Concat(tuple(Id(name) for name in sorted(shared))),
               'z100': Id('tail')}
    for n in reversed(range(100)):
        assigns[f'z{n:03}'] = Concat((Id('a-wide'), Id(f'm{n}'), Id(f'z{n+1:03}')))
    original_ids, independent = emit.expr_ids, emit.cone_leaves
    calls = []

    def follow_in_name_order(expr, **_options):
        return original_ids(expr)

    def count_fallback(root, assigns, stop):
        calls.append(root)
        return independent(root, assigns, stop)

    # Every frame merges the shared cone and one distinct leaf before
    # descending. A short chain already exceeds the active name budget.
    monkeypatch.setattr(emit, 'expr_ids', follow_in_name_order)
    monkeypatch.setattr(emit, 'cone_leaves', count_fallback)
    assert emit._cone_leaves_many({'z000'}, assigns, set(), large_leaf_budget=192) == {
        'z000': shared | {f'm{n}' for n in range(100)} | {'tail'}
    }
    assert calls == ['z000']


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
