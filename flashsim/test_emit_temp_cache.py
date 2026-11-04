"""Nested statement cones stay exact when repeated subtrees are memoized."""
import random

import flashsim.emit as emit
from flashsim.ir import BinOp, Const, Id, If, NbAssign


def _naive(stmts, cached, assigns, stop):
    result = set()
    for stmt in stmts:
        if isinstance(stmt, NbAssign):
            result.update(emit._uncached_needed(stmt.rhs, assigns, cached, stop))
        else:
            result.update(emit._uncached_needed(stmt.cond, assigns, cached, stop))
            result.update(_naive(stmt.then_body, cached, assigns, stop))
            result.update(_naive(stmt.else_body, cached, assigns, stop))
    return result


def test_nested_and_shared_arms_match_uncached_traversal(monkeypatch):
    rng = random.Random(92351)
    assigns = {'t0': BinOp('+', Id('input0'), Const(1)),
               't1': BinOp('&', Id('t0'), Id('input1')),
               't2': BinOp('|', Id('t1'), Id('input2')),
               't3': BinOp('^', Id('t2'), Id('t0'))}
    stop = {'input0', 'input1', 'input2'}
    cached = {'t2'}
    exprs = [Id(name) for name in ('input0', 't0', 't1', 't2', 't3')]
    pool = [NbAssign('r', rng.choice(exprs)) for _ in range(8)]
    layer = pool[:]
    for _ in range(5):
        # Reuse entire statements in multiple arms, as in shared optimized IR.
        layer = [If(rng.choice(exprs), rng.sample(layer, 2),
                    rng.sample(layer, 2)) for _ in range(6)]
        pool.extend(layer)
    expected = [_naive([stmt], cached, assigns, stop) for stmt in pool]
    calls = 0
    original = emit._uncached_needed
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(emit, '_uncached_needed', counted)
    collector = emit._UncachedTempCollector(cached, assigns, stop)
    actual = [collector.collect([stmt]) for stmt in pool]
    assert actual == expected
    assert calls <= len(exprs)
    actual[-1].clear()
    assert collector.collect([pool[-1]]) == expected[-1]


def test_fresh_collector_reflects_a_changed_cached_set():
    assigns = {'temp': BinOp('+', Id('input'), Const(1))}
    body = [NbAssign('result', Id('temp'))]
    assert emit._UncachedTempCollector(set(), assigns, {'input'}).collect(body) == {'temp'}
    assert emit._UncachedTempCollector({'temp'}, assigns, {'input'}).collect(body) == set()
