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
    original = emit._uncached_names
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    monkeypatch.setattr(emit, '_uncached_names', counted)
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


def test_inline_site_count_stops_at_threshold_and_counts_each_statement_once(monkeypatch):
    fat = Id('input')
    for _ in range(12):
        fat = BinOp('+', fat, Const(1))
    shared = BinOp('&', Id('leaf'), Id('stop'))
    assigns = {'shared': shared, 'leaf': Id('input'), 'once': fat, 'twice': fat,
               'stop': Id('blocked'), 'blocked': Const(0)}
    roots = {f'root_{index}': Id('shared') for index in range(40)}
    assigns.update(roots)
    cached = set(roots)
    body = [If(Id('stop'), [NbAssign('q', Id('once')) for _ in range(32)], []),
            NbAssign('q', Id('twice')), NbAssign('q', Id('twice'))]
    original = emit.expr_ids
    shared_walks = 0
    original_count = emit._expr_node_count
    fat_cost_queries = 0

    def counted(expr, **options):
        nonlocal shared_walks
        if expr is shared:
            shared_walks += 1
        return original(expr, **options)

    def count_cost(expr):
        nonlocal fat_cost_queries
        if expr is fat:
            fat_cost_queries += 1
        return original_count(expr)

    monkeypatch.setattr(emit, 'expr_ids', counted)
    monkeypatch.setattr(emit, '_expr_node_count', count_cost)
    monkeypatch.setattr(emit, '_SHARED_INLINE_ROUNDS', 1)
    emit._promote_shared_inlines(body, assigns, cached, {'input', 'stop'})
    assert cached == set(roots) | {'shared', 'leaf', 'twice'}
    assert shared_walks == 2
    assert fat_cost_queries == 1
