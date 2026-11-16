"""Inlining must discard dependency bookkeeping before rebuilding expressions."""
import gc
import weakref
from collections import deque

import pytest

import flashsim.opt as optimizer
from flashsim.ir import BinOp, Const, Id


@pytest.mark.parametrize('fails', [False, True])
def test_dependency_graph_is_released_before_expansion(monkeypatch, fails):
    references = []
    expanding = False
    visited = []
    error = ValueError('expression expansion failed')
    original_counts = optimizer.expr_id_counts
    original_uses = optimizer._count_uses

    class Counts(dict):
        pass

    def track(values):
        counts = Counts(values)
        references.append(weakref.ref(counts))
        return counts

    def count_ids(expr):
        return track(original_counts(expr))

    def count_uses(*args, **kwargs):
        return track(original_uses(*args, **kwargs))

    class Pending(deque):
        def __bool__(self):
            nonlocal expanding
            remaining = len(self) > 0
            if not remaining:
                expanding = True
            return remaining

    class CheckedId(Id):
        def __getattribute__(self, name):
            if name == 'name' and expanding:
                visited.append(True)
                assert references and all(ref() is None for ref in references)
                if fails:
                    raise error
            return super().__getattribute__(name)

    def forbidden_collection(*args, **kwargs):
        raise AssertionError('inlining must not force cyclic collection')

    monkeypatch.setattr(optimizer, 'expr_id_counts', count_ids)
    monkeypatch.setattr(optimizer, '_count_uses', count_uses)
    monkeypatch.setattr(optimizer, 'deque', Pending)
    monkeypatch.setattr(gc, 'collect', forbidden_collection)
    assigns = {'unused': Const(0, 1), 'alias': CheckedId('input'),
               'out': BinOp('^', Id('alias'), CheckedId('other'))}
    enabled = gc.isenabled()
    gc.disable()
    try:
        if fails:
            with pytest.raises(ValueError) as caught:
                optimizer._inline_single_use(assigns, [], {'out'})
            assert caught.value is error
        else:
            result, body = optimizer._inline_single_use(assigns, [], {'out'})
            assert result == {'out': BinOp('^', CheckedId('input'), CheckedId('other'))}
            assert body == []
        assert visited and not gc.isenabled()
    finally:
        if enabled:
            gc.enable()
