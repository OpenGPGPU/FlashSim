"""Large intermediate objects must be released after their last emission use."""
import gc
import re
import weakref

import pytest

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, Id, MemRead, MemWrite, Module, NbAssign, Signal


@pytest.mark.parametrize('width,large,keep_leaves', [
    (8, True, False),
    (256, True, True),
    (256, False, False),
])
def test_leaf_storage_lives_only_until_its_last_consumer(monkeypatch, width, large, keep_leaves):
    outputs = [f'y{i}' for i in range(130)]
    signals = {'clock': Signal('clock', 1, 'input'),
               'data': Signal('data', width, 'input'),
               'q': Signal('q', width, 'reg')}
    signals.update((name, Signal(name, width, 'output')) for name in outputs)
    module = Module('LeafStorage', signals, [Assign(name, Id('data')) for name in outputs],
                    AlwaysFF('clock', [NbAssign('q', Id(outputs[0]))]), list(signals))
    monkeypatch.setattr(emit, '_EAGER', True)
    monkeypatch.setattr(emit, '_EAGER_KEY_RE', re.compile(r'.*'))
    monkeypatch.setattr(emit, '_LARGE_WRITES', 1 if large else 256)
    original_leaves = emit._cone_leaves_many
    original_pass_inputs = emit._pass_inputs
    refs = []

    def track_leaves(*args):
        result = original_leaves(*args)
        refs.extend(weakref.ref(leaves) for leaves in result.values())
        return result

    def check_pass_inputs(*args):
        assert refs and all((ref() is not None) == keep_leaves for ref in refs)
        return original_pass_inputs(*args)

    monkeypatch.setattr(emit, '_cone_leaves_many', track_leaves)
    monkeypatch.setattr(emit, '_pass_inputs', check_pass_inputs)
    assert 'struct LeafStorageDut' in emit.emit_cpp(module)


@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_line_storage_is_released_before_constructor(monkeypatch, memory, fails):
    signals = {name: Signal(name, width, kind) for name, width, kind in (
        ('clock', 1, 'input'), ('d', 8, 'input'), ('en', 1, 'input'),
        ('addr', 2, 'input'), ('q', 8, 'reg'), ('y', 8, 'output'))}
    if memory:
        signals['mem'] = Signal('mem', 8, 'mem', 4)
    module = Module('Buffer', signals,
                    [Assign('y', MemRead('mem', Id('addr')) if memory else Id('q'))],
                    AlwaysFF('clock', [NbAssign('q', Id('d'))]), list(signals),
                    [MemWrite('mem', Id('addr'), Id('d'), Id('en'))] if memory else [])
    # Cover the large-write closure and memory-store aliases of the line
    # buffer, without constructing a product-sized model in a unit test.
    monkeypatch.setattr(emit, '_LARGE_WRITES', 1)
    force_wake = emit._force_wake_method
    constructor = emit._bulk_state_constructor
    refs = []
    marker = '// buffer lifetime test marker'

    class Line(str):
        pass

    def mark(*args, **kwargs):
        lines = force_wake(*args, **kwargs)
        line = Line(marker)
        refs.append(weakref.ref(line))
        lines.append(line)
        return lines

    error = ValueError('constructor failure')

    def check(cpp, cls):
        assert marker in cpp
        assert refs and all(ref() is None for ref in refs)
        assert cpp.endswith('\n\n')
        if fails:
            raise error
        return constructor(cpp, cls)

    monkeypatch.setattr(emit, '_force_wake_method', mark)
    monkeypatch.setattr(emit, '_bulk_state_constructor', check)
    enabled = gc.isenabled()
    gc.disable()
    try:
        if fails:
            with pytest.raises(ValueError) as caught:
                emit.emit_cpp(module)
            assert caught.value is error
        else:
            cpp = emit.emit_cpp(module)
            assert marker in cpp and 'struct BufferDut' in cpp
        assert not gc.isenabled()
    finally:
        if enabled:
            gc.enable()
