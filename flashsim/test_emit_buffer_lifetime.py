"""Text-line objects must be released before whole-model constructor rewriting."""
import gc
import weakref

import pytest

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, Id, MemRead, MemWrite, Module, NbAssign, Signal


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
