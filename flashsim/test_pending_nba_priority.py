"""Idle gates must preserve a later write's priority over a pending NBA."""

import shutil
import subprocess

import pytest

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Const, Id, If, Module, NbAssign, Signal
from flashsim.opt import optimize


def _run(tmp_path, body, check, extra=0):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    signals = {n: Signal(n, w, kind) for n, w, kind in [
        ('clock', 1, 'input'), ('core_reset', 1, 'input'), ('clear', 1, 'input'),
        ('data', 8, 'input'), ('hold', 8, 'input'), ('a', 8, 'reg'), ('b', 8, 'reg')]}
    for i in range(extra):
        name = f'r{i}'
        signals[name] = Signal(name, 8, 'reg')
        body.append(NbAssign(name, Const(0, 8)))
    model = Module('Priority', signals, [], AlwaysFF('clock', body),
                   [n for n, s in signals.items() if s.kind == 'input'])
    optimized = optimize(model)
    (tmp_path / 'dut.h').write_text(emit_cpp(optimized))
    (tmp_path / 'check.cpp').write_text('#include "dut.h"\nint main() { PriorityDut d;\n' + check + '\n}\n')
    binary = tmp_path / 'check'
    subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=address,undefined',
                    str(tmp_path / 'check.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    return optimized


def test_repeated_reset_overrides_data_even_when_state_already_holds_reset(tmp_path):
    body = [If(Id('core_reset'), [NbAssign(n, Const(0, 8))], [NbAssign(n, Id('data'))])
            for n in ('a', 'b')]
    model = _run(tmp_path, body, '''
      for(unsigned i=0;i<40;++i) {
        d.core_reset=(i%4)>=2; d.data=i+11; d.tick();
        unsigned expected=d.core_reset ? 0 : d.data;
        if(d.a!=expected || d.b!=expected) return 1;
      }
    ''')
    assert model.reset_epilogue


@pytest.mark.parametrize('nested', [False, True])
def test_conditional_clear_cancels_a_pending_write(tmp_path, nested):
    clears = [If(Id('clear'), [NbAssign(n, Const(0, 8))], []) for n in ('a', 'b')]
    if nested:
        clears = [If(Id('core_reset'), clears, [])]
    _run(tmp_path, [NbAssign('a', Id('data')), NbAssign('b', Id('data')), *clears], '''
      for(unsigned i=0;i<40;++i) {
        d.clear=(i%4)>=2; d.core_reset=1; d.data=i+11; d.tick();
        unsigned expected=d.clear ? 0 : d.data;
        if(d.a!=expected || d.b!=expected) return 1;
      }
    ''')


@pytest.mark.parametrize('later', [False, True])
def test_reset_hoist_never_crosses_another_writer(tmp_path, later):
    reset = [If(Id('core_reset'), [NbAssign(n, Const(0, 8))], [NbAssign(n, Id('data'))])
             for n in ('a', 'b')]
    other = [NbAssign('a', Id('hold'))]
    _run(tmp_path, reset + other if later else other + reset, f'''
      for(unsigned i=0;i<40;++i) {{
        d.core_reset=(i%4)>=2; d.data=i+11; d.hold=7; d.tick();
        unsigned expected_a={'d.hold' if later else '(d.core_reset ? 0 : d.data)'};
        if(d.a!=expected_a || d.b!=(d.core_reset ? 0 : d.data)) return 1;
      }}
    ''')


@pytest.mark.parametrize('rhs', ['a', 'hold'])
def test_sticky_copy_preserves_a_later_old_value_write(tmp_path, rhs):
    _run(tmp_path, [NbAssign('a', Id('data')), NbAssign('a', Id(rhs))], '''
      for(unsigned i=0;i<40;++i) {
        d.data=i+11; d.hold=0; d.tick();
        if(d.a!=0) return 1;
      }
    ''', extra=256)
