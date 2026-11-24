"""Early wire pruning preserves hierarchy boundaries and observable state."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib
import shutil
import subprocess

import pytest

from flashsim import experiment
from flashsim.circt_frontend import _flatten, parse_hw_mlir
from flashsim.emit import emit_cpp
from flashsim.ir import (
    AlwaysFF, Assign, BinOp, ClockedAssertion, ClockedPrint, ClockedProcess,
    Const, DecimalFormat, Id, If, MemRead, MemWrite, Module, NbAssign, Signal,
)
from flashsim.opt import optimize
from flashsim.preflatten import compact_module_wires


def test_boundaries_and_observation_only_cones_survive_without_mutation():
    names = {
        'clock': (1, 'input'), 'input': (8, 'input'), 'out': (8, 'output'),
        'state': (8, 'reg'), 'ram': (8, 'mem'), 'array': (8, 'wire'),
    }
    wires = ['alias', 'child_in', 'child_out', 'write_data', 'guard',
             'check_clock', 'check', 'print_value', 'dead', 'reset_data']
    signals = {n: Signal(n, width, kind, 4 if n in {'ram', 'array'} else 0)
               for n, (width, kind) in names.items()}
    signals.update({n: Signal(n, 8, 'wire') for n in wires})
    assigns = [Assign(n, Id('input')) for n in wires]
    assigns += [Assign('out', Id('alias'))]
    observation = ClockedPrint(Id('check_clock'), Id('check'),
        ('value=', DecimalFormat(Id('print_value'), 8)), 'print', 'stderr')
    source = Module('Parent', signals, assigns,
        AlwaysFF('clock', [If(Id('guard'), [NbAssign('state', Id('child_out'))], [])]),
        ['clock', 'input', 'out'],
        [MemWrite('ram', Const(0, 2), Id('write_data'), Id('guard'))],
        [observation, ClockedAssertion(Id('check_clock'), Id('check'), 'bad', 'check')],
        [ClockedProcess(Id('check_clock'), 'process')],
        [NbAssign('state', Id('reset_data'))])
    source.instances = [(['child_out'], 'child', 'Leaf', {'data': 'child_in'})]
    before_names = list(source.signals)
    result = compact_module_wires(source)
    retained = {a.lhs for a in result.assigns}
    assert {'child_in', 'write_data', 'guard', 'check_clock', 'check',
            'print_value', 'reset_data', 'out'} <= retained
    assert 'dead' not in result.signals and 'alias' not in result.signals
    assert {'state', 'ram', 'array', 'child_out'} <= result.signals.keys()
    assert list(source.signals) == before_names and len(source.assigns) == len(assigns)
    assert result.always.body is source.always.body
    assert result.reset_epilogue is source.reset_epilogue
    assert result.mem_writes is source.mem_writes
    assert result.assertions is source.assertions
    assert result.clocked_processes is source.clocked_processes
    assert result.instances is source.instances


@pytest.mark.parametrize('compact', [False, True])
def test_reused_leaf_reset_memory_and_pre_edge_assertions_execute(tmp_path, compact):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    specs = [('clock', 1, 'input'), ('reset', 1, 'input'), ('bad', 1, 'input'),
             ('data', 8, 'input'), ('out', 8, 'output'), ('q', 8, 'reg'),
             ('ram', 8, 'mem'), ('a', 8, 'wire'), ('b', 8, 'wire'),
             ('d', 8, 'wire'), ('check', 1, 'wire')]
    leaf = Module('Leaf', {n: Signal(n, w, k, 4 if n == 'ram' else 0)
                           for n, w, k in specs},
        [Assign('a', Id('data')), Assign('b', BinOp('^', Id('a'), Const(3, 8))),
         Assign('d', BinOp('^', Id('b'), Const(5, 8))),
         Assign('out', BinOp('+', Id('q'), MemRead('ram', Const(0, 2)))),
         Assign('check', Id('bad'))],
        AlwaysFF('clock', [If(Id('reset'), [NbAssign('q', Const(0, 8))],
                             [NbAssign('q', Id('d'))])]),
        ['clock', 'reset', 'bad', 'data', 'out'],
        [MemWrite('ram', Const(0, 2), Id('q'), Const(1, 1))],
        [ClockedAssertion(Id('clock'), Id('check'), 'pre-edge check', 'state')],
        [ClockedProcess(Id('clock'), 'leaf_clock')])
    signals = {n: Signal(n, w, k) for n, w, k in [
        ('clock', 1, 'input'), ('reset', 1, 'input'), ('bad', 1, 'input'),
        ('left_data', 8, 'input'), ('right_data', 8, 'input'),
        ('left', 8, 'output'), ('right', 8, 'output')]}
    parent = Module('Top', signals, [], AlwaysFF('clock', []), list(signals))
    parent.instances = [([out], out, 'Leaf',
        {'clock': 'clock', 'reset': 'reset', 'bad': 'bad', 'data': out + '_data'})
        for out in ('left', 'right')]
    if compact:
        leaf, parent = compact_module_wires(leaf), compact_module_wires(parent)
        assert 'a' not in leaf.signals and 'b' not in leaf.signals
    flat = _flatten(parent, {'Leaf': leaf}, Counter(Leaf=2))
    assert len(flat.assertions) == len(flat.clocked_processes) == 2
    assert sum(s.kind == 'reg' for s in flat.signals.values()) == 2
    assert sum(s.depth for s in flat.signals.values() if s.kind == 'mem') == 8
    (tmp_path / 'dut.h').write_text(emit_cpp(optimize(flat)))
    (tmp_path / 'check.cpp').write_text('''#include "dut.h"
int main(int argc, char**) {
  TopDut d; unsigned l=0,r=0;
  for (unsigned i=0; i<40; ++i) {
    d.reset = i==0 || i==17;
    d.left_data = i*3; d.right_data = i*5;
    d.bad = argc>1 && i==23;
    unsigned old_l=l,old_r=r;
    l=d.reset ? 0 : (d.left_data^6); r=d.reset ? 0 : (d.right_data^6);
    d.tick(); d.eval_left(); d.eval_right();
    if(d.left!=((l+old_l)&255) || d.right!=((r+old_r)&255)) {
      fprintf(stderr,"step %u: got %u,%u expected %u,%u\\n",i,
        (unsigned)d.left,(unsigned)d.right,(l+old_l)&255,(r+old_r)&255);
      return 4;
    }
  }
  return 0;
}
''')
    binary = tmp_path / 'check'
    subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=address,undefined',
                    str(tmp_path / 'check.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    failed = subprocess.run([str(binary), 'trip'], capture_output=True, text=True)
    assert failed.returncode != 0 and 'pre-edge check' in failed.stderr


def test_mlir_option_prunes_before_replicating_private_wires():
    src = '''hw.module private @Leaf(in %d : i8, out q : i8) {
      %c = hw.constant 3 : i8
      %a = comb.xor %d, %c : i8
      %b = comb.xor %a, %c : i8
      hw.output %b : i8
    }
    hw.module @Top(in %d : i8, out x : i8, out y : i8) {
      %x = hw.instance "left" @Leaf(d: %d: i8) -> (q: i8)
      %y = hw.instance "right" @Leaf(d: %d: i8) -> (q: i8)
      hw.output %x, %y : i8, i8
    }'''
    plain = parse_hw_mlir(src, top='Top')
    compact = parse_hw_mlir(src, top='Top', preflatten_wires=True)
    assert len(compact.signals) < len(plain.signals)
    assert emit_cpp(optimize(compact)) == emit_cpp(optimize(plain))


@pytest.mark.parametrize('compact', [False, True])
def test_parallel_module_preparation_preserves_compiler_output(monkeypatch, capfd, compact):
    leaf = '''hw.module private @Leaf(in %d : i8, out q : i8) {
      %q__flashsim_value = hw.constant 3 : i8
      %q..flashsim.value_ = hw.constant 5 : i8
      %q = comb.xor %d, %q__flashsim_value : i8
      %tmp = comb.xor %q, %q..flashsim.value_ : i8
      hw.output %tmp : i8
    }'''
    src = leaf + '\n' + leaf.replace('@Leaf', '@Second') + '\n' + leaf.replace('@Leaf', '@Unused')
    src += '''
    hw.module @Top(in %d : i8, out x : i8, out y : i8, out z : i8) {
      %x = hw.instance "left" @Leaf(d: %d: i8) -> (q: i8)
      %y = hw.instance "right" @Leaf(d: %d: i8) -> (q: i8)
      %z = hw.instance "other" @Second(d: %d: i8) -> (q: i8)
      hw.output %x, %y, %z : i8, i8, i8
    }'''
    submit = ProcessPoolExecutor.submit
    submitted = []

    def checked_submit(pool, fn, raw, compact):
        submitted.append(raw.name)
        return submit(pool, fn, raw, compact)

    monkeypatch.setattr(ProcessPoolExecutor, 'submit', checked_submit)
    monkeypatch.setenv('FLASHSIM_MODULE_JOBS', '1')
    serial = parse_hw_mlir(src, top='Top', preflatten_wires=compact)
    assert submitted == []
    monkeypatch.setenv('FLASHSIM_MODULE_JOBS', '2')
    monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', '1')
    parallel = parse_hw_mlir(src, top='Absent', preflatten_wires=compact)
    assert submitted == ['Second']
    observed = capfd.readouterr().err
    assert '"phase": "parse.prepare_modules"' in observed
    assert observed.count('"phase": "opt.inline_graph", "event": "begin"') == (2 if compact else 0)
    assert parallel == serial
    assert list(parallel.signals) == list(serial.signals)
    if not compact:
        assert 'left_q__flashsim_value_' in parallel.signals
        assert 'left_q__flashsim_value__' in parallel.signals
    assert emit_cpp(optimize(parallel)) == emit_cpp(optimize(serial))


def test_reuse_rejects_changed_preflatten_option(tmp_path, monkeypatch):
    rtl, header = tmp_path / 'fixture.v', tmp_path / 'dut.h'
    rtl.write_text('// source\n')
    header.write_text('// generated\n')
    monkeypatch.delenv('FLASHSIM_PREFLATTEN_WIRES', raising=False)
    experiment._record_emit_options(header)
    assert experiment._compiled_dut_fresh(rtl, header)
    monkeypatch.setenv('FLASHSIM_PREFLATTEN_WIRES', '1')
    assert not experiment._compiled_dut_fresh(rtl, header)
    experiment._record_emit_options(header)
    assert experiment._compiled_dut_fresh(rtl, header)
    monkeypatch.setenv('FLASHSIM_PREFLATTEN_WIRES', '0')
    assert not experiment._compiled_dut_fresh(rtl, header)


@pytest.mark.parametrize('setting', ['0', '1'])
def test_compile_option_reaches_the_real_circt_parser(tmp_path, monkeypatch, setting):
    compiler = importlib.import_module('flashsim.compile')
    mlir = '''hw.module @Top(in %clock : i1, in %d : i8, out q : i8) {
      %c = hw.constant 3 : i8
      %a = comb.xor %d, %c : i8
      %b = comb.xor %a, %c : i8
      hw.output %b : i8
    }'''
    parser = compiler.parse_hw_mlir
    used = []
    def checked(*args, **kwargs):
        used.append(kwargs.get('preflatten_wires', False))
        return parser(*args, **kwargs)
    # Only the tool's text transport is fixed. Parsing, optional compaction,
    # ordinary optimization and C++ emission execute normally.
    monkeypatch.setattr(compiler, 'circt_verilog_to_mlir', lambda *a, **k: mlir)
    monkeypatch.setattr(compiler, 'parse_hw_mlir', checked)
    monkeypatch.setenv('FLASHSIM_PREFLATTEN_WIRES', setting)
    model, cpp = compiler.compile_verilog('// source', frontend='circt', path=tmp_path / 'Top.sv')
    assert used == [setting == '1']
    assert model.name == 'Top' and 'struct TopDut' in cpp


def test_invalid_preflatten_option_is_rejected_before_compilation(tmp_path, monkeypatch):
    compiler = importlib.import_module('flashsim.compile')
    monkeypatch.setenv('FLASHSIM_PREFLATTEN_WIRES', 'invalid')
    with pytest.raises(ValueError, match='FLASHSIM_PREFLATTEN_WIRES'):
        compiler.compile_verilog('', frontend='native')
    with pytest.raises(ValueError, match='FLASHSIM_PREFLATTEN_WIRES'):
        experiment._record_emit_options(tmp_path / 'dut.h')
