"""Future generation phase observations must not alter emitted model bytes."""
import importlib
import json

import pytest

from flashsim import phase_timing
from flashsim.compile import compile_file


SOURCE = '''module Gate(input clock, input a, input b, output y);
reg q;
assign y = q;
always @(posedge clock) q <= a ^ b;
endmodule
'''
MLIR = '''hw.module @Gate(in %clock : i1, in %a : i1, in %b : i1, out y : i1) {
  %x = comb.xor %a, %b : i1
  hw.output %x : i1
}'''


def events(text):
    prefix = '[flashsim-phase] '
    return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]


@pytest.mark.parametrize('frontend', ['native', 'circt'])
@pytest.mark.parametrize('split', [0, 2])
def test_observations_preserve_complete_generated_bytes(tmp_path, monkeypatch, capsys, frontend, split):
    compiler = importlib.import_module('flashsim.compile')
    # Exercise the actual parser/optimizer/emitter, but never launch CIRCT or
    # any external model. The frontend transport alone is a fixed unit stub.
    def transport(path, *, assertions, output):
        output.write(MLIR.encode())
    monkeypatch.setattr(compiler, 'circt_verilog_to_mlir', transport)
    source = tmp_path / 'Gate.sv'
    source.write_text(SOURCE)
    output = []
    for setting in ('0', '1'):
        monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', setting)
        directory = tmp_path / setting
        compile_file(source, directory / 'dut.h', frontend=frontend, split=split)
        output.append({p.name: p.read_bytes() for p in directory.glob('dut*')})
        records = events(capsys.readouterr().err)
        if setting == '0':
            assert records == []
            continue
        stack, names = [], set()
        for record in records:
            names.add(record['phase'])
            if record['event'] == 'begin':
                assert set(record) == {'event', 'phase'}
                stack.append(record['phase'])
            else:
                assert set(record) == {'event', 'phase', 'seconds', 'status'}
                assert stack.pop() == record['phase']
                assert record['status'] == 'ok' and record['seconds'] >= 0
        assert not stack
        assert {'optimize', 'emit_cpp', 'opt.merge_sop', 'opt.reassoc_and',
                'opt.dedup', 'opt.used_bits', 'opt.lut', 'opt.dce',
                'opt.inline_graph', 'opt.inline_worklist', 'opt.inline_expand',
                'opt.post_dedup_fold', 'opt.post_used_bits_fold',
                'emit.cached_wires', 'emit.deep_mux_chunks',
                'emit.leaf_dependencies', 'emit.cached_dependencies',
                'emit.eager_regions', 'emit.partitions', 'emit.pass_inputs',
                'emit.hold_wakes', 'emit.pass_wakes', 'emit.wake_tables',
                'emit.eager_bodies', 'emit.join_text', 'emit.constructor'} <= names
        assert ('split_dut_methods' in names) == bool(split)
        assert ('frontend.circt' in names) == (frontend == 'circt')
        assert ('parse.hw_mlir' if frontend == 'circt' else 'parse.verilog') in names
    assert output[0] == output[1]


def test_disabled_observation_does_not_read_clock(monkeypatch, capsys):
    monkeypatch.delenv('FLASHSIM_PHASE_TIMINGS', raising=False)
    def forbidden():
        raise AssertionError('disabled phase read the timer')
    monkeypatch.setattr(phase_timing.time, 'perf_counter_ns', forbidden)
    with phase_timing.phase('disabled'):
        pass
    assert capsys.readouterr().err == ''


def test_failure_is_reported_and_original_exception_preserved(monkeypatch, capsys):
    monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', '1')
    error = ValueError('original exception')
    with pytest.raises(ValueError) as caught:
        with phase_timing.phase('failed'):
            raise error
    assert caught.value is error
    observed = events(capsys.readouterr().err)
    assert observed[0] == {'event': 'begin', 'phase': 'failed'}
    assert observed[1]['status'] == 'error'


def test_broken_diagnostic_stream_does_not_fail_compile(monkeypatch):
    class Broken:
        def write(self, _):
            raise OSError('diagnostic stream closed')
    monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', '1')
    monkeypatch.setattr(phase_timing.sys, 'stderr', Broken())
    with phase_timing.phase('ignored diagnostic failure'):
        pass


def test_emitter_dependency_failure_keeps_its_original_exception(tmp_path, monkeypatch, capsys):
    emitter = importlib.import_module('flashsim.emit')
    error = ValueError('dependency input failure')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(emitter, '_pass_inputs', fail)
    monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', '1')
    source = tmp_path / 'Gate.sv'
    source.write_text(SOURCE)
    with pytest.raises(ValueError) as caught:
        compile_file(source, tmp_path / 'dut.h', frontend='native')
    assert caught.value is error
    observed = [event for event in events(capsys.readouterr().err)
                if event['phase'] == 'emit.pass_inputs']
    assert [event['event'] for event in observed] == ['begin', 'end']
    assert observed[-1]['status'] == 'error'


def test_constructor_failure_is_reported_after_join(tmp_path, monkeypatch, capsys):
    emitter = importlib.import_module('flashsim.emit')
    error = ValueError('constructor input failure')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(emitter, '_bulk_state_constructor', fail)
    monkeypatch.setenv('FLASHSIM_PHASE_TIMINGS', '1')
    source = tmp_path / 'Gate.sv'
    source.write_text(SOURCE)
    with pytest.raises(ValueError) as caught:
        compile_file(source, tmp_path / 'dut.h', frontend='native')
    assert caught.value is error
    observed = [event for event in events(capsys.readouterr().err)
                if event['phase'] in {'emit.join_text', 'emit.constructor'}]
    assert [(event['phase'], event['event']) for event in observed] == [
        ('emit.join_text', 'begin'), ('emit.join_text', 'end'),
        ('emit.constructor', 'begin'), ('emit.constructor', 'end')]
    assert observed[1]['status'] == 'ok'
    assert observed[3]['status'] == 'error'
