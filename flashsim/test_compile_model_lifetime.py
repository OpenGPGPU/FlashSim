"""Compilation must release its consumed frontend model before optimization."""
import gc
import importlib
import weakref

import pytest

from flashsim.parse import parse_verilog


SOURCE = '''module Gate(input clock, input a, input b, output y);
reg q;
assign y = q;
always @(posedge clock) q <= a ^ b;
endmodule
'''
MLIR = '''hw.module @Gate(in %clock : i1, in %a : i1, in %b : i1, out y : i1) {
  %clk = seq.to_clock %clock
  %x = comb.xor %a, %b : i1
  %q = seq.firreg %x clock %clk : i1
  hw.output %q : i1
}'''


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('frontend', ['native', 'circt'])
def test_compiler_does_not_retain_original_model_through_passes(tmp_path, monkeypatch, fails, frontend):
    compiler = importlib.import_module('flashsim.compile')
    optimizer = importlib.import_module('flashsim.opt')
    parser_name = 'parse_verilog' if frontend == 'native' else 'parse_hw_mlir'
    original_parse = getattr(compiler, parser_name)
    original_dce = optimizer._dce
    parsed = []
    texts = []
    calls = []
    error = ValueError('optimization pass failed')

    def parse(*args, **kwargs):
        model = original_parse(*args, **kwargs)
        parsed.append(weakref.ref(model))
        return model

    class Text(str):
        pass

    def transport(*args, **kwargs):
        text = Text(MLIR)
        texts.append(weakref.ref(text))
        return text

    def dce(*args, **kwargs):
        calls.append(True)
        assert parsed and all(ref() is None for ref in parsed)
        assert all(ref() is None for ref in texts)
        if fails:
            raise error
        return original_dce(*args, **kwargs)

    monkeypatch.setattr(compiler, parser_name, parse)
    # Only the CIRCT transport is fixed data. Both parsers, the optimizer and
    # emitter execute normally; complete real-tool tests cover semantics.
    monkeypatch.setattr(compiler, 'circt_verilog_to_mlir', transport)
    monkeypatch.setattr(optimizer, '_dce', dce)
    enabled = gc.isenabled()
    gc.disable()
    path = tmp_path / 'Gate.sv'
    path.write_text(SOURCE)
    try:
        if fails:
            with pytest.raises(ValueError) as caught:
                compiler.compile_verilog(SOURCE, frontend=frontend, path=path)
            assert caught.value is error
        else:
            model, cpp = compiler.compile_verilog(SOURCE, frontend=frontend, path=path)
            assert model.name == 'Gate' and 'struct GateDut' in cpp
        assert calls and not gc.isenabled()
    finally:
        if enabled:
            gc.enable()


def test_public_optimizer_preserves_callers_input():
    optimizer = importlib.import_module('flashsim.opt')
    model = parse_verilog(SOURCE)
    assigns = model.assigns
    signals = model.signals
    body = model.always.body
    values = [(a.lhs, a.rhs) for a in assigns]
    optimized = optimizer.optimize(model)
    assert optimized is not model
    assert model.assigns is assigns and model.signals is signals
    assert model.always.body is body
    assert [(a.lhs, a.rhs) for a in assigns] == values
