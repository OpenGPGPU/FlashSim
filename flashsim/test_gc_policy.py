"""GC policy preserves model bytes, caller state and failure behavior."""
import gc
import importlib
import subprocess

import pytest

from flashsim.compile import compile_file, compile_verilog
from flashsim.gc_policy import compilation_gc_policy
from flashsim.parse import ParseError


_SOURCE = """module Gate(input clock, input a, input b, output y);
reg q;
assign y = q;
always @(posedge clock) q <= a ^ b;
endmodule
"""


@pytest.fixture(autouse=True)
def restore_gc_state():
    enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)
        (gc.enable if enabled else gc.disable)()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_deferred_nested_scope_restores_exact_caller_state(monkeypatch, enabled, failure):
    monkeypatch.setenv("FLASHSIM_GC_POLICY", "defer")
    (gc.enable if enabled else gc.disable)()
    thresholds = gc.get_threshold()

    @compilation_gc_policy
    def inner():
        assert not gc.isenabled()
        # Even an inner user hook changing the process state cannot leak it.
        gc.enable()
        if failure:
            raise RuntimeError("generation failed")
        return 17

    @compilation_gc_policy
    def outer():
        assert not gc.isenabled()
        try:
            return inner()
        finally:
            assert not gc.isenabled()

    if failure:
        with pytest.raises(RuntimeError, match="generation failed"):
            outer()
    else:
        assert outer() == 17
    assert gc.isenabled() == enabled
    assert gc.get_threshold() == thresholds


@pytest.mark.parametrize("policy", [None, "normal"])
def test_normal_policy_leaves_collection_untouched(monkeypatch, policy):
    if policy is None:
        monkeypatch.delenv("FLASHSIM_GC_POLICY", raising=False)
    else:
        monkeypatch.setenv("FLASHSIM_GC_POLICY", policy)

    @compilation_gc_policy
    def hook():
        assert gc.isenabled()
        gc.disable()

    gc.enable()
    hook()
    assert not gc.isenabled()


def test_compile_scope_includes_lowering_emission_and_shards(tmp_path, monkeypatch):
    module = importlib.import_module("flashsim.compile")
    monkeypatch.setenv("FLASHSIM_GC_POLICY", "defer")
    observed = []
    for name in ["parse_verilog", "optimize", "emit_cpp", "split_dut_methods",
                 "group_primitive_storage", "_write_generated"]:
        original = getattr(module, name)

        def checked(*args, _name=name, _original=original, **kwargs):
            assert not gc.isenabled(), _name
            observed.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, checked)
    source = tmp_path / "Gate.sv"
    source.write_text(_SOURCE)
    gc.enable()
    compile_file(source, tmp_path / "model/dut.h", frontend="native", split=2)
    assert gc.isenabled()
    assert set(observed) == {"parse_verilog", "optimize", "emit_cpp",
                             "split_dut_methods", "group_primitive_storage",
                             "_write_generated"}


def test_failed_lowering_restores_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHSIM_GC_POLICY", "defer")
    gc.enable()
    with pytest.raises(ParseError):
        compile_verilog("not verilog", frontend="native")
    assert gc.isenabled()


@pytest.mark.parametrize("policy", ["", "off", "DEFER"])
def test_invalid_policy_rejected_before_outputs(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("FLASHSIM_GC_POLICY", policy)
    output = tmp_path / "must-not-be-created/dut.h"
    with pytest.raises(ValueError, match="FLASHSIM_GC_POLICY"):
        compile_file(tmp_path / "missing.sv", output, frontend="native")
    assert not output.parent.exists()


def test_policies_generate_identical_split_and_monolithic_files(tmp_path, monkeypatch):
    source = tmp_path / "Gate.sv"
    source.write_text(_SOURCE)
    for split in [0, 2]:
        outputs = []
        for policy in ["normal", "defer"]:
            monkeypatch.setenv("FLASHSIM_GC_POLICY", policy)
            directory = tmp_path / f"{policy}-{split}"
            compile_file(source, directory / "dut.h", frontend="native", split=split)
            outputs.append({path.name: path.read_bytes() for path in directory.iterdir()})
        assert outputs[0] == outputs[1]


@pytest.mark.parametrize("split", [0, 2])
def test_circt_bytes_and_bound_runtime_checks_match(tmp_path, monkeypatch, split):
    # Use the real two-instance bound-check fixture and its original runtime
    # oracle so the policy comparison also exercises reset and pre-NBA state.
    from flashsim.test_clocked_assertions import _design, _native, _tools

    _tools()
    source, _ = _design(tmp_path)
    outputs = []
    for policy in ["normal", "defer"]:
        monkeypatch.setenv("FLASHSIM_GC_POLICY", policy)
        directory = tmp_path / policy
        model = compile_file(source, directory / "dut.h", frontend="circt",
                             assertions=True, split=split)
        assert len(model.assertions) == 4
        outputs.append({path.name: path.read_bytes() for path in
                        [directory / "dut.h", *sorted(directory.glob("dut_*.cpp"))]})
        binary = _native(directory)
        passed = subprocess.run([str(binary), "pass"], capture_output=True, text=True)
        assert passed.returncode == 0, passed.stderr
        for instance in ["left", "right"]:
            failed = subprocess.run([str(binary), instance], capture_output=True, text=True)
            assert failed.returncode == 42, failed.stderr
            assert instance + ".verification_assert" in failed.stderr
            assert 'old count before NBA {check} "quoted"' in failed.stderr
    assert outputs[0] == outputs[1]
