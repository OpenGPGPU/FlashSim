"""CIRCT prints a bare hw.output when a module has no results."""

import pytest

from flashsim.circt_frontend import CirctError, parse_hw_mlir


def test_operand_less_hw_output_lowers_a_module_with_no_results():
    src = """
    hw.module @Sink(in %clock : i1, in %io_ready : i1) {
      %true = hw.constant true
      hw.output
    }
    """
    mod = parse_hw_mlir(src, top="Sink")
    assert mod.name == "Sink"
    assert [name for name in mod.ports if mod.signals[name].kind == "output"] == []


def test_operand_less_hw_output_rejects_a_module_that_still_has_results():
    src = """
    hw.module @Gate(in %a : i1, out y : i1) {
      hw.output
    }
    """
    with pytest.raises(CirctError, match="arity 0 != output ports 1"):
        parse_hw_mlir(src, top="Gate")


def test_zero_result_instance_inlines_a_sink_module():
    src = """
    hw.module private @Sink(in %clock : i1, in %io_ready : i1) {
      hw.output
    }
    hw.module @Parent(in %clock : i1, in %ready : i1, out y : i1) {
      hw.instance "textured" @Sink(clock: %clock: i1, io_ready: %ready: i1) -> ()
      hw.output %ready : i1
    }
    """
    mod = parse_hw_mlir(src, top="Parent")
    assert mod.name == "Parent"
    assert any(assign.lhs == "y" for assign in mod.assigns)


def test_hw_output_still_connects_named_results():
    src = """
    hw.module @Buf(in %a : i1, out y : i1) {
      hw.output %a : i1
    }
    """
    mod = parse_hw_mlir(src, top="Buf")
    assert any(assign.lhs == "y" for assign in mod.assigns)
