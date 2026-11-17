"""Folded-away checks leave empty event loops without observable effects."""
import shutil
import subprocess

import pytest

from flashsim.circt_frontend import CirctError, parse_hw_mlir
from flashsim.compile import compile_file
from flashsim.toolchain import find_bin


_EMPTY = """
hw.module @Empty(in %clock : i1, in %data : i8, out q : i8) {
  llhd.process {
    cf.br ^wait
  ^wait:
    llhd.wait (%clock : i1), ^wake(%clock : i1)
  ^wake(%previous: i1):
    cf.br ^wait
  }
  hw.output %data : i8
}
"""


def test_empty_event_loop_has_no_checks_or_state():
    model = parse_hw_mlir(_EMPTY)
    assert not model.assertions
    assert not model.clocked_processes
    assert not model.always.body
    assert model.assigns[-1].lhs == "q"


@pytest.mark.parametrize("source", [
    _EMPTY.replace("cf.br ^wait\n  }", 'sim.terminate failure\n    cf.br ^wait\n  }'),
    _EMPTY.replace("  }\n  hw.output", '  ^hidden:\n    sim.terminate failure\n'
                   '    cf.br ^wait\n  }\n  hw.output'),
    _EMPTY.replace("^wake(%previous: i1):\n    cf.br ^wait",
                   "^wake(%previous: i1):\n    cf.br ^wake(%previous : i1)"),
    _EMPTY.replace("llhd.process {", "%result = llhd.process -> i8 {").replace(
        "llhd.wait (%clock : i1)", "llhd.wait yield (%data : i8), (%clock : i1)"),
])
def test_event_loops_with_effects_results_or_extra_control_flow_are_rejected(source):
    with pytest.raises(CirctError):
        parse_hw_mlir(source)


@pytest.mark.parametrize("split", [0, 2])
def test_folded_check_keeps_live_checks_and_pre_nba_state(tmp_path, split):
    compiler = shutil.which("c++")
    if not compiler or not find_bin("circt-verilog"):
        pytest.skip("CIRCT and a C++ compiler are required")
    top = tmp_path / "Folded.sv"
    top.write_text("""
module Folded(input clock, reset, go, bad, ready, write_rd,
              input [4:0] rd, output logic [7:0] q);
  wire needs_write = write_rd & (|rd);
  wire write_valid = go & needs_write & ready;
  always @(posedge clock)
    if (~reset & write_valid != (ready & go & needs_write))
      $fatal(1, "folded check");
  always @(posedge clock) begin
    if (reset) q <= 0;
    else if (go) q <= q + 1;
    if (!reset && bad) $fatal(1, "live check");
  end
endmodule
""")
    directory = tmp_path / "model"
    model = compile_file(top, directory / "dut.h", frontend="circt",
                         assertions=True, split=split)
    assert len(model.assertions) == 1
    main = directory / "check.cpp"
    main.write_text(r'''
#include "dut.h"
#include <stdexcept>
#include <string>
int main() {
  FoldedDut d;
  d.reset=1; d.go=1; d.ready=1; d.write_rd=1; d.rd=3; d.tick();
  d.reset=0;
  for(unsigned i=1;i!=6;++i) {
    d.tick(); d.eval_q(); if(d.q!=i) return 1;
  }
  d.bad=1;
  try { d.tick(); return 2; }
  catch(const std::exception &e) {
    d.eval_q();
    return d.q!=5 || std::string(e.what()).find("live check")==std::string::npos;
  }
}
''')
    binary = directory / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main),
                    *map(str, sorted(directory.glob("dut_*.cpp"))),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
