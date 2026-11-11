"""Clock casts and clock-typed module ports preserve their one-bit value."""
import shutil
import subprocess

import pytest

from flashsim.circt_frontend import CirctError, parse_hw_mlir
from flashsim.emit import emit_cpp
from flashsim.opt import optimize
from flashsim.compile import compile_file
from flashsim.toolchain import find_bin


def test_clock_casts_cross_module_ports_and_observe_input(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module private @Leaf(in %clk : !seq.clock, in %d : i8, out q : i8, out seen : i1) {
      %level = seq.from_clock %clk
      %roundtrip = seq.to_clock %level
      %value = seq.firreg %d clock %roundtrip : i8
      hw.output %value, %level : i8, i1
    }
    hw.module @Top(in %clock : !seq.clock, in %d : i8, out q : i8, out seen : i1) {
      %level = seq.from_clock %clock
      %cast = seq.to_clock %level
      %q0, %seen0 = hw.instance "u" @Leaf(clk: %cast: !seq.clock, d: %d: i8) -> (q: i8, seen: i1)
      hw.output %q0, %seen0 : i8, i1
    }
    """
    model = optimize(parse_hw_mlir(src, top="Top"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  TopDut d;
  d.clock = 0; d.d = 42; d.poke_inputs(); d.eval_seen();
  if (d.seen != 0) return 1;
  d.clock = 1; d.poke_inputs(); d.eval_seen();
  if (d.seen != 1) return 2;
  d.tick(); d.eval_q();
  if (d.q != 42) return 3;
  d.clock = 0; d.poke_inputs(); d.eval_seen();
  return d.seen != 0;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_clock_level_used_only_as_register_data_is_available(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @ClockData(in %clock : !seq.clock, out q : i1) {
      %level = seq.from_clock %clock
      %value = seq.firreg %level clock %clock : i1
      hw.output %value : i1
    }
    """
    model = optimize(parse_hw_mlir(src, top="ClockData"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "clock_data.cpp"
    main.write_text('''#include "dut.h"
int main() {
  ClockDataDut d;
  d.clock = 1; d.tick(); d.eval_q(); if (d.q != 1) return 1;
  d.clock = 0; d.tick(); d.eval_q(); return d.q != 0;
}
''')
    binary = tmp_path / "clock_data"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_abbreviated_signal_element_type_preserves_the_driven_value(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @SignalTop(in %d : i8, out q : i8) {
      %zero = hw.constant 0 : i8
      %delay = llhd.constant_time <0ns, 0d, 1e>
      %wire = llhd.sig name "wire" %zero : <i8>
      llhd.drv %wire, %d after %delay : i8
      %observed = llhd.prb %wire : i8
      hw.output %observed : i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="SignalTop"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "signal.cpp"
    main.write_text('''#include "dut.h"
int main() {
  SignalTopDut d;
  d.d = 17; d.poke_inputs(); d.eval_q(); if (d.q != 17) return 1;
  d.d = 93; d.poke_inputs(); d.eval_q(); return d.q != 93;
}
''')
    binary = tmp_path / "signal"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_uninitialized_signal_with_immediate_assignment_and_conditional_rejection(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @SignalTop(in %d : i8, in %enable : i1, out q : i8) {
      %wire = llhd.sig : <i8>
      llhd.assign %wire, %d : i8
      %observed = llhd.prb %wire : i8
      hw.output %observed : i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="SignalTop"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "immediate.cpp"
    main.write_text('''#include "dut.h"
int main() {
  SignalTopDut d;
  d.d = 17; d.poke_inputs(); d.eval_q(); if (d.q != 17) return 1;
  d.d = 93; d.poke_inputs(); d.eval_q(); return d.q != 93;
}
''')
    binary = tmp_path / "immediate"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    with pytest.raises(CirctError, match="conditional immediate"):
        parse_hw_mlir(src.replace("%wire, %d : i8", "%wire, %d if %enable : i8"), top="SignalTop")


def test_two_state_comb_ops_keep_their_values(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @Binary(in %a : i8, in %b : i8, out q : i8) {
      %x = comb.xor bin %a, %b : i8
      %y = comb.and bin %x, %a : i8
      hw.output %y : i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="Binary"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "binary.cpp"
    main.write_text('''#include "dut.h"
int main() {
  BinaryDut d;
  d.a = 0xf0; d.b = 0xaa; d.poke_inputs(); d.eval_q();
  return d.q != 0x50;
}
''')
    binary = tmp_path / "binary"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_two_field_nonblocking_delay_and_unknown_initializer_rejection(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @State(in %clock : i1, in %d : i8, out q : i8) {
      %zero = hw.constant 0 : i8
      %false = hw.constant false
      %true = hw.constant true
      %delay = llhd.constant_time <0ns, 1d>
      %state = llhd.sig %zero : <i8>
      %oldstate = llhd.prb %state : i8
      %result:2 = llhd.process -> i8, i1 {
        cf.br ^wait(%zero, %false : i8, i1)
      ^wait(%next: i8, %enable: i1):
        llhd.wait yield (%next, %enable : i8, i1), (%clock : i1), ^edge(%clock : i1)
      ^edge(%oldclock: i1):
        %notold = comb.xor bin %oldclock, %true : i1
        %rise = comb.and bin %notold, %clock : i1
        cf.cond_br %rise, ^update, ^wait(%oldstate, %false : i8, i1)
      ^update:
        cf.br ^wait(%d, %true : i8, i1)
      }
      llhd.drv %state, %result#0 after %delay if %result#1 : i8
      hw.output %oldstate : i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="State"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "state.cpp"
    main.write_text('''#include "dut.h"
int main() {
  StateDut d;
  d.d = 9; d.tick(); d.eval_q(); if (d.q != 9) return 1;
  d.d = 31; d.tick(); d.eval_q(); return d.q != 31;
}
''')
    binary = tmp_path / "state"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    with pytest.raises(CirctError, match="unknown LLHD state initializer"):
        parse_hw_mlir(src.replace("llhd.sig %zero : <i8>", "llhd.sig : <i8>"), top="State")


def test_real_circt_clock_path_matches_verilator(tmp_path):
    compiler = shutil.which("c++")
    verilator = shutil.which("verilator")
    circt = find_bin("circt-verilog")
    if not compiler or not verilator or not circt:
        pytest.skip("CIRCT, Verilator and a C++ compiler are required")
    source = tmp_path / "ClockPath.sv"
    source.write_text('''module ClockPath(input clock, input [7:0] d, output reg [7:0] q, output seen);
  assign seen = clock;
  always @(posedge clock) q <= d;
endmodule
''')
    compile_file(source, tmp_path / "dut.h", frontend="circt")
    cpp = tmp_path / "flashsim.cpp"
    cpp.write_text('''#include "dut.h"
#include <cstdio>
int main() {
  ClockPathDut d;
  for (unsigned i = 0; i < 16; ++i) {
    d.clock = 0; d.d = (i * 31) & 255; d.poke_inputs(); d.eval_seen();
    std::printf("%u ", d.seen);
    d.clock = 1; d.tick(); d.eval_seen(); d.eval_q();
    std::printf("%u %u\\n", d.seen, d.q);
  }
}
''')
    flashsim = tmp_path / "flashsim-model"
    subprocess.run([compiler, "-std=c++17", "-O1", str(cpp), "-o", str(flashsim)], check=True)
    ref = tmp_path / "reference.cpp"
    ref.write_text('''#include "VClockPath.h"
#include <cstdio>
int main() {
  VClockPath d;
  for (unsigned i = 0; i < 16; ++i) {
    d.clock = 0; d.d = (i * 31) & 255; d.eval();
    std::printf("%u ", d.seen);
    d.clock = 1; d.eval(); std::printf("%u %u\\n", d.seen, d.q);
  }
}
''')
    subprocess.run([verilator, "--cc", "--exe", "--build", "--CFLAGS", "-std=c++17", "--top-module", "ClockPath",
                    "--Mdir", str(tmp_path / "vobj"), str(source), str(ref)],
                   check=True, stdout=subprocess.DEVNULL)
    assert subprocess.check_output([str(flashsim)]) == subprocess.check_output([str(tmp_path / "vobj/VClockPath")])
