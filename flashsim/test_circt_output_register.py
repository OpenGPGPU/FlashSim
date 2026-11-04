"""A registered output may share its SSA name with its port."""

import shutil
import subprocess

import pytest

from flashsim.circt_frontend import parse_hw_mlir
from flashsim.emit import emit_cpp
from flashsim.opt import optimize


def test_register_named_after_output_preserves_clocked_behavior(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = """
    hw.module @RegisteredOutput(in %clock : i1, in %d : i8, out dataout : i8) {
      %clk = seq.to_clock %clock
      %dataout = seq.firreg %d clock %clk : i8
      hw.output %dataout : i8
    }
    """
    model = optimize(parse_hw_mlir(src, top="RegisteredOutput"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  RegisteredOutputDut d;
  d.d = 42; d.tick(); d.eval_dataout();
  if (d.dataout != 42) return 1;
  d.d = 7; d.poke_inputs(); d.eval_dataout();
  if (d.dataout != 42) return 2;
  d.tick(); d.eval_dataout();
  return d.dataout != 7;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("width", [32, 64])
def test_registered_sram_read_holds_and_follows_writes(tmp_path, width):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    # This is the CIRCT-lowered form of the H6 characterized SRAM interface:
    # a selected write wins over a read, and unselected reads hold the output.
    src = f"""
    hw.module @Sram(in %clk : i1, in %address : i8, in %wd : i{width}, in %banksel : i1, in %read : i1, in %write : i1, out dataout : i{width}) {{
      %enable = comb.and %write, %banksel : i1
      %updated = hw.array_inject %mem[%address], %wd : !hw.array<256xi{width}>, i8
      %next = comb.mux %enable, %updated, %mem : !hw.array<256xi{width}>
      %c = seq.to_clock %clk
      %mem = seq.firreg %next clock %c : !hw.array<256xi{width}>
      %value = hw.array_get %mem[%address] : !hw.array<256xi{width}>, i8
      %true = hw.constant true
      %no_write = comb.xor %enable, %true : i1
      %read_selected = comb.and %read, %banksel, %no_write : i1
      %out_next = comb.mux %read_selected, %value, %dataout : i{width}
      %dataout = seq.firreg %out_next clock %c : i{width}
      hw.output %dataout : i{width}
    }}
    """
    model = optimize(parse_hw_mlir(src, top="Sram"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  SramDut d;
  d.banksel = 1; d.write = 1; d.address = 2; d.wd = 123; d.tick();
  d.write = 0; d.read = 1; d.tick(); d.eval_dataout();
  if (d.dataout != 123) return 1;
  d.read = 0; d.address = 3; d.tick(); d.eval_dataout();
  if (d.dataout != 123) return 2;
  d.read = 1; d.write = 1; d.address = 2; d.wd = 77; d.tick(); d.eval_dataout();
  if (d.dataout != 123) return 3;
  d.write = 0; d.tick(); d.eval_dataout();
  return d.dataout != 77;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
