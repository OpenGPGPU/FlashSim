"""CIRCT aggregate constants preserve every element and reject ambiguity."""
import shutil
import subprocess

import pytest

from flashsim.circt_frontend import CirctError, parse_hw_mlir
from flashsim.compile import compile_file
from flashsim.emit import emit_cpp
from flashsim.opt import optimize
from flashsim.toolchain import find_bin


def _compile_model(directory, source, *, optimized=True):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    model = parse_hw_mlir(source, top="Aggregate")
    if optimized:
        model = optimize(model)
    (directory / "dut.h").write_text(emit_cpp(model))
    return compiler


@pytest.mark.parametrize("width", [1, 8, 64, 128, 129, 256])
@pytest.mark.parametrize("optimized", [False, True])
def test_aggregate_array_index_order_and_wide_values(tmp_path, width, optimized):
    if width == 1:
        literals = ["true", "false", "false", "false"]
        values = [1, 0, 0, 0]
    elif width == 8:
        literals = ["000 : i8", "0xff : i8", "-125 : i8", "0x2a : i8"]
        values = [0, 255, -125, 42]
    else:
        values = [0, -1, (1 << (width - 1)) + 3, 0x2a]
        literals = [f"{value} : i{width}" for value in values]
    src = f"""
    hw.module @Aggregate(in %clock : i1, in %index : i2, out value : i{width}) {{
      %arr = hw.aggregate_constant [{', '.join(literals)}] : !hw.array<4xi{width}>
      %selected = hw.array_get %arr[%index] : !hw.array<4xi{width}>, i2
      hw.output %selected : i{width}
    }}
    """
    compiler = _compile_model(tmp_path, src, optimized=optimized)
    steps = []
    for index, value in enumerate(reversed(values)):
        value &= (1 << width) - 1
        steps.append(f"d.index={index}; d.poke_inputs(); d.eval_value();")
        if width > 128:
            for limb in range((width + 63)//64):
                word = (value >> (64 * limb)) & ((1 << 64) - 1)
                steps.append(f"if (d.value[{limb}] != {word}ull) return 1;")
        elif width > 64:
            lo, hi = value & ((1 << 64) - 1), value >> 64
            steps.append(f"if ((uint64_t)d.value != {lo}ull || (uint64_t)(d.value >> 64) != {hi}ull) return 2;")
        else:
            steps.append(f"if (d.value != {value}ull) return 3;")
    (tmp_path / "check.cpp").write_text('#include "dut.h"\nint main() { AggregateDut d;\n' +
                                         "\n".join(steps) + "\n}\n")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    str(tmp_path / "check.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_large_dense_aggregate_has_no_recursive_expression_chain(tmp_path):
    count = 1025
    values = [i % 251 for i in range(count)]
    literals = ", ".join(f"{value} : i8" for value in values)
    src = f"""
    hw.module @Aggregate(in %clock : i1, in %index : i11, out value : i8) {{
      %arr = hw.aggregate_constant [{literals}] : !hw.array<{count}xi8>
      %selected = hw.array_get %arr[%index] : !hw.array<{count}xi8>, i11
      hw.output %selected : i8
    }}
    """
    compiler = _compile_model(tmp_path, src)
    cases = [0, 1, 250, 512, 1024]
    steps = [f"d.index={i}; d.poke_inputs(); d.eval_value(); "
             f"if (d.value != {values[-1-i]}u) return 1;" for i in cases]
    (tmp_path / "check.cpp").write_text('#include "dut.h"\nint main() { AggregateDut d;\n' +
                                         "\n".join(steps) + "\n}\n")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(tmp_path / "check.cpp"),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("literal,ty", [
    ("1 : i8", "!hw.array<2xi8>"),
    ("1 : i16, 0 : i8", "!hw.array<2xi8>"),
    ("true, false", "!hw.array<2xi8>"),
    ("true : i1, false", "!hw.array<2xi1>"),
    ("256 : i8, 0 : i8", "!hw.array<2xi8>"),
    ("-129 : i8, 0 : i8", "!hw.array<2xi8>"),
    ("2 : i1, false", "!hw.array<2xi1>"),
    ("1, 0", "!hw.array<2xi8>"),
    ("[1 : i8], 0 : i8", "!hw.array<2xi8>"),
    ("poison, 0 : i8", "!hw.array<2xi8>"),
    ("", "!hw.array<0xi8>"),
])
def test_aggregate_rejects_invalid_or_unsupported_elements(literal, ty):
    src = f"""
    hw.module @Aggregate(in %clock : i1, out value : i8) {{
      %arr = hw.aggregate_constant [{literal}] : {ty}
      %zero = hw.constant 0 : i1
      %selected = hw.array_get %arr[%zero] : !hw.array<2xi8>, i1
      hw.output %selected : i8
    }}
    """
    with pytest.raises(CirctError):
        parse_hw_mlir(src, top="Aggregate")


def test_circt_reset_memory_matches_icarus_and_checks_assertion(tmp_path):
    compiler, iverilog, vvp = shutil.which("c++"), shutil.which("iverilog"), shutil.which("vvp")
    if not all((compiler, iverilog, vvp, find_bin("circt-verilog"))):
        pytest.skip("CIRCT, Icarus and a C++ compiler are required")
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    top = rtl / "Top.sv"
    top.write_text('''module Top(input logic clock, reset, write,
 input logic [1:0] index, input logic [7:0] data, output logic [7:0] q);
 logic [7:0] words [0:3];
 always_ff @(posedge clock) begin
   if (reset) for (int i=0;i<4;i=i+1) words[i] <= i*3;
   else if (write) words[index] <= data;
 end
 assign q = words[index];
 always @(posedge clock) if (!reset && write && data==8'hff)
   $fatal(1, "bad memory write");
endmodule
''')
    tb = tmp_path / "tb.sv"
    tb.write_text('''module tb;
 reg clock=0, reset=1, write=0; reg [1:0] index=0; reg [7:0] data=0; wire [7:0] q;
 Top dut(clock,reset,write,index,data,q);
 initial begin
   #1 clock=1; #1 clock=0; reset=0;
   for (integer i=0;i<4;i=i+1) begin index=i; #1 $display("%0d",q); end
   for (integer i=0;i<4;i=i+1) begin index=i; data=20+i; write=1; #1 clock=1; #1 clock=0; end
   write=0;
   for (integer i=0;i<4;i=i+1) begin index=i; #1 $display("%0d",q); end
   $finish;
 end
endmodule
''')
    reference = tmp_path / "reference"
    subprocess.run([iverilog,"-g2012","-s","tb","-o",str(reference),str(top),str(tb)],
                   check=True, capture_output=True, text=True)
    expected = [line for line in subprocess.check_output([vvp,str(reference)], text=True).splitlines()
                if line.isdecimal()]
    assert len(expected) == 8
    model = compile_file(top, tmp_path / "dut.h", frontend="circt", assertions=True)
    assert len(model.assertions) == 1
    (tmp_path / "check.cpp").write_text(r'''#include "dut.h"
#include <iostream>
#include <stdexcept>
#include <string>
int main() {
  TopDut d; d.reset=1; d.write=0; d.tick(); d.reset=0;
  for (unsigned i=0;i<4;i++) { d.index=i; d.poke_inputs(); d.eval_q(); std::cout << unsigned(d.q) << '\n'; }
  for (unsigned i=0;i<4;i++) { d.index=i; d.data=20+i; d.write=1; d.tick(); }
  d.write=0;
  for (unsigned i=0;i<4;i++) { d.index=i; d.poke_inputs(); d.eval_q(); std::cout << unsigned(d.q) << '\n'; }
  try { d.write=1; d.data=255; d.tick(); }
  catch (const std::runtime_error &e) { return std::string(e.what()).find("bad memory write") == std::string::npos; }
  return 1;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler,"-std=c++17","-O1","-fsanitize=address,undefined",
                    str(tmp_path / "check.cpp"),"-o",str(binary)], check=True)
    assert subprocess.check_output([str(binary)], text=True).splitlines() == expected
