"""Runtime checks for bound assertions, pre-NBA sampling and explicit rejection."""
import os
import shutil
import subprocess

import pytest

from flashsim.circt_frontend import CirctError, circt_verilog_to_mlir, parse_hw_mlir
from flashsim.compile import compile_file, compile_verilog
from flashsim.emit import emit_cpp
from flashsim.ir import (AlwaysFF, Assign, BinOp, ClockedAssertion, Const, Id,
                         Module, NbAssign, Signal)
from flashsim.opt import optimize
from flashsim.toolchain import find_bin


def _tools():
    if find_bin("circt-verilog") is None or shutil.which("c++") is None:
        pytest.skip("CIRCT and a C++ compiler are required")


def _design(tmp_path):
    root = tmp_path / "rtl"
    checks = root / "verification/assert"
    checks.mkdir(parents=True)
    top = root / "Top.sv"
    top.write_text("""
module Leaf(input logic clock, reset, go, input logic [7:0] expected,
            output logic [7:0] q);
  always_ff @(posedge clock)
    if (reset) q <= 0;
    else if (go) q <= q + 1;
endmodule
module Top(input logic clock, reset, go,
           input logic [7:0] expected_left, expected_right,
           output logic [7:0] left_q, right_q);
  Leaf left(clock, reset, go, expected_left, left_q);
  Leaf right(clock, reset, go, expected_right, right_q);
endmodule
""")
    check = checks / "Leaf_Verification_Assert.sv"
    check.write_text(r'''
module Leaf_Verification_Assert;
  always @(posedge Leaf.clock) begin
    if (!Leaf.reset && Leaf.go && Leaf.q != Leaf.expected) begin
      $error("old count before NBA {check} \"quoted\"\n");
      $fatal;
    end
    if (!Leaf.reset && !Leaf.clock)
      $fatal(1, "clock must be high at posedge");
  end
endmodule
''')
    (checks / "layers-Leaf-Verification-Assert.sv").write_text(
        "`ifndef leaf_assert_bind\n`define leaf_assert_bind\n"
        "bind Leaf Leaf_Verification_Assert verification_assert ();\n`endif\n")
    (checks / "layers-Top-Verification-Assert.sv").write_text(
        '`include "layers-Leaf-Verification-Assert.sv"\n')
    (root / "filelist.f").write_text(
        "Top.sv\nverification/assert/Leaf_Verification_Assert.sv\n")
    return top, check


_MAIN = r'''
#include "dut.h"
#include <iostream>
#include <string>
int main(int argc, char **argv) {
  try {
    TopDut d;
    d.reset = 1; d.go = 1; d.expected_left = 99; d.expected_right = 99;
    d.tick();
    d.reset = 0;
    for (unsigned i = 0; i != 8; ++i) {
      d.expected_left = i; d.expected_right = i; d.tick();
      d.eval_left_q(); d.eval_right_q();
      if (d.left_q != i + 1 || d.right_q != i + 1) return 2;
    }
    d.go = 0; d.expected_left = 55; d.expected_right = 66;
    for (unsigned i = 0; i != 4; ++i) d.tick();
    d.reset = 1; d.go = 1; d.tick();
    d.reset = 0; d.expected_left = 0; d.expected_right = 0; d.tick();
    d.expected_left = 1; d.expected_right = 1;
    const std::string mode = argc > 1 ? argv[1] : "pass";
    if (mode == "left") d.expected_left = 44;
    if (mode == "right") d.expected_right = 44;
    d.tick();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what();
    return 42;
  }
}
'''


def _native(directory, main=_MAIN):
    (directory / "check.cpp").write_text(main)
    binary = directory / "check"
    sources = [directory / "check.cpp", *sorted(directory.glob("dut_*.cpp"))]
    subprocess.run([shutil.which("c++"), "-std=c++17", "-O1", *map(str, sources),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("split", [0, 2])
def test_bound_checks_sample_before_nba_and_attribute_both_instances(tmp_path, split):
    _tools()
    top, _ = _design(tmp_path)
    directory = tmp_path / "model"
    mod = compile_file(top, directory / "dut.h", frontend="circt",
                       assertions=True, split=split)
    assert len(mod.assertions) == 4
    binary = _native(directory)
    passing = subprocess.run([str(binary), "pass"], capture_output=True, text=True)
    assert passing.returncode == 0, passing.stderr
    for instance in ["left", "right"]:
        failing = subprocess.run([str(binary), instance], capture_output=True, text=True)
        assert failing.returncode == 42, failing.stderr
        assert instance + ".verification_assert" in failing.stderr
        assert 'old count before NBA {check} "quoted"' in failing.stderr


def test_cache_cannot_reuse_assertion_free_or_changed_checks(tmp_path):
    _tools()
    top, check = _design(tmp_path)
    header = tmp_path / "model/dut.h"
    assert not compile_file(top, header, frontend="circt").assertions
    original = header.read_bytes()
    assert len(compile_file(top, header, frontend="circt", assertions=True).assertions) == 4
    assert not compile_file(top, header, frontend="circt").assertions
    assert header.read_bytes() == original
    compile_file(top, header, frontend="circt", assertions=True)
    check.write_text(check.read_text().replace("Leaf.q != Leaf.expected",
                                              "Leaf.q == Leaf.expected"))
    compile_file(top, header, frontend="circt", assertions=True)
    failing = subprocess.run([str(_native(header.parent)), "pass"],
                             capture_output=True, text=True)
    assert failing.returncode == 42
    assert "old count before NBA" in failing.stderr


def test_included_assertion_header_changes_invalidate_cache(tmp_path):
    _tools()
    top, check = _design(tmp_path)
    config = check.parent.parent / "diagnostic.svh"
    config.write_text(chr(96) + 'define CHECK_TEXT "included diagnostic before NBA"\n')
    check.write_text(chr(96) + 'include "diagnostic.svh"\n' +
        check.read_text().replace(
            r'"old count before NBA {check} \"quoted\"\n"', chr(96)+"CHECK_TEXT"))
    header = tmp_path / "model/dut.h"
    compile_file(top, header, frontend="circt", assertions=True)
    before = config.stat().st_mtime
    config.write_text(chr(96) + 'define CHECK_TEXT "changed included diagnostic"\n')
    # Hash the include closure rather than assuming a newer source mtime.
    os.utime(config, (before, before))
    compile_file(top, header, frontend="circt", assertions=True)
    failing = subprocess.run([str(_native(header.parent)), "right"],
                             capture_output=True, text=True)
    assert failing.returncode == 42 and "changed included diagnostic" in failing.stderr


def test_hierarchy_result_does_not_alias_private_child_register(tmp_path):
    _tools()
    top, _ = _design(tmp_path)
    directory = tmp_path / "model"
    mod = compile_file(top, directory / "dut.h", frontend="circt")
    assert not mod.assertions
    passing = subprocess.run([str(_native(directory)), "pass"],
                             capture_output=True, text=True)
    assert passing.returncode == 0, passing.stderr


def test_bound_checks_resolve_typed_child_instance_results(tmp_path):
    _tools()
    top, check = _design(tmp_path)
    check.write_text("""
module Compare(input [7:0] observed, expected,
               output mismatch, output [7:0] sampled);
  assign mismatch = observed != expected;
  assign sampled = observed;
endmodule
module Leaf_Verification_Assert;
  wire mismatch;
  wire [7:0] sampled;
  Compare compare(Leaf.q, Leaf.expected, mismatch, sampled);
  always @(posedge Leaf.clock)
    if (!Leaf.reset && Leaf.go && (mismatch || sampled != Leaf.expected))
      $fatal(1, "child comparator before NBA");
endmodule
""")
    directory = tmp_path / "model"
    mod = compile_file(top, directory / "dut.h", frontend="circt", assertions=True)
    assert len(mod.assertions) == 2
    assert 'hw.instance "compare"' in (directory / "hw.mlir").read_text()
    binary = _native(directory)
    passing = subprocess.run([str(binary), "pass"], capture_output=True, text=True)
    assert passing.returncode == 0, passing.stderr
    for instance in ["left", "right"]:
        failing = subprocess.run([str(binary), instance], capture_output=True, text=True)
        assert failing.returncode == 42, failing.stderr
        assert instance + ".verification_assert" in failing.stderr
        assert "child comparator before NBA" in failing.stderr


@pytest.mark.parametrize("event", ["negedge clock", "posedge clock or posedge reset"])
def test_other_events_fail_instead_of_becoming_posedge_checks(tmp_path, event):
    _tools()
    top = tmp_path / "Unsupported.sv"
    top.write_text(f"""
module Unsupported(input clock, reset, good);
  always @({event}) if (!reset && !good) $fatal(1, "bad");
endmodule
""")
    with pytest.raises(CirctError, match="posedge|exactly one clock|event wait"):
        compile_file(top, tmp_path / "model/dut.h", frontend="circt", assertions=True)


def test_nonfatal_side_effects_are_not_silently_omitted(tmp_path):
    _tools()
    top = tmp_path / "Print.sv"
    top.write_text("""
module Print(input clock, good);
  always @(posedge clock) begin
    $display("a passing side effect");
    if (!good) $fatal(1, "bad");
  end
endmodule
""")
    with pytest.raises(CirctError, match="nonfatal procedural printing"):
        compile_file(top, tmp_path / "model/dut.h", frontend="circt", assertions=True)


def test_native_frontend_cannot_accept_bound_assertions():
    with pytest.raises(CirctError, match="require the CIRCT frontend"):
        compile_verilog("module T(input clk); endmodule", frontend="native",
                        assertions=True)


def test_assertion_only_cone_survives_optimization_and_large_nba_demand(tmp_path):
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler unavailable")
    signals = {name: Signal(name, width, kind) for name, width, kind in [
        ("clk", 1, "input"), ("data", 8, "input"), ("expected", 8, "input"),
        ("bad", 1, "wire"), ("value", 8, "wire")]}
    body = []
    # Exercise the large-model cache/demand path with many independently
    # updated registers while a separate assertion observes the same inputs.
    for index in range(260):
        name = "r" + str(index)
        signals[name] = Signal(name, 8, "reg")
        body.append(NbAssign(name, Id("value")))
    mod = Module("Demand", signals,
        [Assign("value", BinOp("+", Id("data"), Const(1, 8))),
         Assign("bad", BinOp("!=", Id("data"), Id("expected")))],
        AlwaysFF("clk", body), ["clk", "data", "expected"],
        assertions=[ClockedAssertion(Id("clk"), Id("bad"), "data mismatch", "observer")])
    (tmp_path / "dut.h").write_text(emit_cpp(optimize(mod)))
    main = r'''
#include "dut.h"
#include <iostream>
int main(int argc, char **) {
  try {
    DemandDut d;
    for (unsigned i = 0; i != 20; ++i) {
      d.data = i; d.expected = i; d.tick();
      if (d.r0 != i+1 || d.r259 != i+1) return 2;
    }
    if (argc > 1) { d.expected = 99; d.tick(); }
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what(); return 42; }
}
'''
    binary = _native(tmp_path, main)
    assert subprocess.run([str(binary)]).returncode == 0
    failing = subprocess.run([str(binary), "fail"], capture_output=True, text=True)
    assert failing.returncode == 42 and "data mismatch" in failing.stderr


def test_clocked_state_and_checks_merge_many_independent_nba_paths(tmp_path):
    _tools()
    count = 24
    declarations = "\n".join(f"logic [7:0] r{i}=0;" for i in range(count))
    reset = " ".join(f"r{i} <= 0;" for i in range(count))
    updates = "\n".join(f"if (enable[{i}]) r{i} <= data;" for i in range(count))
    checksum = " ^ ".join(f"r{i}" for i in range(count))
    top = tmp_path / "Capture.sv"
    top.write_text(f"""
module Capture(input clock, reset, check, input [23:0] enable,
               input [7:0] data, expected,
               output [7:0] first, last, checksum);
  {declarations}
  assign first=r0; assign last=r23; assign checksum={checksum};
  always @(posedge clock) begin
    if (!reset && check && checksum != expected)
      $fatal(1, "snapshot before NBA");
    if (reset) begin {reset} end
    else begin {updates} end
  end
endmodule
""")
    directory = tmp_path / "model"
    model = compile_file(top,directory/"dut.h",frontend="circt",assertions=True)
    assert model.assertions and model.clocked_processes
    main = r'''
#include "dut.h"
#include <array>
#include <iostream>
int main(int argc,char **) {
  CaptureDut d;
  std::array<unsigned,24> expected{};
  auto sum=[&]() { unsigned value=0; for(auto x:expected) value^=x; return value; };
  try {
    unsigned random=0x9e3779b9u;
    for(unsigned tick=0;tick!=40;++tick) {
      d.reset=tick==0 || tick==17; d.check=1; d.expected=sum();
      random^=random<<13; random^=random>>17; random^=random<<5;
      d.enable=random&0xffffffu; d.data=(tick*37u)&255u;
      d.tick();
      for(unsigned i=0;i!=24;++i)
        if(d.reset) expected[i]=0;
        else if(d.enable&(1u<<i)) expected[i]=d.data;
      d.eval_first(); d.eval_last(); d.eval_checksum();
      if(d.first!=expected.front() || d.last!=expected.back() || d.checksum!=sum()) return 2;
    }
    if(argc>1) { d.expected=sum()^1; d.tick(); }
    return 0;
  } catch(const std::exception &error) {
    d.eval_first(); d.eval_last(); d.eval_checksum();
    if(d.first!=expected.front() || d.last!=expected.back() || d.checksum!=sum()) return 3;
    std::cerr<<error.what();
    return 42;
  }
}
'''
    binary = _native(directory,main)
    passing = subprocess.run([str(binary)],capture_output=True,text=True)
    assert passing.returncode==0,passing.stderr
    failing = subprocess.run([str(binary),"fail"],capture_output=True,text=True)
    assert failing.returncode==42 and "snapshot before NBA" in failing.stderr


def test_long_check_chain_flattens_and_shards_with_literal_braces(tmp_path):
    _tools()
    checks = "\n".join(
        f'if (bad[{i}]) $fatal(1,"unmatched }} check {i}");' for i in range(1024))
    top = tmp_path / "DeepCheck.sv"
    top.write_text(f"""
module Observer(input clock, input [1023:0] bad);
  always @(posedge clock) begin
    {checks}
  end
endmodule
module DeepCheck(input clock, input [1023:0] bad);
  Observer leaf(clock,bad);
endmodule
""")
    directory = tmp_path / "model"
    model = compile_file(top,directory/"dut.h",frontend="circt",assertions=True,split=2)
    assert len(model.assertions)==1024
    main = r'''
#include "dut.h"
#include <iostream>
#include <string>
int main(int argc,char **argv) {
  try {
    DeepCheckDut d;
    d.tick();
    if(argc>1) {
      if(std::string(argv[1])=="first") d.bad[0]=1;
      else d.bad[15]=1ull<<63;
      d.tick();
    }
    return 0;
  } catch(const std::exception &e) { std::cerr<<e.what(); return 42; }
}
'''
    binary = _native(directory,main)
    assert subprocess.run([str(binary)]).returncode==0
    for mode,index in [("first",0),("last",1023)]:
        failing = subprocess.run([str(binary),mode],capture_output=True,text=True)
        assert failing.returncode==42,failing.stderr
        assert f"unmatched }} check {index}" in failing.stderr
        assert "leaf.Observer.process" in failing.stderr


@pytest.mark.parametrize("count", [8, 2048])
def test_bound_combinational_case_keeps_every_reference_arm(tmp_path, count):
    _tools()
    width = (count-1).bit_length()
    ports = ", ".join("b"+str(i) for i in range(count))
    cases = "\n".join(f"{width}'d{i}: selected=Select.b{i};" for i in range(count))
    top = tmp_path/"Select.sv"
    top.write_text(f"""
module Select(input clock, go, expected, input [{width-1}:0] index, input {ports});
endmodule
module Select_Verification_Assert;
  logic selected;
  always_comb case (Select.index)
    {cases}
    default: selected=0;
  endcase
  always @(posedge Select.clock)
    if (Select.go && selected!=Select.expected) $fatal(1,"selected reference mismatch");
endmodule
bind Select Select_Verification_Assert verification_assert ();
""")
    directory = tmp_path/"model"
    compile_file(top,directory/"dut.h",frontend="circt",assertions=True,split=2)
    assert "llhd.combinational" in (directory/"hw.mlir").read_text()
    main = r'''
#include "dut.h"
#include <iostream>
int main(int argc,char **) {
  try {
    SelectDut d;
    auto bit=[](unsigned index,unsigned seed) {
      unsigned value=(index+1)*0x9e3779b9u+seed;
      value^=value>>16; value*=0x85ebca6bu; value^=value>>13;
      return value&1;
    };
    for(unsigned seed=1;seed!=3;++seed) {
      // SET_BITS
      for(unsigned i=0;i!=SELECT_COUNT;++i) {
        d.index=i; d.go=1; d.expected=bit(i,seed); d.tick();
      }
    }
    if(argc>1) { d.expected^=1; d.tick(); }
    return 0;
  } catch(const std::exception &e) { std::cerr<<e.what(); return 42; }
}
'''
    main = main.replace("// SET_BITS", "\n".join(
        f"d.b{i}=bit({i},seed);" for i in range(count))).replace("SELECT_COUNT",str(count))
    binary = _native(directory,main)
    passing = subprocess.run([str(binary)],capture_output=True,text=True)
    assert passing.returncode==0,passing.stderr
    failing = subprocess.run([str(binary),"fail"],capture_output=True,text=True)
    assert failing.returncode==42 and "selected reference mismatch" in failing.stderr


def test_wide_assertion_compares_every_limb(tmp_path):
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler unavailable")
    signals = {name:Signal(name,width,"input") for name,width in [
        ("clk",1),("data",256),("expected",256)]}
    mod = Module("WideCheck",signals,[],AlwaysFF("clk",[]),list(signals),
        assertions=[ClockedAssertion(Id("clk"),BinOp("!=",Id("data"),Id("expected")),
                                      "wide mismatch","observer")])
    (tmp_path/"dut.h").write_text(emit_cpp(optimize(mod)))
    main = r'''
#include "dut.h"
#include <iostream>
int main(int argc,char **) {
  try {
    WideCheckDut d;
    for(unsigned i=0;i!=4;++i) d.data[i]=d.expected[i]=(1ull<<63)|(17ull*i);
    d.tick();
    if(argc>1) { d.data[3]^=1ull<<61; d.tick(); }
    return 0;
  } catch(const std::exception &e) { std::cerr<<e.what(); return 42; }
}
'''
    binary = _native(tmp_path,main)
    assert subprocess.run([str(binary)]).returncode==0
    failing = subprocess.run([str(binary),"fail"],capture_output=True,text=True)
    assert failing.returncode==42 and "wide mismatch" in failing.stderr
