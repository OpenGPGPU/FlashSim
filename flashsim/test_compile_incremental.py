"""Regeneration should retain cacheable files and remove obsolete shards."""
import os

from flashsim.compile import compile_file


def test_regeneration_preserves_identical_outputs_and_drops_stale_shards(tmp_path):
    source = tmp_path / "Gate.sv"
    verilog = """module Gate(input clock, input a, input b, output y);
reg q;
assign y = q;
always @(posedge clock) q <= a ^ b;
endmodule
"""
    source.write_text(verilog)
    header = tmp_path / "dut.h"
    compile_file(source, header, frontend="native", split=2)
    paths = [header, tmp_path / "dut_0.cpp", tmp_path / "dut_1.cpp"]
    old = 1590000000000000000
    for path in paths:
        os.utime(path, ns=(old,old))
    obsolete = tmp_path / "dut_obsolete.cpp"
    obsolete.write_text("// old generation\n")
    compile_file(source, header, frontend="native", split=2)
    assert all(path.stat().st_mtime_ns == old for path in paths)
    assert not obsolete.exists()
    before = header.read_bytes()
    source.write_text(verilog.replace("a ^ b", "a | b"))
    compile_file(source, header, frontend="native", split=2)
    assert header.read_bytes() != before
    assert header.stat().st_mtime_ns != old
    compile_file(source, header, frontend="native", split=0)
    assert not list(tmp_path.glob("dut_*.cpp"))
