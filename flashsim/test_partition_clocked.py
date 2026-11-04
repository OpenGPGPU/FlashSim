"""Clocked-process partitioning preserves four-state nonblocking behavior."""
import shutil
import subprocess
import sys

import pytest

from flashsim.partition_clocked import partition_clocked_process


def partition(source, groups=128, **kwargs):
    return partition_clocked_process(source, groups, assume_stable_edge_inputs=True, **kwargs)


RTL = """module PartitionDut(input clock, reset, trigger, input [1:0] choose,
 output reg [7:0] a, b, c, d);
  always @(posedge clock) begin
    if (reset) begin
      a <= 0; b <= 0; c <= 0; d <= 0;
    end else begin
      if (trigger) begin
        a <= a + 1; b <= a;
      end else begin
        a <= a + 2; c <= b;
      end
      if (choose[0]) a <= a + 5;
      if (choose[1]) begin b <= c; d <= a; end
      else if (trigger) d <= c;
      d <= d + 1;
    end
  end // always @(posedge)
endmodule
"""


def test_partition_preserves_four_state_ordered_assignments(tmp_path):
    compiler, runner = shutil.which("iverilog"), shutil.which("vvp")
    if not compiler or not runner:
        pytest.skip("Icarus Verilog is unavailable")
    changed, counts = partition(RTL, groups=3)
    assert counts == {"targets": 4, "assignments": 13, "groups": 3}
    tb = tmp_path / "tb.sv"
    tb.write_text("""module tb;
  reg clock=0, reset=1, trigger=0; reg [1:0] choose=0;
  wire [7:0] a,b,c,d; integer i;
  PartitionDut dut(clock,reset,trigger,choose,a,b,c,d);
  initial begin
    #1 clock=1; #1 clock=0; reset=0;
    for (i=0;i<100;i=i+1) begin
      choose=i%4;
      case (i%3) 0: trigger=0; 1: trigger=1; 2: trigger=1'bx; endcase
      #1 clock=1; #1 $display("%b %b %b %b",a,b,c,d); clock=0; #1;
    end
    $finish;
  end
endmodule
""")
    results = []
    for name, source in (("original", RTL), ("partitioned", changed)):
        design = tmp_path / (name + ".sv")
        binary = tmp_path / name
        design.write_text(source)
        subprocess.run([compiler, "-g2012", "-s", "tb", "-o", str(binary),
                        str(tb), str(design)], check=True)
        results.append(subprocess.check_output([runner, str(binary)]))
    assert results[0] == results[1]


@pytest.mark.parametrize("before,after", [
    ("a <= a + 1", "a = a + 1"),
    ("a <= a + 1", "a <= $random"),
    ("a <= a + 1", "a <= mutate(a)"),
    ("if (trigger)", "if (mutate(trigger))"),
    ("d <= d + 1", "d <= d++"),
])
def test_side_effects_and_blocking_writes_are_rejected(before, after):
    with pytest.raises(ValueError):
        partition(RTL.replace(before, after, 1), groups=2)


def test_multiple_drivers_and_ambiguous_initialization_are_rejected():
    elsewhere = RTL.replace("endmodule", "always_comb a = 1;\nendmodule")
    with pytest.raises(ValueError, match="written elsewhere"):
        partition(elsewhere)
    for write in ("a[choose[0]] = 1'b0;", "a.member = 1;", "a++;", "++ a;", "a += 1;",
                  "{b,{a,c}} = 24'b0;"):
        with pytest.raises(ValueError, match="written elsewhere"):
            partition(RTL.replace("endmodule", f"always_comb {write}\nendmodule"))
    with pytest.raises(ValueError, match="outside-process"):
        partition(RTL.replace("endmodule", "initial $deposit(a, 1);\nendmodule"))
    guarded = RTL.replace("endmodule", "`ifdef RANDOMIZE_REG_INIT\ninitial a = 1;\n`endif\nendmodule")
    with pytest.raises(ValueError, match="written elsewhere"):
        partition(guarded)
    changed, _ = partition(guarded, groups=2, assume_default_initialization=True)
    assert guarded.split("  always @(posedge clock) begin")[0] in changed
    assert "initial a = 1;" in changed
    with pytest.raises(ValueError, match="written elsewhere"):
        partition(elsewhere, assume_default_initialization=True)


def test_invalid_process_and_group_counts_are_rejected():
    for groups in (0, 257):
        with pytest.raises(ValueError, match="groups"):
            partition(RTL, groups)
    with pytest.raises(ValueError, match="exactly one"):
        partition(RTL.replace("  end // always @(posedge)", "  end"))
    with pytest.raises(ValueError, match="clock-edge inputs"):
        partition_clocked_process(RTL)


def test_instance_port_requires_explicit_driver_assumption():
    connected = RTL.replace("endmodule", "SomeModule u(.port({a, b}));\nendmodule")
    with pytest.raises(ValueError, match="instance port"):
        partition(connected)
    changed, _ = partition(connected, assume_no_instance_drivers=True)
    assert "SomeModule u(.port({a, b}));" in changed


def test_cli_writes_a_separate_output_and_preserves_source(tmp_path):
    original, output = tmp_path / "input.sv", tmp_path / "output.sv"
    original.write_text(RTL)
    command = [sys.executable, "-m", "flashsim.partition_clocked", str(original),
               "--assume-stable-edge-inputs"]
    result = subprocess.run([*command, str(original)], capture_output=True, text=True)
    assert result.returncode != 0
    assert original.read_text() == RTL
    subprocess.run([*command, str(output), "--groups", "3"], check=True)
    assert original.read_text() == RTL
    assert output.read_text().count("always @(posedge clock)") == 3
    original.write_text(RTL.replace("a <= a + 1", "a = a + 1"))
    prior = output.read_text()
    result = subprocess.run([*command, str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert output.read_text() == prior
