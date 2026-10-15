"""Scalar expressions embedded in limb arithmetic must be zero extended."""
import shutil
import subprocess

import pytest

from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Assign, BinOp, Const, Id, MemRead, Module, Signal, Ternary


@pytest.mark.parametrize("compare_width", [32, 256])
@pytest.mark.parametrize("op", ["|", "&", "^", "+", "-", "*", "/", "%", "==", "!=", "<", ">="])
def test_wide_operations_with_nested_scalar_comparison(tmp_path, op, compare_width):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    result_width = 1 if op in ["==", "!=", "<", ">="] else 256
    signals = {name: Signal(name, width, kind) for name, width, kind in [
        ("clock", 1, "input"), ("wide", 256, "input"),
        ("a", compare_width, "input"), ("b", compare_width, "input"), ("enable", 1, "input"),
        ("result", result_width, "output")]}
    scalar = Ternary(Id("enable"), BinOp("!=", Id("a"), Id("b")), Const(0, 1))
    expr = BinOp(op, Id("wide"), scalar)
    mod = Module(name="WideScalar", signals=signals,
                 assigns=[Assign("result", expr)], always=AlwaysFF("clock", []),
                 ports=list(signals))
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    checks = []
    mask = (1 << 256) - 1
    # Zero, limb carries/borrows, upper limbs, and matching/nonmatching inputs.
    for value in [0, 1, (1 << 64) - 1, 1 << 192, mask]:
        for enabled, different in [(0, 1), (1, 0), (1, 1)]:
            scalar_value = int(enabled and different)
            if op in ["/", "%"] and scalar_value == 0:
                continue
            expected = {"|": lambda: value | scalar_value,
                        "&": lambda: value & scalar_value,
                        "^": lambda: value ^ scalar_value,
                        "+": lambda: value + scalar_value,
                        "-": lambda: value - scalar_value,
                        "*": lambda: value * scalar_value,
                        "/": lambda: value // scalar_value,
                        "%": lambda: value % scalar_value,
                        "==": lambda: value == scalar_value,
                        "!=": lambda: value != scalar_value,
                        "<": lambda: value < scalar_value,
                        ">=": lambda: value >= scalar_value}[op]()
            expected = int(expected) & mask
            checks += [f"d.wide[{i}]=0x{(value >> (64*i)) & ((1<<64)-1):x}ull;" for i in range(4)]
            if compare_width == 32:
                checks += [f"d.a=7;d.b={8 if different else 7};"]
            else:
                checks += [f"d.a[{i}]=0;d.b[{i}]=0;" for i in range(3)]
                checks += [f"d.a[3]=7;d.b[3]={8 if different else 7};"]
            checks += [f"d.enable={enabled};d.poke_inputs();d.eval_result();"]
            if result_width == 1:
                checks += [f"if(d.result!={expected})return 1;"]
            else:
                checks += [f"if(d.result[{i}]!=0x{(expected >> (64*i)) & ((1<<64)-1):x}ull)return {i+1};" for i in range(4)]
    main = tmp_path / "check.cpp"
    main.write_text('#include "dut.h"\nint main(){WideScalarDut d;\n' + "\n".join(checks) + "\n}\n")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=address", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_zero_extension_does_not_read_the_next_memory_entry(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    signals = {"clock": Signal("clock", 1, "input"),
               "wide": Signal("wide", 256, "input"),
               "m": Signal("m", 192, "mem", depth=2),
               "result": Signal("result", 256, "output")}
    mod = Module(name="WideMemory", signals=signals,
                 assigns=[Assign("result", BinOp("|", Id("wide"), MemRead("m", Const(0))))],
                 always=AlwaysFF("clock", []), ports=["clock", "wide", "result"])
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    main = tmp_path / "check.cpp"
    main.write_text('''#include "dut.h"
int main() {
  WideMemoryDut d;
  d.wide[3]=1;d.m[0][0]=3;d.m[0][1]=5;d.m[0][2]=7;
  d.m[1][0]=9; // Poison the limb after the selected 192-bit entry.
  d.poke_inputs();d.eval_result();
  return d.result[0]!=3 || d.result[1]!=5 || d.result[2]!=7 || d.result[3]!=1;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=address", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
