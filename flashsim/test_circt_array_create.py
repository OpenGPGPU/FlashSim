"""CIRCT array construction retains element order and live input updates."""

import shutil
import subprocess

import pytest

from flashsim.circt_frontend import parse_hw_mlir
from flashsim.emit import emit_cpp
from flashsim.opt import optimize


@pytest.mark.parametrize("width", [1, 8, 64, 256])
def test_created_array_uses_circt_index_order(tmp_path, width):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    src = f"""
    hw.module @ArrayCreateTest(in %clock : i1, in %a : i{width}, in %b : i{width}, in %c : i{width}, in %idx : i2, out result : i{width}) {{
      %arr = hw.array_create %a, %b, %c : i{width}
      %selected = hw.array_get %arr[%idx] : !hw.array<3xi{width}>, i2
      hw.output %selected : i{width}
    }}
    """
    model = optimize(parse_hw_mlir(src, top="ArrayCreateTest"))
    (tmp_path / "dut.h").write_text(emit_cpp(model))
    steps = []
    mask = (1 << width) - 1
    for round_no in range(4):
        # Distinct values for wide entries, changing single-bit inputs too.
        values = [((round_no + 1) << (i * 3)) ^ (i + round_no) for i in range(3)]
        if width == 256:
            values = [value | ((round_no + i + 1) << 192) for i, value in enumerate(values)]
        values = [value & mask for value in values]
        for name, value in zip(("a", "b", "c"), values):
            if width > 128:
                steps.extend(f"d.{name}[{i}] = 0x{(value >> (64 * i)) & ((1 << 64) - 1):x}ull;" for i in range(4))
            else:
                steps.append(f"d.{name} = 0x{value:x}ull;")
        for idx in range(3):
            expected = values[2 - idx]
            steps.append(f"d.idx = {idx}; d.poke_inputs(); d.eval_result();")
            if width > 128:
                steps.extend(f"if (d.result[{i}] != 0x{(expected >> (64 * i)) & ((1 << 64) - 1):x}ull) return {i + 1};" for i in range(4))
            else:
                steps.append(f"if (d.result != 0x{expected:x}ull) return 1;")
    main = tmp_path / "check.cpp"
    main.write_text('#include "dut.h"\nint main() { ArrayCreateTestDut d;\n' + "\n".join(steps) + "\n}\n")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
