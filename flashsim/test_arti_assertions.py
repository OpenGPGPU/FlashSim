"""Exercise bound assertions through the generated ARTI C++ bridge."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from flashsim.arti_model import SOURCE, SYSTEM_SOURCE, _checked_assertion_source
from flashsim.toolchain import find_bin


def _rtl(root):
    checks = root / 'verification/assert'
    checks.mkdir(parents=True)
    fields = set(re.findall(r'g_rtl->([A-Za-z_]\w*)', SOURCE))
    outputs = {name[5:] for name in fields if name.startswith('eval_')}
    inputs = fields - outputs - {'tick', 'poke_inputs', 'reset'}
    inputs = {name for name in inputs if not name.startswith('eval_')}
    ports = ['input logic clock', 'input logic reset']
    ports += ['input logic [63:0] ' + name for name in sorted(inputs)]
    ports += ['output logic [63:0] ' + name for name in sorted(outputs)]
    body = '\n'.join('assign ' + name + " = 64'b0;" for name in sorted(outputs))
    top = root / 'GpuHostAxi.sv'
    top.write_text('module GpuHostAxi(\n' + ',\n'.join(ports) + '\n);\n' + body +
                   '\nlogic state; always_ff @(posedge clock) state <= !state;\nendmodule\n')
    check = checks / 'GpuHostAxi_Verification_Assert.sv'
    check.write_text('''module GpuHostAxi_Verification_Assert;
  always @(posedge GpuHostAxi.clock)
    if (!GpuHostAxi.reset && GpuHostAxi.io_s_axi_awvalid[0]) begin
      $error("invalid guest write"); $fatal;
    end
endmodule
''')
    (checks / 'layers-GpuHostAxi-Verification-Assert.sv').write_text(
        'bind GpuHostAxi GpuHostAxi_Verification_Assert verification_assert ();\n')
    (root / 'filelist.f').write_text(
        'GpuHostAxi.sv\nverification/assert/GpuHostAxi_Verification_Assert.sv\n')
    return top


def test_arti_model_cli_executes_opt_in_bound_assertions(tmp_path):
    compiler = shutil.which('c++')
    if not compiler or not find_bin('circt-verilog'):
        pytest.skip('CIRCT and a C++ compiler are required')
    top = _rtl(tmp_path / 'rtl')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    for enabled in (False, True):
        directory = tmp_path / ('checked' if enabled else 'default')
        cmd = [sys.executable, '-m', 'flashsim', 'arti-model', str(top),
               '-o', str(directory), '--frontend', 'circt']
        if enabled:
            cmd.append('--assertions')
        subprocess.run(cmd, cwd=root, env=env, check=True, capture_output=True, text=True)
        header = (directory / 'dut.h').read_text()
        assert ('assertion_count = 1' in header) == enabled
        main = directory / 'check.cpp'
        main.write_text('''#include "arti_rtl_model.h"
int main(int argc, char **) {
  arti_rtl_model_init();
  if (argc > 1) arti_rtl_model_write(0, 0, 4);
  return 0;
}
''')
        binary = directory / 'check'
        subprocess.run([compiler, '-std=c++17', '-O1', str(main),
                        str(directory / 'arti_rtl_model.cpp'), '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        passing = subprocess.run([str(binary)], capture_output=True, text=True)
        assert passing.returncode == 0, passing.stderr
        result = subprocess.run([str(binary), 'write'], capture_output=True, text=True)
        if enabled:
            assert result.returncode != 0
            assert 'FlashSim ARTI tick failed' in result.stderr
            assert 'verification_assert' in result.stderr
            assert 'invalid guest write' in result.stderr
        else:
            assert result.returncode == 0, result.stderr


def test_system_bridge_wraps_the_tick_and_fails_closed_on_template_drift():
    checked = _checked_assertion_source(SYSTEM_SOURCE, 'tick_nba')
    assert 'g_rtl->tick_nba();' in checked and 'std::abort();' in checked
    with pytest.raises(ValueError, match='insertion point'):
        _checked_assertion_source(SYSTEM_SOURCE, 'not_a_tick')
