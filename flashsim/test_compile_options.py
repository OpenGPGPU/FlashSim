"""Opt-in compiler flags must reach shards and invalidate cached objects."""
from pathlib import Path
import shutil
import subprocess

import pytest

from flashsim import experiment
from flashsim.circt_frontend import parse_hw_mlir
from flashsim.emit import emit_cpp, split_dut_methods
from flashsim.opt import optimize


def test_alias_and_commit_options_invalidate_the_shard_cache(tmp_path, monkeypatch):
    for name in ('dut.h','dut_0.cpp','dut_commit.cpp','dut_commit_chunks.cpp',
                 'dut_commit_part000.cpp'):
        (tmp_path/name).write_text('// fixture\n')
    calls=[]
    monkeypatch.setattr(experiment, '_need', lambda _name: '/fake/c++')
    monkeypatch.setattr(experiment, 'flashsim_main', lambda *_args: '// wrapper\n')
    def run(command, **_kwargs):
        calls.append(command)
        if '-o' in command:
            Path(command[command.index('-o')+1]).write_bytes(b'object')
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(experiment, '_run', run)
    monkeypatch.setenv('FLASHSIM_SHARD_OPT','-O2')
    monkeypatch.setenv('FLASHSIM_COMPILE_JOBS','2')
    monkeypatch.delenv('FLASHSIM_CONSERVATIVE_ALIAS',raising=False)
    monkeypatch.delenv('FLASHSIM_COMMIT_CHUNKS_OPT',raising=False)
    monkeypatch.delenv('FLASHSIM_STRICT_FLEX_ARRAYS',raising=False)
    def compile_once():
        before=len(calls)
        experiment._compile_flash(tmp_path,'Fixture','check',4)
        return [c for c in calls[before:] if '-c' in c]
    first=compile_once()
    assert len(first)==4 and all('-fno-strict-aliasing' not in c for c in first)
    assert not compile_once()
    monkeypatch.setenv('FLASHSIM_CONSERVATIVE_ALIAS','1')
    changed=compile_once()
    assert len(changed)==4 and all('-fno-strict-aliasing' in c for c in changed)
    assert not compile_once()
    monkeypatch.setenv('FLASHSIM_STRICT_FLEX_ARRAYS','1')
    changed=compile_once()
    assert len(changed)==4 and all('-fstrict-flex-arrays=1' in c for c in changed)
    assert not compile_once()
    monkeypatch.setenv('FLASHSIM_STRICT_FLEX_ARRAYS','bad')
    with pytest.raises(ValueError,match='FLASHSIM_STRICT_FLEX_ARRAYS'):
        compile_once()
    monkeypatch.setenv('FLASHSIM_STRICT_FLEX_ARRAYS','1')
    monkeypatch.setenv('FLASHSIM_COMMIT_CHUNKS_OPT','-O0')
    changed=compile_once()
    chunks=next(c for c in changed if any(str(a).endswith('dut_commit_chunks.cpp') for a in c))
    part=next(c for c in changed if any(str(a).endswith('dut_commit_part000.cpp') for a in c))
    assert len(changed)==4 and '-O0' in chunks and '-O0' in part
    monkeypatch.setenv('FLASHSIM_CONSERVATIVE_ALIAS','invalid')
    with pytest.raises(ValueError,match='FLASHSIM_CONSERVATIVE_ALIAS'):
        compile_once()
    monkeypatch.setenv('FLASHSIM_CONSERVATIVE_ALIAS','1')
    monkeypatch.setenv('FLASHSIM_COMMIT_CHUNKS_OPT','-O9')
    with pytest.raises(ValueError,match='FLASHSIM_COMMIT_CHUNKS_OPT'):
        compile_once()


def test_real_fixed_array_model_matches_with_strict_flex(tmp_path, monkeypatch):
    compiler=shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    probe=subprocess.run([compiler,'-fstrict-flex-arrays=1','-x','c++','-fsyntax-only','-'],
                         input='int main() {}\n',text=True,capture_output=True)
    if probe.returncode:
        pytest.skip('compiler does not support strict flexible-array levels')
    model=optimize(parse_hw_mlir('''
      hw.module @ArrayCheck(in %clock : i1, in %a : i32, in %b : i32, in %c : i32, in %idx : i2, out result : i32) {
        %arr = hw.array_create %a, %b, %c : i32
        %selected = hw.array_get %arr[%idx] : !hw.array<3xi32>, i2
        hw.output %selected : i32
      }
    ''',top='ArrayCheck'))
    header,parts=split_dut_methods(emit_cpp(model),n_shards=2)
    main='''#include "dut.h"
      #include <cstdio>
      int main() { ArrayCheckDut d;
        for (unsigned i=0; i<128; ++i) {
          d.a=i+11; d.b=i*3+17; d.c=i*5+23; d.idx=i%3;
          d.poke_inputs(); d.eval_result(); std::printf("%u\\n",d.result);
        }
      }
    '''
    monkeypatch.setattr(experiment,'flashsim_main',lambda *_args:main)
    for key in ('FLASHSIM_CONSERVATIVE_ALIAS','FLASHSIM_COMMIT_CHUNKS_OPT'):
        monkeypatch.delenv(key,raising=False)
    outputs=[]
    for value in ('0','1'):
        work=tmp_path/value
        work.mkdir()
        (work/'dut.h').write_text(header)
        for name,body in parts:
            (work/name).write_text(body)
        monkeypatch.setenv('FLASHSIM_STRICT_FLEX_ARRAYS',value)
        executable=experiment._compile_flash(work,'ArrayCheck','check',128)
        outputs.append(subprocess.check_output([str(executable)],text=True).splitlines())
    expected=[str((i*5+23,i*3+17,i+11)[i%3]) for i in range(128)]
    assert outputs[0]==outputs[1]==expected
