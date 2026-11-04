"""Opt-in compiler flags must reach shards and invalidate cached objects."""
from pathlib import Path
import subprocess

import pytest

from flashsim import experiment


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
