from pathlib import Path
import subprocess
import sys

import pytest

from flashsim import __main__ as cli
from flashsim import experiment, qemu


def test_experiment_includes_only_requested_external_gpu_tops(tmp_path, monkeypatch, capsys):
    synthetic = tmp_path / 'counter.v'
    gpu_dir = tmp_path / 'external gpu rtl'
    gpu_top = gpu_dir / experiment.GPU_BENCHES['DrawContextFifo']
    gpu_top.parent.mkdir(parents=True)
    gpu_top.write_text('module DrawContextFifo; endmodule\n')
    calls = []
    monkeypatch.setattr(experiment, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(experiment, 'write_benches', lambda *_args, **_kwargs: {'counter': synthetic})
    monkeypatch.setattr(experiment, '_resolve_frontend', lambda _frontend: 'circt')

    def run_bench(name, vfile, work, frontend):
        calls.append((name, vfile))
        return experiment.BenchResult(name, True, 1, 1, 1, True)

    monkeypatch.setattr(experiment, 'run_bench', run_bench)
    experiment.run_experiment()
    assert calls == [('counter', synthetic)]
    assert 'GPU benches omitted' in capsys.readouterr().out
    calls.clear()
    experiment.run_experiment(gpu_rtl_dir=gpu_dir)
    assert calls == [('counter', synthetic), ('DrawContextFifo', gpu_top)]
    assert 'GPU benches: 1/16' in capsys.readouterr().out


def test_experiment_rejects_missing_or_empty_external_directory(tmp_path):
    with pytest.raises(SystemExit, match='GPU RTL directory does not exist'):
        experiment.run_experiment(gpu_rtl_dir=tmp_path / 'missing')
    with pytest.raises(SystemExit, match='no recognized GPU benchmark tops'):
        experiment.run_experiment(gpu_rtl_dir=tmp_path)


def test_cli_forwards_external_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'run_experiment', lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(cli, 'print_summary', lambda _results: 0)
    monkeypatch.setattr(cli, 'run_qemu_smoke',
                        lambda rtl, frontend: calls.append((rtl, frontend)) or 0)
    assert cli.main(['experiment', '--gpu-rtl-dir', str(tmp_path), '--frontend', 'circt']) == 0
    assert calls.pop() == {'stages': 12, 'rounds': 8, 'frontend': 'circt', 'gpu_rtl_dir': tmp_path}
    top = tmp_path / 'GpuHostAxi.sv'
    assert cli.main(['qemu-smoke', '--rtl', str(top), '--frontend', 'native']) == 0
    assert calls.pop() == (top, 'native')
    with pytest.raises(SystemExit):
        cli.main(['qemu-smoke'])
    with pytest.raises(SystemExit, match='GPU RTL top does not exist'):
        qemu.run_qemu_smoke(top)


def test_commit_profiler_requires_external_gpu_top(tmp_path):
    script = Path(__file__).resolve().parent.parent / 'scripts/commit_profile.py'
    result = subprocess.run([sys.executable, str(script), 'GpuSystem', '--gpu-rtl-dir', str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert str(tmp_path / 'gpu-system/GpuSystem.sv') in result.stderr
