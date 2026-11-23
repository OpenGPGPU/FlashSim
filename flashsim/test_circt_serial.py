from pathlib import Path

import pytest

from flashsim import circt_frontend, compile as compiler
from flashsim.circt_frontend import CirctError


def test_serial_flag_and_validation(tmp_path, monkeypatch):
    top = tmp_path / "Top.sv"
    top.write_text("module Top; endmodule\n")
    monkeypatch.setattr(circt_frontend, "find_bin", lambda _: Path("/circt-verilog"))
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
    monkeypatch.delenv("FLASHSIM_CIRCT_THREADS", raising=False)
    assert "--mlir-disable-threading" not in circt_frontend.circt_verilog_command(top)
    monkeypatch.setenv("FLASHSIM_CIRCT_SERIAL", "1")
    command = circt_frontend.circt_verilog_command(top, assertions=True)
    assert "--mlir-disable-threading" in command
    assert "--single-unit" in command
    monkeypatch.setenv("FLASHSIM_CIRCT_SERIAL", "invalid")
    with pytest.raises(CirctError, match="must be 0 or 1"):
        circt_frontend.circt_verilog_command(top)


def test_serial_setting_invalidates_mlir_cache(tmp_path, monkeypatch):
    top = tmp_path / "Top.sv"
    top.write_text("module Top; endmodule\n")
    mlir = tmp_path / "hw.mlir"
    calls = []

    def import_rtl(path, **kwargs):
        calls.append(circt_frontend.circt_serial_import())
        return "hw.module @Top() { hw.output }\n"

    monkeypatch.setattr(compiler, "circt_verilog_to_mlir", import_rtl)
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
    monkeypatch.delenv("FLASHSIM_CIRCT_THREADS", raising=False)
    for value in ("0", "0", "1", "1", "0"):
        monkeypatch.setenv("FLASHSIM_CIRCT_SERIAL", value)
        compiler.compile_verilog(top.read_text(), frontend="circt", path=top, mlir_path=mlir)
    assert calls == [False, True, False]
    mlir.with_suffix(".mlir.circt-serial").unlink()
    compiler.compile_verilog(top.read_text(), frontend="circt", path=top, mlir_path=mlir)
    assert calls == [False, True, False, False]


def test_bounded_workers_preserve_assertions_and_bind_cache(tmp_path, monkeypatch):
    top = tmp_path / "Top.sv"
    top.write_text("module Top; endmodule\n")
    mlir = tmp_path / "hw.mlir"
    calls = []

    def import_rtl(path, *, assertions):
        calls.append(circt_frontend.circt_verilog_command(path, assertions=assertions))
        return "hw.module @Top() { hw.output }\n"

    monkeypatch.setattr(circt_frontend, "find_bin", lambda _: Path("/circt-verilog"))
    monkeypatch.setattr(compiler, "circt_verilog_to_mlir", import_rtl)
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
    for threads in ("4", "4", "2", "2"):
        monkeypatch.setenv("FLASHSIM_CIRCT_THREADS", threads)
        compiler.compile_verilog(top.read_text(), frontend="circt", path=top,
                                 mlir_path=mlir, assertions=True)
    assert len(calls) == 2
    for command, workers in zip(calls, (4, 2)):
        assert f"--mlir-threads={workers}" in command
        assert "--single-unit" in command
        assert "--mlir-disable-threading" not in command
    monkeypatch.delenv("FLASHSIM_CIRCT_THREADS")
    compiler.compile_verilog(top.read_text(), frontend="circt", path=top,
                             mlir_path=mlir, assertions=True)
    assert len(calls) == 3
    assert all(not flag.startswith("--mlir-threads") for flag in calls[-1])


@pytest.mark.parametrize("threads", ["", "-1", "0", "4.0", "4294967296"])
def test_invalid_worker_limit(threads, monkeypatch):
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
    monkeypatch.setenv("FLASHSIM_CIRCT_THREADS", threads)
    with pytest.raises(CirctError, match="must be a positive 32-bit integer"):
        circt_frontend.circt_import_parallelism()


def test_bounded_workers_conflict_with_serial(monkeypatch):
    monkeypatch.setenv("FLASHSIM_CIRCT_SERIAL", "1")
    monkeypatch.setenv("FLASHSIM_CIRCT_THREADS", "4")
    with pytest.raises(CirctError, match="conflicts"):
        circt_frontend.circt_import_parallelism()


def test_largest_first_preserves_worker_limit_and_assertions(tmp_path, monkeypatch):
    top = tmp_path / "Top.sv"
    top.write_text("module Top; endmodule\n")
    monkeypatch.setattr(circt_frontend, "find_bin", lambda _: Path("/circt-verilog"))
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
    monkeypatch.setenv("FLASHSIM_CIRCT_THREADS", "4")
    monkeypatch.delenv("FLASHSIM_CIRCT_LARGEST_FIRST", raising=False)
    option = "--mlir-pass-schedule-largest-first"
    assert option not in circt_frontend.circt_verilog_command(top, assertions=True)
    monkeypatch.setenv("FLASHSIM_CIRCT_LARGEST_FIRST", "1")
    command = circt_frontend.circt_verilog_command(top, assertions=True)
    assert option in command
    assert "--mlir-threads=4" in command
    assert "--single-unit" in command
    monkeypatch.setenv("FLASHSIM_CIRCT_LARGEST_FIRST", "invalid")
    with pytest.raises(CirctError, match="must be 0 or 1"):
        circt_frontend.circt_verilog_command(top, assertions=True)
