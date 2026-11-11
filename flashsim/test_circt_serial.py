from pathlib import Path

import pytest

from flashsim import circt_frontend, compile as compiler
from flashsim.circt_frontend import CirctError


def test_serial_flag_and_validation(tmp_path, monkeypatch):
    top = tmp_path / "Top.sv"
    top.write_text("module Top; endmodule\n")
    monkeypatch.setattr(circt_frontend, "find_bin", lambda _: Path("/circt-verilog"))
    monkeypatch.delenv("FLASHSIM_CIRCT_SERIAL", raising=False)
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
    for value in ("0", "0", "1", "1", "0"):
        monkeypatch.setenv("FLASHSIM_CIRCT_SERIAL", value)
        compiler.compile_verilog(top.read_text(), frontend="circt", path=top, mlir_path=mlir)
    assert calls == [False, True, False]
    mlir.with_suffix(".mlir.circt-serial").unlink()
    compiler.compile_verilog(top.read_text(), frontend="circt", path=top, mlir_path=mlir)
    assert calls == [False, True, False, False]
