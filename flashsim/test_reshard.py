"""A saved model can be repartitioned without changing its execution."""
from pathlib import Path
import shutil
import subprocess

import pytest

from flashsim.reshard import create, verify


def fixture(path: Path) -> None:
    path.mkdir()
    (path / "dut.h").write_text('''#include <cstdint>
struct TinyDut {
  uint32_t value = 0;
  void a(); void b(); void c(); void d(); void _commit();
};
''')
    (path / "dut_0.cpp").write_text('''#include "dut.h"

void TinyDut::a() {
  value += 1;
}

void TinyDut::b() {
  value *= 3;
}

void TinyDut::c() {
  value ^= 12;
}

void TinyDut::d() {
  value += 9;
}
''')
    (path / "dut_1.cpp").write_text('#include "dut.h"\n\n')
    (path / "dut_commit_chunks.cpp").write_text('''#include "dut.h"
static void TinyDut_commit_part0(TinyDut *d, unsigned k) {
  d->value += k + 2;
}
static void TinyDut_commit_part1(TinyDut *d, unsigned k) {
  d->value ^= k + 5;
}
void TinyDut::_commit() {
  TinyDut_commit_part0(this, 3);
  TinyDut_commit_part1(this, 4);
}
''')
    (path / "dut_commit_helpers.cpp").write_text("int unrelated_helper() { return 7; }\n")


def run_model(model: Path, tmp: Path, name: str) -> str:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    main = tmp / (name + ".cpp")
    main.write_text('''#include "dut.h"
#include <cstdio>
int main() {
  TinyDut d; d.a(); d.b(); d.c(); d.d(); d._commit();
  std::printf("%u\\n", d.value);
}
''')
    binary = tmp / name
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(model), str(main),
                    *map(str, sorted(model.glob("dut_*.cpp"))), "-o", str(binary)], check=True)
    return subprocess.check_output([str(binary)], text=True)


def test_method_and_commit_partitions_preserve_compiled_behavior(tmp_path):
    source, out = tmp_path / "original", tmp_path / "derived"
    fixture(source)
    baseline = run_model(source, tmp_path, "baseline")
    create(source, out, maximum=70, commit_parts=2)
    assert len(list(out.glob("dut_extra_*.cpp"))) >= 1
    assert len(list(out.glob("dut_commit_part*.cpp"))) == 2
    assert (out / "dut.h").read_text().startswith("#pragma once\n")
    verify(source, out)
    assert run_model(out, tmp_path, "resharded") == baseline
    assert not (source / "dut.h").read_text().startswith("#pragma once\n")
    with pytest.raises(FileExistsError):
        create(source, out, maximum=70, commit_parts=2)


def test_rejects_tampering_and_unexpected_top_level_content(tmp_path):
    source, out = tmp_path / "original", tmp_path / "derived"
    fixture(source)
    create(source, out, maximum=70, commit_parts=2)
    path = out / "dut_extra_0_0001.cpp"
    data = path.read_text()
    path.write_text(data + "\n// changed\n")
    with pytest.raises(ValueError, match="output differs"):
        verify(source, out)
    path.write_text(data)
    (source / "dut_0.cpp").write_text((source / "dut_0.cpp").read_text() + "\nint injected;\n")
    with pytest.raises(ValueError, match="input differs"):
        verify(source, out)
    with pytest.raises(ValueError, match="top-level content"):
        create(source, tmp_path / "failed", maximum=70, commit_parts=2)
    assert not (tmp_path / "failed").exists()
