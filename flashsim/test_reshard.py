"""A saved model can be repartitioned without changing its execution."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from flashsim.emit import emit_cpp, split_dut_methods
from flashsim.ir import AlwaysFF, Assign, BinOp, Const, Id, MemRead, MemWrite, Module, Signal
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


def test_optnone_marks_only_large_methods_and_preserves_behavior(tmp_path, monkeypatch):
    compiler = shutil.which("clang++")
    if not compiler:
        pytest.skip("Clang unavailable")
    source, out = tmp_path / "original", tmp_path / "derived"
    fixture(source)
    shard = source / "dut_0.cpp"
    original = shard.read_text()
    shard.write_text(original.replace("  value *= 3;", "  value *= 3;\n" + "  value += 0;\n" * 20))
    baseline = run_model(source, tmp_path, "unoptimized_baseline")
    create(source, out, maximum=70, commit_parts=2, optnone_bytes=100)
    record = json.loads((out / "flashsim-reshard.json").read_text())
    assert record["optnone_methods"]["dut_0.cpp"] == ["b"]
    annotated = next(p for p in out.glob("dut_*.cpp") if b"optnone" in p.read_bytes())
    assert b"#error" in annotated.read_bytes()
    verify(source, out)
    monkeypatch.setattr(shutil, "which", lambda name: compiler if name == "c++" else None)
    assert run_model(out, tmp_path, "unoptimized_derived") == baseline

    record["optnone_methods"]["dut_0.cpp"] = ["a"]
    (out / "flashsim-reshard.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="optnone methods"):
        verify(source, out)


def store_fixture(path: Path, width: int) -> str:
    """96 ordered ports exercise store chunks and independently enabled writes."""
    signal_specs = [
        ("clock", 1, "input", 0), ("addr", 2, "input", 0),
        ("even", 1, "input", 0), ("odd", 1, "input", 0),
        ("even_enabled", 1, "wire", 0), ("odd_enabled", 1, "wire", 0),
        ("m", width, "mem", 4),
    ]
    if width == 32:
        signal_specs += [("old", 32, "output", 0)]
    else:
        signal_specs += [("a", width, "input", 0), ("b", width, "input", 0)]
    signals = {name: Signal(name, bits, kind, depth=depth)
               for name, bits, kind, depth in signal_specs}
    assigns = [Assign("even_enabled", Id("even")), Assign("odd_enabled", Id("odd"))]
    if width == 32:
        assigns.append(Assign("old", MemRead("m", Id("addr"))))
    writes = [MemWrite("m", Id("addr"),
                       BinOp("+", Id("old"), Const(i)) if width == 32 else Id("b" if i % 2 else "a"),
                       Id("odd_enabled" if i % 2 else "even_enabled")) for i in range(96)]
    ports = [name for name, signal in signals.items() if signal.kind in ("input", "output")]
    module = Module(name="StoreProof", signals=signals, assigns=assigns,
                    always=AlwaysFF("clock", []), ports=ports, mem_writes=writes)
    header, parts = split_dut_methods(emit_cpp(module), n_shards=2)
    path.mkdir()
    (path / "dut.h").write_text(header)
    for name, content in parts:
        (path / name).write_text(content)
    if width == 32:
        return '''#include "dut.h"
int main() { StoreProofDut d; unsigned expected[4] = {}; unsigned s = 1;
  for (int i = 0; i < 2000; ++i) {
    s = s * 1664525u + 1013904223u; d.addr = (s >> 12) & 3;
    d.even = (s >> 4) & 1; d.odd = (s >> 8) & 1;
    if (d.odd) expected[d.addr] += 95; else if (d.even) expected[d.addr] += 94;
    d.tick(); for (int j = 0; j < 4; ++j) if (d.m[j] != expected[j]) return 1;
  }
}
'''
    return '''#include "dut.h"
int main() { StoreProofDut d; uint64_t expected[4][3] = {}; unsigned s = 1;
  for (int i = 0; i < 2000; ++i) {
    s = s * 1664525u + 1013904223u; d.addr = (s >> 12) & 3;
    d.even = (s >> 4) & 1; d.odd = (s >> 8) & 1;
    for (int j = 0; j < 3; ++j) {
      d.a[j] = s + j; d.b[j] = s * 5ull + j;
      if (d.odd) expected[d.addr][j] = d.b[j];
      else if (d.even) expected[d.addr][j] = d.a[j];
    }
    d.tick();
    for (int j = 0; j < 4; ++j) for (int k = 0; k < 3; ++k)
      if (d.m[j][k] != expected[j][k]) return 1;
  }
}
'''


@pytest.mark.parametrize("width", [32, 192])
def test_local_store_helpers_preserve_ordered_writes(tmp_path, width):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source, out = tmp_path / "original", tmp_path / "localized"
    main = tmp_path / "main.cpp"
    main.write_text(store_fixture(source, width))
    create(source, out, maximum=4096, commit_parts=1, local_store_helpers=True)
    record = json.loads((out / "flashsim-reshard.json").read_text())
    assert record["local_store_count"] == 3
    assert "dut_mem_stores.cpp" in record["input_sha256"]
    assert "dut_mem_stores.cpp" not in record["output_sha256"]
    assert b"_mem_store_part_" not in (out / "dut.h").read_bytes()
    verify(source, out)
    for model in (source, out):
        for optimization in ("-O0", "-O2"):
            binary = tmp_path / (model.name + optimization)
            subprocess.run([compiler, "-std=c++17", optimization, "-I", str(model), str(main),
                            *map(str, sorted(model.glob("dut_*.cpp"))), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


def test_local_store_helpers_reject_changed_bodies_and_guards(tmp_path):
    source, out = tmp_path / "original", tmp_path / "localized"
    store_fixture(source, 32)
    create(source, out, maximum=4096, commit_parts=1, local_store_helpers=True)
    record_path = out / "flashsim-reshard.json"
    record = json.loads(record_path.read_text())
    shard = next(p for p in out.glob("dut_*.cpp") if b"[this]()" in p.read_bytes())
    contents = shard.read_bytes()
    shard.write_bytes(contents.replace(b"[this]()", b"[this] ()", 1))
    from flashsim.reshard import digest
    record["output_sha256"][shard.name] = digest(shard)
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="do not reconstruct"):
        verify(source, out)

    shard.write_bytes(contents)
    record["output_sha256"][shard.name] = digest(shard)
    header = out / "dut.h"
    header.write_bytes(header.read_bytes() + b"// changed\n")
    record["output_sha256"][header.name] = digest(header)
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="derived header"):
        verify(source, out)

    helper = source / "dut_mem_stores.cpp"
    helper.write_text(helper.read_text().replace("    if (__we0) {", "    (void)__func__;\n    if (__we0) {", 1))
    with pytest.raises(ValueError, match="context-sensitive"):
        create(source, tmp_path / "unsafe", maximum=4096, commit_parts=1, local_store_helpers=True)
    assert not (tmp_path / "unsafe").exists()


def test_local_store_helpers_require_matching_port_guards(tmp_path):
    source, out = tmp_path / "original", tmp_path / "localized"
    store_fixture(source, 32)
    shard = next(p for p in source.glob("dut_[0-9]*.cpp")
                 if b"_mem_store_part_0();" in p.read_bytes())
    lines = shard.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if b"_mem_store_part_0();" in line:
            assert b"__we0 || __we1" in line
            lines[index] = line.replace(b"__we0 || __we1", b"__we1 || __we0", 1)
            break
    shard.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="guards do not match"):
        create(source, out, maximum=4096, commit_parts=1, local_store_helpers=True)
    assert not out.exists()


def test_local_store_helpers_restrict_header_macros(tmp_path):
    source = tmp_path / "original"
    store_fixture(source, 32)
    header = source / "dut.h"
    original = header.read_bytes()
    header.write_bytes(b"#define FLASHSIM_DATA_REGIONS 1\n" + original +
                       b'#pragma GCC diagnostic push\n'
                       b'#pragma GCC diagnostic ignored "-Winvalid-offsetof"\n'
                       b'#pragma GCC diagnostic pop\n')
    out = tmp_path / "allowed"
    create(source, out, maximum=4096, commit_parts=1, local_store_helpers=True)
    verify(source, out)
    header.write_bytes(b"#define WHERE __func__\n" + original)
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="preprocessor directive"):
        create(source, rejected, maximum=4096, commit_parts=1, local_store_helpers=True)
    assert not rejected.exists()
