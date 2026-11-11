"""An opt-in Clang PCH must leave generated model objects unchanged."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from flashsim.arti_model import BUILD_EMBEDDED
from flashsim.emit import emit_cpp
from flashsim.ir import AlwaysFF, Module


def test_emitted_header_is_safe_to_include_twice(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    header = emit_cpp(Module("Empty", {}, [], AlwaysFF("clk", []), []))
    assert header.startswith("#pragma once\n")
    (tmp_path / "dut.h").write_text(header)
    (tmp_path / "check.cpp").write_text(
        '#include "dut.h"\n#include "dut.h"\nint main() { EmptyDut d; d.tick(); }\n')
    subprocess.run([compiler, "-std=c++17", str(tmp_path / "check.cpp"),
                    "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)


def test_pch_matches_normal_objects_with_different_optimization_levels(tmp_path):
    compiler = shutil.which("c++")
    if not compiler or "clang" not in subprocess.check_output([compiler, "--version"], text=True).lower():
        pytest.skip("Clang unavailable")
    model = tmp_path / "model with spaces"
    model.mkdir()
    stage = tmp_path / "qemu staging"
    (stage / "hw/misc").mkdir(parents=True)
    (model / "dut.h").write_text('''#pragma once
#ifdef __OPTIMIZE__
static constexpr int optimized = 10;
#else
static constexpr int optimized = 20;
#endif
struct TinyDut { int x; int bump(); };
''')
    (model / "dut_0.cpp").write_text('''#include "dut.h"
int TinyDut::bump() { return ++x + optimized; }
''')
    (model / "dut_commit_chunks.cpp").write_text('''#include "dut.h"
int commit(TinyDut *d) { return d->x + optimized; }
''')
    (model / "dut_commit_helpers.cpp").write_text("int helper() { return 7; }\n")
    (model / "arti_rtl_model.cpp").write_text('''#include "dut.h"
extern "C" int run() { TinyDut d{3}; return d.bump(); }
''')
    (model / "arti_rtl_model.h").write_text("extern \"C\" int run();\n")
    script = model / "build_embedded.sh"
    script.write_text(BUILD_EMBEDDED.replace("__TOP__", "Tiny"))
    env = {key: value for key, value in os.environ.items() if not key.startswith("FLASHSIM_")}
    env.update(QEMU_SRC=str(stage), SKIP_QEMU_REBUILD="1", CCACHE_DISABLE="1",
               FLASHSIM_COMPILE_JOBS="2", FLASHSIM_COMMIT_CHUNKS_OPT="-O0",
               TMPDIR=str(tmp_path))
    plain = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    objects = {p.name: p.read_bytes() for p in model.glob("*.o")}
    env["FLASHSIM_PRECOMPILED_HEADER"] = "1"
    pch = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert pch.returncode == 0, pch.stdout + pch.stderr
    assert "c++ -O0 precompiled dut.h" in pch.stdout
    assert "c++ -O2 precompiled dut.h" in pch.stdout
    assert {p.name: p.read_bytes() for p in model.glob("*.o")} == objects
    assert not list(tmp_path.glob("flashsim-pch.*"))
    main = model / "main.cpp"
    main.write_text('extern "C" int run(); int main() { return run() == 14 ? 0 : 1; }\n')
    subprocess.run([compiler, str(main), str(model / "libarti_rtl_model.a"),
                    "-o", str(model / "check")], check=True)
    subprocess.run([str(model / "check")], check=True)


def test_pch_rejects_unguarded_existing_header(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    stage = tmp_path / "stage"
    (stage / "hw/misc").mkdir(parents=True)
    for name, text in {"dut.h": "struct X {};\n", "dut_0.cpp": '#include "dut.h"\n',
                       "arti_rtl_model.cpp": "// wrapper\n", "arti_rtl_model.h": "// header\n"}.items():
        (model / name).write_text(text)
    script = model / "build_embedded.sh"
    script.write_text(BUILD_EMBEDDED.replace("__TOP__", "Tiny"))
    env = dict(os.environ, QEMU_SRC=str(stage), SKIP_QEMU_REBUILD="1",
               FLASHSIM_PRECOMPILED_HEADER="1", FLASHSIM_COMPILE_JOBS="1")
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "newly generated guarded dut.h" in result.stderr
    assert not (stage / "hw/misc/libarti_rtl_model.a").exists()
