"""Large DUTs must preserve late member initializers on supported compilers."""
import shutil
import subprocess
import sys

import pytest

from flashsim.emit import _bulk_state_constructor


def compile_and_run(tmp_path, source, name):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    path = tmp_path / (name + ".cpp")
    path.write_text(source)
    binary = tmp_path / name
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_bulk_constructor_preserves_typed_values_arrays_and_alignment(tmp_path):
    source = r'''#include <cstdint>
#include <cstring>
#include <cstddef>
struct Mixed {
  uint8_t a = 7;
  uint16_t b = 513;
  uint32_t c = 0, marker = 17;
  uint32_t _eg0 = 0, _ed0 = 0;
  uint32_t noinit, later = 23;
  uint64_t pair[2] = {1, 2}, other[2] = {3, 4};
  uint32_t comma = (void(1), 9), next = 13;
  uint64_t d = ~0ull;
  unsigned __int128 wide = ((unsigned __int128)1 << 100) | 3;
  alignas(32) uint64_t flags[3] = {~0ull, 0, 0x123};
  uint8_t sparse[2][3] = {{1}, {2, 3}};
  uint32_t zeros[9] = {};
  unsigned count = 2;
  bool enabled = true;
  void exercise() {
    uint8_t local = 11;
    c = local;
  }
};
int main() {
    Mixed d;
    if(d.a!=7||d.b!=513||d.c||d.d!=~0ull||d.count!=2||!d.enabled)return 1;
    if(d.wide!=(((unsigned __int128)1<<100)|3))return 2;
    if(d.flags[0]!=~0ull||d.flags[1]||d.flags[2]!=0x123)return 3;
    if(reinterpret_cast<uintptr_t>(d.flags)%32)return 4;
    if(d.sparse[0][0]!=1||d.sparse[0][1]||d.sparse[1][0]!=2||d.sparse[1][1]!=3||d.sparse[1][2])return 5;
    for(auto value:d.zeros)if(value)return 6;
    d.exercise();if(d.c!=11)return 7;
    if(d.marker!=17||d._eg0||d._ed0||d.noinit||d.later!=23)return 8;
    if(d.pair[0]!=1||d.pair[1]!=2||d.other[0]!=3||d.other[1]!=4)return 9;
    if(d.comma!=9||d.next!=13)return 10;
}
'''
    transformed = _bulk_state_constructor(source, "Mixed", min_fields=1)
    assert "Mixed() {" in transformed
    assert "uint8_t local = 11;" in transformed
    assert "uint8_t a;" in transformed
    assert "uint32_t c, marker;" in transformed
    assert "uint32_t _eg0, _ed0;" in transformed
    compile_and_run(tmp_path, transformed, "mixed")


def test_bulk_constructor_initializes_fields_beyond_262144(tmp_path):
    # Apple Clang16 silently truncates a sufficiently large implicit ctor's
    # initializer list. Keep late nonzero fields beyond that boundary.
    source = "#include <cstdint>\n#include <cstring>\nstruct Large {\n"
    source += "".join(f"  uint8_t zero{i} = 0;\n" for i in range(262144))
    source += '''  uint8_t tail = 7;
  uint64_t pending[3] = {~0ull, 0, 0x12345678};
  unsigned words = 2;
};
int main(){auto*d=new Large();
 if(d->zero0||d->zero131071||d->zero262143||d->tail!=7)return 1;
 if(d->pending[0]!=~0ull||d->pending[1]||d->pending[2]!=0x12345678||d->words!=2)return 2;
 delete d;
}
'''
    transformed = _bulk_state_constructor(source, "Large")
    assert "Large() {" in transformed
    assert "uint8_t tail;" in transformed
    compile_and_run(tmp_path, transformed, "large")


def test_small_dut_constructor_text_stays_unchanged():
    source = "struct Small {\n  unsigned value = 3;\n};\n"
    assert _bulk_state_constructor(source, "Small") == source


def test_bulk_constructor_threshold_counts_declarators():
    source = "struct Paired {\n  uint32_t first = 0, second = 2;\n};\n"
    assert "Paired() {" in _bulk_state_constructor(source, "Paired", min_fields=2)
    assert _bulk_state_constructor(source, "Paired", min_fields=3) == source


def test_bulk_constructor_with_optimized_python(tmp_path):
    source = '''#include <cstdint>
#include <cstring>
struct Paired {
  uint64_t first[2] = {1, 2}, second[2] = {3, 4};
  uint32_t third = (void(0), 5), fourth = 6;
};
int main() {
  Paired p;
  return p.first[0]!=1 || p.first[1]!=2 || p.second[0]!=3 ||
         p.second[1]!=4 || p.third!=5 || p.fourth!=6;
}
'''
    code = ("import sys; from flashsim.emit import _bulk_state_constructor; "
            "print(_bulk_state_constructor(sys.argv[1], 'Paired', min_fields=1))")
    transformed = subprocess.check_output(
        [sys.executable, "-O", "-c", code, source], text=True)
    compile_and_run(tmp_path, transformed, "optimized-python")
