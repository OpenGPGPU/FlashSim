"""The optimized commit helper must preserve every invalidation increment."""
import shutil
import subprocess

import pytest

from flashsim.emit import split_dut_methods


def test_commit_helper_preserves_duplicate_ids_and_wraparound(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    cpp = """#include <cstdint>
static inline void fs_bump(uint32_t *pg, const uint16_t *ids, unsigned n) {
  for (unsigned i = 0; i < n; ++i) pg[ids[i]]++;
}
struct CheckDut {
  uint32_t pg[4] = {0xffffffffu, 9u, 10u, 0u};
  const uint16_t ids[3] = {0u, 3u, 0u};
  void _commit() {
""" + "    // Force the commit out of line.\n" * 20 + """    fs_bump(pg, ids, 3u);
    fs_bump(pg, ids, 0u);
  }
};
"""
    header, parts = split_dut_methods(cpp, n_shards=2)
    names = dict(parts)
    assert "CheckDut_commit_bump(" in names["dut_commit.cpp"]
    assert "fs_bump(" not in names["dut_commit.cpp"]
    assert "CheckDut_commit_bump(" in names["dut_commit_helpers.cpp"]
    (tmp_path / "dut.h").write_text(header)
    objects = []
    for name, source in parts:
        path = tmp_path / name
        path.write_text(source)
        obj = path.with_suffix(".o")
        opt = "-O0" if name == "dut_commit.cpp" else "-O2"
        subprocess.run([compiler, "-std=c++17", opt, "-c", str(path),
                        "-o", str(obj)], check=True)
        objects.append(str(obj))
    main = tmp_path / "main.cpp"
    main.write_text('''#include "dut.h"
extern void CheckDut_commit_bump(uint32_t*,const uint16_t*,unsigned);
int main() {
  CheckDut d;
  d._commit();
  if (d.pg[0] != 1u || d.pg[1] != 9u || d.pg[2] != 10u || d.pg[3] != 1u)
    return 1;
  d._commit();
  if (d.pg[0] != 3u || d.pg[3] != 2u) return 2;
  uint16_t ids[100];
  for (unsigned i=0;i<100;i++) ids[i]=(i*7)%3;
  for (unsigned n=0;n<=100;n++) {
    uint32_t a[4]={UINT32_MAX,UINT32_MAX-2,7,99};
    uint32_t b[4]={UINT32_MAX,UINT32_MAX-2,7,99};
    for (unsigned i=0;i<n;i++) b[ids[i]]++;
    CheckDut_commit_bump(a,ids,n);
    for (unsigned i=0;i<4;i++) if(a[i]!=b[i]) return 3;
  }
  CheckDut_commit_bump(nullptr,nullptr,0);
  return 0;
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", str(main), *objects,
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_commit_without_invalidation_needs_no_helper():
    cpp = "struct QuietDut {\n  void _commit() {\n" + "    ;\n" * 100 + "  }\n};\n"
    _, parts = split_dut_methods(cpp)
    assert "dut_commit_helpers.cpp" not in dict(parts)
