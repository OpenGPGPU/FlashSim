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


def test_chunked_commit_preserves_typed_cases_epochs_and_wakes(tmp_path, monkeypatch):
    from flashsim import emit
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    monkeypatch.setattr(emit, "_COMMIT_CHUNK_MIN_CASES", 4)
    monkeypatch.setattr(emit, "_COMMIT_CHUNK_CASES", 4)
    fields = [("a", "uint8_t"), ("b", "uint16_t"), ("c", "uint32_t"),
              ("d", "uint64_t"), ("e", "unsigned __int128")]
    cpp = """#include <cstdint>
#include <cstring>
static const uint16_t ids[] = {0,1,0,2,2};
static inline void fs_bump(uint32_t *pg,const uint16_t *ids,unsigned n) {
  for(unsigned i=0;i<n;i++)pg[ids[i]]++;
}
struct ChunkDut {
  uint32_t _pg[3] = {UINT32_MAX,7,9};
  uint64_t _bep=0,_bst[1]={0},_cc=0;
  unsigned _nw=8,_chg=0,_h_words=0;
  uint64_t _h_need[1]={0},_h_any[1]={0};
  uint16_t _wl[8]={0,1,2,3,4,5,6,65535};
"""
    for name, typ in fields:
        cpp += f"  {typ} {name}=0,{name}__n=0;\n"
    cpp += "  uint64_t f[3]={},f__n[3]={};\n"
    cpp += "  void _commit() {\n    _cc += _nw;\n    _bep++;\n"
    cpp += "    for (unsigned i = 0; i < _nw; i++) {\n      switch (_wl[i]) {\n"
    for i, name in enumerate([n for n,t in fields] + ["f"]):
        condition = (f"{name}__n != {name}" if name != "f" else
                     "memcmp(f__n, f, sizeof(f))")
        assignment = (f"{name} = {name}__n;" if name != "f" else
                      "memcpy(f, f__n, sizeof(f));")
        cpp += f"      case {i}u: {{\n        if ({condition}) {{\n"
        cpp += "          if (_bst[0] != _bep) { _bst[0]=_bep; fs_bump(_pg,ids,5u); }\n"
        cpp += f"          _h_need[0] |= 1ull << {i};\n"
        cpp += "          if (!_h_any[0]) ++_h_words; _h_any[0] |= 1ull;\n"
        cpp += f"          _chg++;\n          {assignment}\n        }}\n      }} break;\n"
    cpp += "      }\n    }\n  }\n};\n"
    header, parts = split_dut_methods(cpp, n_shards=2)
    names = dict(parts)
    assert "dut_commit_chunks.cpp" in names
    assert "dut_commit.cpp" not in names
    assert "ChunkDut_commit_part1" in names["dut_commit_chunks.cpp"]
    (tmp_path/"dut.h").write_text(header)
    objects = []
    for name, source in parts:
        path = tmp_path/name
        path.write_text(source)
        obj = path.with_suffix(".o")
        subprocess.run([compiler,"-std=c++17","-O2","-c",str(path),"-o",str(obj)],check=True)
        objects.append(str(obj))
    main = tmp_path/"main.cpp"
    main.write_text('''#include "dut.h"
int main() {
  ChunkDut d;
  d.a__n=1;d.b__n=2;d.c__n=3;d.d__n=4;
  d.e__n=((unsigned __int128)1<<100)|5;
  d.f__n[0]=11;d.f__n[1]=22;d.f__n[2]=33;
  d._commit();
  if(d.a!=1 || d.b!=2 || d.c!=3 || d.d!=4 || d.e!=d.e__n || d.f[2]!=33)return 1;
  if(d._cc!=8 || d._chg!=6 || d._bep!=1 || d._h_need[0]!=63 || d._h_words!=1)return 2;
  if(d._pg[0]!=1 || d._pg[1]!=8 || d._pg[2]!=11)return 3;
  d._chg=0;d._commit();
  if(d._chg || d._cc!=16 || d._bep!=2 || d._pg[0]!=1)return 4;
  d.a__n=2;d.f__n[2]=44;d._h_need[0]=d._h_any[0]=0;d._h_words=0;
  d._commit();
  if(d.a!=2 || d.f[2]!=44 || d._chg!=2 || d._cc!=24 || d._bep!=3)return 5;
  if(d._pg[0]!=3 || d._pg[1]!=9 || d._pg[2]!=13)return 6;
  if(d._h_need[0]!=33 || d._h_words!=1)return 7;
}
''')
    binary=tmp_path/"check"
    subprocess.run([compiler,"-std=c++17",str(main),*objects,"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
