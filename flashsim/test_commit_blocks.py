"""Immutable invalidation tables can vectorize without changing any counter."""
from collections import Counter
import re
import shutil
import subprocess

import pytest

from flashsim import emit


def test_block_encoding_preserves_every_increment_and_uint16_boundaries():
    for ids in [list(range(65536)), list(range(19)) * 3,
                [65535, 0, 0] + list(range(17)) + [3, 3, 65535],
                list(range(7, 30)) + list(range(18, 43))]:
        encoded = emit._invalidation_blocks(ids)
        assert encoded is not None
        words, scalar_count = encoded
        n8, n4 = words[:2]
        values = [v for start in words[2:2+n8] for v in range(start, start+8)]
        values += [v for start in words[2+n8:2+n8+n4] for v in range(start, start+4)]
        values += words[2+n8+n4:]
        assert len(words[2+n8+n4:]) == scalar_count
        assert Counter(values) == Counter(ids)
        assert all(0 <= v <= 65535 for v in words)
    for ids in [[], [0, 1, 2], list(range(0, 200, 3)), [-1], [65536]]:
        assert emit._invalidation_blocks(ids) is None


def test_unknown_mutable_shadow_and_partial_uses_keep_generic_path(monkeypatch):
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "1")
    declaration = "static const uint16_t _inv0[] = {" + ",".join(map(str, range(32))) + "};\n"
    header = declaration + "struct ProofDut { uint32_t _pg[32]; void _commit(); };\n"
    body = '#include "dut.h"\nvoid ProofDut::_commit() { ProofDut_commit_bump(_pg, _inv0, 32u); }\n'
    changed, helper = emit._outline_static_commit_blocks(header, "ProofDut", body)
    assert helper and changed != body
    # A second transform must not add or duplicate metadata.
    assert emit._outline_static_commit_blocks(header, "ProofDut", changed) == (changed, None)
    cases = [
        (header.replace("static const", "static"), body),
        (header.replace("uint32_t _pg[32];", "uint32_t _pg[32]; uint16_t _inv0[32];"), body),
        (header, body.replace("32u);", "31u);")),
        (header, body.replace("32u);", "n);")),
        (header, body.replace("ProofDut_commit_bump(_pg", "uint16_t _inv0[32]={}; ProofDut_commit_bump(_pg")),
        (header + "const auto alias = _inv0;\n", body),
        (header, body + 'const char *message="_inv0";\n'),
    ]
    for h, b in cases:
        assert emit._outline_static_commit_blocks(h, "ProofDut", b) == (b, None)
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "0")
    assert emit._outline_static_commit_blocks(header, "ProofDut", body) == (body, None)


def test_split_commit_blocks_match_legacy_counters_and_preserve_header(tmp_path, monkeypatch):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    ids = list(range(3, 35)) + [9, 9, 63, 0, 0] + list(range(40, 64))
    table = ", ".join(map(str, ids))
    cpp = f'''#include <cstdint>
static const uint16_t _inv0[] = {{{table}}};
static inline void fs_bump(uint32_t*pg,const uint16_t*ids,unsigned n) {{
 for(unsigned i=0;i<n;i++)pg[ids[i]]++;
}}
struct ProofDut {{
 uint32_t _pg[64]={{}};
  void _commit() {{
''' + "    // Enough body to outline the commit.\n" * 30 + f'''    fs_bump(_pg, _inv0, {len(ids)}u);
  }}
}};
'''
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "0")
    old_header, old_parts = emit.split_dut_methods(cpp, n_shards=2)
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "1")
    header, parts = emit.split_dut_methods(cpp, n_shards=2)
    assert header == old_header
    old = dict(old_parts)
    new = dict(parts)
    assert "dut_commit_blocks.cpp" in new
    assert new["dut_commit_helpers.cpp"] == old["dut_commit_helpers.cpp"]
    for name in old:
        if name != "dut_commit.cpp":
            assert new[name] == old[name]
    (tmp_path / "dut.h").write_text(header)
    objects = []
    for name, source in parts:
        path = tmp_path / name
        path.write_text(source)
        obj = path.with_suffix(".o")
        subprocess.run([compiler, "-std=c++17", "-O2", "-fsanitize=address,undefined",
                        "-c", str(path), "-o", str(obj)], check=True)
        objects.append(str(obj))
    source = tmp_path / "check.cpp"
    source.write_text('''#include "dut.h"
#include <cassert>
int main(){
 ProofDut d;uint32_t expected[64];
 for(unsigned i=0;i<64;i++)d._pg[i]=expected[i]=UINT32_MAX-i;
 for(unsigned pass=0;pass<100;pass++){
  d._commit();for(auto id:_inv0)expected[id]++;
  for(unsigned i=0;i<64;i++)assert(d._pg[i]==expected[i]);
 }
}
''')
    exe = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fsanitize=address,undefined",
                    str(source), *objects, "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
