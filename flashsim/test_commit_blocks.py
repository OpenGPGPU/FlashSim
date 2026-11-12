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


def test_block_encoding_accepts_uint32_ids_and_large_block_counts():
    cases = [list(range(65520, 65560)) + [65536, 65536, 0],
             list(range(0xfffffff0, 0x100000000)),
             list(range(8)) * 65536]
    for ids in cases:
        words, scalar_count = emit._invalidation_blocks(ids, max_index=0xffffffff)
        n8, n4 = words[:2]
        values = [v for start in words[2:2+n8] for v in range(start, start+8)]
        values += [v for start in words[2+n8:2+n8+n4] for v in range(start, start+4)]
        values += words[2+n8+n4:]
        assert len(words[2+n8+n4:]) == scalar_count
        assert Counter(values) == Counter(ids)
        assert all(0 <= value <= 0xffffffff for value in words)
    assert emit._invalidation_blocks(cases[-1]) is None
    assert emit._invalidation_blocks(list(range(16)) + [0x100000000],
                                     max_index=0xffffffff) is None


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
        (header, body.replace("32u);", "32u); ProofDut_commit_bump(_pg, _inv0, 31u);")),
        (header, body.replace("ProofDut_commit_bump(_pg", "uint16_t _inv0[32]={}; ProofDut_commit_bump(_pg")),
        (header + "const auto alias = _inv0;\n", body),
        (header + "#define _inv0_blocks 0\n", body),
        (header + "const uint16_t _inv0_blocks[] = {0};\n", body),
        (header + "void ProofDut_commit_bump_blocks();\n", body),
        (header, body + 'const char *message="_inv0";\n'),
    ]
    for h, b in cases:
        assert emit._outline_static_commit_blocks(h, "ProofDut", b) == (b, None)
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "0")
    assert emit._outline_static_commit_blocks(header, "ProofDut", body) == (body, None)


def test_partition_keeps_only_referenced_block_tables_and_falls_back_for_macros():
    prefix = '''#include "dut.h"
extern void ProofDut_commit_bump_blocks32(uint32_t *, const uint32_t *, unsigned);
static const uint32_t _inv0_blocks[] = {2, 0, 65536, 65544};
static const uint32_t _inv1_blocks[] = {2, 0, 65552, 65560};
'''
    body = prefix + '''static void ProofDut_commit_part0(ProofDut *d, unsigned k) {
  ProofDut_commit_bump_blocks32(d->_pg, _inv0_blocks, 0u);
}
static void ProofDut_commit_part1(ProofDut *d, unsigned k) {
  ProofDut_commit_bump_blocks32(d->_pg, _inv1_blocks, 0u);
}
void ProofDut::_commit() {
  ProofDut_commit_part0(this, 0u);
  ProofDut_commit_part1(this, 0u);
}
'''
    parts = dict(emit._partition_commit_chunks("ProofDut", body, 2))
    assert "_inv0_blocks[]" in parts["dut_commit_part000.cpp"]
    assert "_inv1_blocks[]" not in parts["dut_commit_part000.cpp"]
    assert "_inv1_blocks[]" in parts["dut_commit_part001.cpp"]
    assert "_inv0_blocks[]" not in parts["dut_commit_part001.cpp"]
    assert "_inv0_blocks[]" not in parts["dut_commit_chunks.cpp"]
    assert "_inv1_blocks[]" not in parts["dut_commit_chunks.cpp"]
    macro_body = body.replace('#include "dut.h"\n', '#include "dut.h"\n#define EXTRA 1\n')
    for text in dict(emit._partition_commit_chunks("ProofDut", macro_body, 2)).values():
        assert "_inv0_blocks[]" in text and "_inv1_blocks[]" in text


def test_small_wide_table_keeps_generic_helper(monkeypatch):
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "1")
    for count in (1023, 1024):
        table = ",".join(map(str, range(65536, 65536 + count)))
        header = f"static const uint32_t _inv0[] = {{{table}}};\n"
        body = f'#include "dut.h"\nvoid ProofDut::_commit() {{ ProofDut_commit_bump(_pg, _inv0, {count}u); }}\n'
        changed, helper = emit._outline_static_commit_blocks(header, "ProofDut", body)
        assert bool(helper) == (count == 1024)
        assert (changed != body) == (count == 1024)


@pytest.mark.parametrize("opt", ["-O0", "-O2"])
def test_split_commit_blocks_support_mixed_widths_and_uint32_wrap(tmp_path, monkeypatch, opt):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    low = list(range(32)) + [0, 0, 9]
    high = list(range(65520, 66560)) + [65536, 65536, 2]
    table = lambda ids: ", ".join(map(str, ids))
    cpp = f'''#include <cstdint>
static const uint32_t _inv0[] = {{{table(low)}}};
static const uint32_t _inv1[] = {{{table(high)}}};
static inline void fs_bump(uint32_t*pg,const uint32_t*ids,unsigned n) {{
 for(unsigned i=0;i<n;i++)pg[ids[i]]++;
}}
struct ProofDut {{
 uint32_t _pg[66580]={{}};
  void _commit() {{
''' + "    // Enough body to outline the commit.\n" * 30 + f'''    fs_bump(_pg, _inv0, {len(low)}u);
    fs_bump(_pg, _inv1, {len(high)}u);
  }}
}};
'''
    monkeypatch.setenv("FLASHSIM_COMMIT_BLOCKS", "1")
    header, parts = emit.split_dut_methods(cpp, n_shards=2)
    files = dict(parts)
    assert "static const uint16_t _inv0_blocks[]" in files["dut_commit.cpp"]
    assert "static const uint32_t _inv1_blocks[]" in files["dut_commit.cpp"]
    assert "ProofDut_commit_bump_blocks32" in files["dut_commit_blocks.cpp"]
    (tmp_path / "dut.h").write_text(header)
    objects = []
    for name, source in parts:
        path = tmp_path / name
        path.write_text(source)
        obj = path.with_suffix(".o")
        subprocess.run([compiler, "-std=c++17", opt, "-fsanitize=address,undefined",
                        "-c", str(path), "-o", str(obj)], check=True)
        objects.append(str(obj))
    source = tmp_path / "check.cpp"
    source.write_text('''#include "dut.h"
#include <cassert>
int main(){
 ProofDut d;uint32_t expected[66580];
 for(unsigned i=0;i<66580;i++)d._pg[i]=expected[i]=UINT32_MAX-i;
 for(unsigned pass=0;pass<100;pass++){
  d._commit();for(auto id:_inv0)expected[id]++;for(auto id:_inv1)expected[id]++;
  for(unsigned i=0;i<66580;i++)assert(d._pg[i]==expected[i]);
 }
}
''')
    exe = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", opt, "-fsanitize=address,undefined",
                    str(source), *objects, "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


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
