"""Large invalidation and wake indices must not truncate to 16 bits."""
import shutil
import subprocess

import pytest

from flashsim import emit
from flashsim.ir import AlwaysFF, Assign, Id, Module, Signal


@pytest.mark.parametrize('count,expected', [(0,'uint16_t'), (65536,'uint16_t'),
                                             (65537,'uint32_t'), (249320,'uint32_t')])
def test_index_width_covers_the_last_index(count, expected):
    assert emit._index_type_for_count(count) == expected


def test_unrepresentable_counts_are_rejected():
    for count in (-1, 1 << 32):
        with pytest.raises(ValueError):
            emit._index_type_for_count(count)


def test_generated_wide_tables_invalidate_and_wake_without_aliasing(tmp_path, monkeypatch):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    sigs = {'clock': Signal('clock', 1, 'input'), 'x': Signal('x', 1, 'input'),
            'result': Signal('result', 1, 'output')}
    mod = Module('WideIndices', sigs, [Assign('result', Id('x'))],
                 AlwaysFF('clock', []), list(sigs))
    original = emit._partition_maps
    def wide_pages(*args):
        wire_part, leaves, _count = original(*args)
        leaves['x'] = [0, 1, 2, 3, 4, 65535, 65536]
        return wire_part, leaves, 65537
    monkeypatch.setattr(emit, '_partition_maps', wide_pages)
    # Force a high hold index in a small IR so this boundary is tested without
    # constructing 65,537 hold buckets or rendering a frame.
    wake_ids = (0, 1, 2, 3, 4, 65535, 65536)
    monkeypatch.setattr(emit, '_hold_wake_tables', lambda _sets: {wake_ids: 0})
    original_type = emit._index_type_for_count
    monkeypatch.setattr(emit, '_index_type_for_count',
                        lambda count: original_type(65537 if count == 0 else count))
    cpp = emit.emit_cpp(mod)
    assert 'static const uint32_t _inv0[]' in cpp
    assert 'static const uint32_t _hw0[]' in cpp
    (tmp_path/'dut.h').write_text(cpp)
    source = tmp_path/'main.cpp'
    source.write_text('''#include "dut.h"
static WideIndicesDut d;
int main() {
  uint32_t pages[65537] = {};
  fs_bump(pages, _inv0, 7);
  if (pages[0]!=1 || pages[65535]!=1 || pages[65536]!=1 || pages[5]) return 1;
  uint64_t need[1025]={}, any[17]={}; unsigned words=0;
  fs_wake(need, any, &words, _hw0, 7);
  if (!(need[0]&1) || !(need[1023]&(1ull<<63)) || !(need[1024]&1)) return 2;
  if (!(any[0]&1) || !(any[15]&(1ull<<63)) || !(any[16]&1)) return 3;
  if (words!=3) return 4;
}
''')
    binary=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17','-O1',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_split_commit_preserves_wide_and_packed_indices(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    low=', '.join(map(str, range(32)))
    cpp=f'''#include <cstdint>
static const uint32_t _inv0[] = {{0, 65535, 65536, 249318, 65536, 0, 1}};
static const uint32_t _inv1[] = {{{low}}};
static inline void fs_bump(uint32_t *pg, const uint32_t *ids, unsigned n) {{
  for(unsigned i=0;i<n;i++)pg[ids[i]]++;
}}
struct WideDut {{
  uint32_t _pg[249320] = {{}};
  void _commit() {{
''' + '    // Force the commit out of line.\n'*30 + '''    fs_bump(_pg, _inv0, 7u);
    fs_bump(_pg, _inv1, 32u);
  }
};
'''
    header,parts=emit.split_dut_methods(cpp,n_shards=2)
    files=dict(parts)
    assert 'const uint32_t *ids' in files['dut_commit_helpers.cpp']
    assert 'dut_commit_blocks.cpp' in files
    (tmp_path/'dut.h').write_text(header)
    objects=[]
    for name,text in parts:
        path=tmp_path/name
        path.write_text(text)
        obj=path.with_suffix('.o')
        subprocess.run([compiler,'-std=c++17','-O1','-c',str(path),'-o',str(obj)],check=True)
        objects.append(str(obj))
    main=tmp_path/'main.cpp'
    main.write_text('''#include "dut.h"
static WideDut d;
int main() {
  d._pg[65536]=0xffffffffu;
  d._commit();
  if(d._pg[0]!=3 || d._pg[1]!=2 || d._pg[2]!=1) return 1;
  if(d._pg[65535]!=1 || d._pg[65536]!=1 || d._pg[249318]!=1) return 2;
  d._commit();
  if(d._pg[0]!=6 || d._pg[65536]!=3 || d._pg[249318]!=2) return 3;
}
''')
    binary=tmp_path/'check'
    subprocess.run([compiler,'-std=c++17',str(main),*objects,'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
