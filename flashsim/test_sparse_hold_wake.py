from __future__ import annotations
import subprocess
from pathlib import Path

from flashsim.emit import _hold_wake_sparse, _sparse_hold_masks, emit_cpp
from flashsim.ir import AlwaysFF, Assign, Id, Module, NbAssign, Signal


def test_sparse_wake_selection_is_bounded_and_deterministic() -> None:
    dense = list(range(16))
    scattered = [i * 64 for i in range(16)]
    selected = _hold_wake_sparse({'z': dense, 'a': list(range(4096, 4112)), 'b': dense,
                                  'small': list(range(7)), 'scattered': scattered})
    assert selected == {tuple(range(4096, 4112)): (0, 1), tuple(dense): (1, 1)}
    assert _hold_wake_sparse({'a': list(range(8))}) == {tuple(range(8)): (0, 1)}
    assert _hold_wake_sparse({'a': []}) == {}
    for holds, (_index, words) in selected.items():
        assert words * 16 <= len(holds) * 2


def test_sparse_masks_preserve_exact_ids_across_summary_boundaries() -> None:
    holds = tuple(list(range(24)) + list(range(4095, 4120)) + list(range(65512, 65536)))
    masks = _sparse_hold_masks(holds)
    decoded = [word * 64 + bit for word, mask in masks for bit in range(64) if mask >> bit & 1]
    assert decoded == list(holds)
    assert all(0 < mask < 1 << 64 for _, mask in masks)
    assert masks == _sparse_hold_masks(tuple(reversed(holds)))
    assert _sparse_hold_masks((63, 63, 64)) == [(0, 1 << 63), (1, 1)]


def test_generated_sparse_helper_matches_legacy_bits_and_summaries(tmp_path: Path) -> None:
    signals = {'clock': Signal('clock', 1, 'input'), 'x': Signal('x', 32, 'input'),
               'r': Signal('r', 32, 'reg'), 'out': Signal('out', 32, 'output')}
    cpp = emit_cpp(Module('SparseProof', signals, [Assign('out', Id('r'))],
                         AlwaysFF('clock', [NbAssign('r', Id('x'))]), ['clock', 'x', 'out']))
    start = cpp.index('static inline void fs_wake(')
    end = cpp.index('static inline void fs_wake_mask(', start)
    helpers = cpp[start:end]
    assert 'const uint64_t (*m)[2]' in helpers and 'fs_wake_sparse' in helpers
    cases = [tuple(range(64)), tuple(range(63, 127)), tuple(range(4095, 4160)),
             tuple(list(range(24)) + list(range(4096, 4120)) + list(range(65512, 65536))),
             (0, 63, 64, 4095, 4096, 65535)]
    arrays = []
    calls = []
    for i, holds in enumerate(cases):
        masks = _sparse_hold_masks(holds)
        arrays.append(f'static const uint16_t ids{i}[]={{' + ','.join(map(str, holds)) + '};')
        arrays.append(f'static const uint64_t masks{i}[][2]={{' + ','.join(
            f'{{{word}u,{mask:#x}ull}}' for word, mask in masks) + '};')
        calls.append(f'check(ids{i},{len(holds)}u,masks{i},{len(masks)}u);')
    source = tmp_path / 'proof.cpp'
    source.write_text('#include <cstdint>\n#include <cstring>\n#include <cassert>\n' + helpers + '\n' + '\n'.join(arrays) + r'''
static_assert(sizeof(uint64_t[2])<=16,"sparse entry storage exceeds density assumption");
static uint64_t random_word(uint64_t &s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
static void check(const uint16_t*ids,unsigned count,const uint64_t(*m)[2],unsigned words){
 for(unsigned trial=0;trial<64;trial++){
  uint64_t a[1024]={},b[1024]={},busy[1024]={},sa[16]={},sb[16]={};unsigned na=0,nb=0;uint64_t seed=trial+9173;
  if(trial%4)for(unsigned w=0;w<1024;w++){
   if(w%7==trial%7)a[w]=random_word(seed);
   if(w%11==trial%11)busy[w]=random_word(seed);
  }
  memcpy(b,a,sizeof(a));
  for(unsigned w=0;w<1024;w++)if(a[w]|busy[w])sa[w>>6]|=1ull<<(w&63);
  memcpy(sb,sa,sizeof(sa));for(auto v:sa)na+=v!=0;nb=na;
  for(unsigned repeat=0;repeat<3;repeat++){
   fs_wake(a,sa,&na,ids,count);fs_wake_sparse(b,sb,&nb,m,words);
   assert(!memcmp(a,b,sizeof(a)));assert(!memcmp(sa,sb,sizeof(sa)));assert(na==nb);
   uint64_t expected[16]={};for(unsigned w=0;w<1024;w++)if(b[w]|busy[w])expected[w>>6]|=1ull<<(w&63);
   assert(!memcmp(sb,expected,sizeof(sb)));unsigned n=0;for(auto v:expected)n+=v!=0;assert(n==nb);
  }
 }
}
int main(){
''' + '\n'.join(calls) + '\nreturn 0;}\n')
    exe = tmp_path / 'proof'
    subprocess.run(['clang++', '-std=c++17', '-O1', '-fsanitize=address,undefined',
                    '-fno-omit-frame-pointer', str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    _check_complete_state(tmp_path)


def _check_complete_state(tmp_path: Path) -> None:
    """Differential circuits use the old ID-list path as their control."""
    import flashsim.emit as emitter
    from flashsim.ir import BinOp, Concat, Const, If, MemRead, MemWrite

    def model_cpp(n: int, sparse: bool) -> str:
        signals = {name: Signal(name, width, 'input') for name, width in
                   [('clock', 1), ('reset', 1), ('en', 1), ('a', 32), ('b', 32), ('addr', 3), ('we', 1)]}
        signals.update(out=Signal('out', 32, 'output'), m=Signal('m', 32, 'mem', depth=8),
                       wide=Signal('wide', 128, 'reg'), wide_word=Signal('wide_word', 128, 'wire'),
                       read_word=Signal('read_word', 32, 'wire'), write_gate=Signal('write_gate', 1, 'wire'))
        assignments = [Assign('read_word', MemRead('m', Id('addr')))]
        body = []
        for i in range(n):
            signals[f'r{i}'] = Signal(f'r{i}', 32, 'reg')
            name = f'stage{i}_pipeline_value'
            signals[name] = Signal(name, 32, 'wire')
            previous = Id(f'stage{i-1}_pipeline_value') if i else Id('a')
            assignments.append(Assign(name, BinOp('^', BinOp('+', previous, Id(f'r{i}')), Id('b'))))
            payload = BinOp('^', BinOp('+', Id(name), Id('read_word')), Const(i * 0x10203, 32))
            body.append(If(Id('reset'), [NbAssign(f'r{i}', Const(0, 32))],
                           [If(Id('en'), [NbAssign(f'r{i}', payload)])]))
        assignments += [Assign('wide_word', Concat(tuple(Id(f'stage{i}_pipeline_value') for i in range(4)))),
                        Assign('write_gate', BinOp('&', Id('we'), Id('en'))),
                        Assign('out', BinOp('^', Id('read_word'), Id(f'stage{n-1}_pipeline_value')))]
        body.append(If(Id('reset'), [NbAssign('wide', Const(0, 128))],
                       [If(Id('we'), [NbAssign('wide', Id('wide_word'))])]))
        module = Module('WakeProof', signals, assignments, AlwaysFF('clock', body),
                        ['clock', 'reset', 'en', 'a', 'b', 'addr', 'we', 'out'],
                        [MemWrite('m', Id('addr'), Id(f'stage{n-1}_pipeline_value'), Id('write_gate'))])
        old_limit, old_cached, old_sparse = emitter._LARGE_WRITES, emitter.cached_wires, emitter._hold_wake_sparse
        try:
            emitter._LARGE_WRITES = 8
            emitter.cached_wires = lambda mod, assigns: set(assigns)
            if not sparse:
                emitter._hold_wake_sparse = lambda holds: {}
            cpp = emitter.emit_cpp(module)
            return emitter._bulk_state_constructor(cpp, 'WakeProofDut', min_fields=1)
        finally:
            emitter._LARGE_WRITES, emitter.cached_wires, emitter._hold_wake_sparse = old_limit, old_cached, old_sparse

    for n in (160, 256):
        old = model_cpp(n, False)
        new = model_cpp(n, True)
        assert 'fs_wake_sparse(_h_need' not in old
        assert 'fs_wake_sparse(_h_need' in new
        (tmp_path / f'old{n}.h').write_text(old)
        (tmp_path / f'new{n}.h').write_text(new)
        checks = '\n'.join(f'if(a.r{i}!=b.r{i})return fail(t,"r{i}");' for i in range(n))
        program = r'''#include <cstdint>
#include <cstring>
#include <cstdio>
namespace old_model {
#include "OLD"
}
namespace new_model {
#include "NEW"
}
static uint32_t seed=0x19812347;
static uint32_t random_word(){seed^=seed<<13;seed^=seed>>17;return seed^=seed<<5;}
static int fail(unsigned t,const char*f){fprintf(stderr,"tick=%u field=%s\n",t,f);return 1;}
int main(int argc,char**){
 const bool forced=argc>1;old_model::WakeProofDut a;new_model::WakeProofDut b;
 for(unsigned t=0;t<10000;t++){
  a.reset=b.reset=(t<2||t%127==0);a.en=b.en=(random_word()%5!=0);a.we=b.we=(random_word()%3==0);
  a.addr=b.addr=random_word()&7;a.a=b.a=(t%7?random_word():0);a.b=b.b=(t%9?random_word():0);
  if(t%97>=85){a.reset=b.reset=0;a.en=b.en=0;a.we=b.we=0;a.addr=b.addr=0;a.a=b.a=0;a.b=b.b=0;}
  if(forced&&(t%17==0||t%97==96))b._force_wake_all();
  if(t%3==0){a.poke_inputs();b.poke_inputs();a.eval_out();b.eval_out();if(a.out!=b.out)return fail(t,"pre_out");}
  if(t%2){a.tick();b.tick();}else{a.poke_inputs();b.poke_inputs();a.tick_nba();b.tick_nba();}
  CHECKS
  if(memcmp(&a.wide,&b.wide,sizeof(a.wide)))return fail(t,"wide");
  if(memcmp(a.m,b.m,sizeof(a.m)))return fail(t,"memory");
  if(a._nt!=b._nt||a._chg!=b._chg||(!forced&&(a._cc!=b._cc||a._hc!=b._hc||a._hb!=b._hb)))return fail(t,"counters");
  a.eval_out();b.eval_out();if(a.out!=b.out)return fail(t,"post_out");
  if(!forced){
   if(memcmp(a._h_need,b._h_need,sizeof(a._h_need))||memcmp(a._h_busy,b._h_busy,sizeof(a._h_busy))||memcmp(a._h_any,b._h_any,sizeof(a._h_any))||a._h_words!=b._h_words)return fail(t,"hold_worklist");
   if(memcmp(a._p_need,b._p_need,sizeof(a._p_need))||memcmp(a._p_any,b._p_any,sizeof(a._p_any))||a._p_words!=b._p_words)return fail(t,"pass_worklist");
  }
 }
 return 0;
}
'''.replace('OLD', f'old{n}.h').replace('NEW', f'new{n}.h').replace('CHECKS', checks)
        source = tmp_path / f'circuit{n}.cpp'
        source.write_text(program)
        exe = tmp_path / f'circuit{n}'
        subprocess.run(['clang++', '-std=c++17', '-O1', '-fsanitize=address,undefined',
                        '-fno-omit-frame-pointer', str(source), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        subprocess.run([str(exe), 'forced'], check=True)
