"""Shared dispatch must preserve cache writes, chunk order and same-tick wakes."""
from __future__ import annotations

import subprocess

from flashsim import emit


def test_uniform_guards_match_wrappers_with_gaps_chunks_fallback_and_wakes(tmp_path, monkeypatch):
    # Gapped region IDs and three bitmap words catch accidental use of a
    # region ID as a dispatch index. Include a three-chunk ready region and
    # one lazy fallback in the middle of the worklist.
    count = 130
    region_ids = [2 * i + 1 for i in range(count)]
    passes = {r: i for i, r in enumerate(region_ids)}
    regions = [[] for _ in range(max(region_ids) + 1)]
    parts = {}
    for i, r in enumerate(region_ids):
        regions[r] = [f"w{r}_{k}" for k in range(5 if i == 63 else 1)]
        parts[regions[r][0]] = 2 * i
    pages = [0] * len(regions)
    for i, r in enumerate(region_ids):
        pages[r] = 2 * i + 1
    monkeypatch.setattr(emit, "_PASS_PAGE", pages)
    monkeypatch.setattr(emit, "_EAGER_CHUNK", 2)
    drained = set(region_ids) - {region_ids[64]}

    def source(cls, uniform):
        monkeypatch.setenv("FLASHSIM_UNIFORM_PASSES", str(int(uniform)))
        table = []
        body = emit._emit_pass_tables(table, cls, passes, drained, regions, parts)
        emit._emit_pass_dispatch(table, 3, count, body)
        fields = []
        methods = []
        for i, r in enumerate(region_ids):
            fields.append(f"uint32_t _eg{r}=0,_ed{r}=0;uint8_t _up{r}=0;")
            nchunks = 3 if i == 63 else 1
            for k in range(nchunks):
                # The first pass wakes later bits in its current word, the
                # next word and the third word from an initially empty summary.
                wake = "wake(63);wake(64);wake(129);" if i == k == 0 else ""
                methods.append(f"void _eager{r}_{k}(){{events.push_back({10*i+k});{wake}}}")
            dependency = "events.push_back(99999);" if i == 64 else ""
            methods.append(f"""void _pn{r}(){{
              {dependency}
              if(_eg{r}==_pg[{2*i}])return;
              _eg{r}=_pg[{2*i}];
              if(_ed{r}==_pg[{2*i+1}]&&!_up{r})return;
              _ed{r}=_pg[{2*i+1}];_up{r}=0;
              """ + "".join(f"_eager{r}_{k}();" for k in range(nchunks)) + "}")
        return f"struct {cls}{{\n" + "\n".join(fields) + r"""
          uint32_t _pg[260]={};uint64_t _p_need[3]={},_p_any[1]={};unsigned _p_words=0;
          std::vector<unsigned> events;
          void wake(unsigned i){unsigned w=i>>6;_p_need[w]|=1ull<<(i&63);
            if(!_p_any[0])++_p_words;_p_any[0]|=1ull<<w;}
        """ + "\n".join(methods) + "\nvoid drain(){\n" + "\n".join(table) + "\n}};\n"

    legacy = source("Legacy", False)
    shared = source("Shared", True)
    assert "if (!guard.generation)" in shared
    comparisons = "".join(f"assert(a._eg{r}==b._eg{r}&&a._ed{r}==b._ed{r}&&a._up{r}==b._up{r});" for r in region_ids)
    assignments = []
    for i, r in enumerate(region_ids):
        assignments.append(f"""
          a._eg{r}=b._eg{r}=trial%5==0?generation:prior;
          a._ed{r}=b._ed{r}=(trial%5==1||trial%5==2)?generation:prior;
          a._up{r}=b._up{r}=trial%5==2;
        """)
    program = r"""#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>
""" + legacy + shared + r"""
int main(){
 for(unsigned trial=0;trial<200;trial++){
  Legacy a;Shared b;
  // Include wraparound: guards compare equality, not generation ordering.
  uint32_t generation=trial%7==0?0:trial%7==1?UINT32_MAX:trial+1;
  uint32_t prior=generation-1;
  for(unsigned i=0;i<260;i++)a._pg[i]=b._pg[i]=generation;
""" + "\n".join(assignments) + r"""
  a.wake(0);b.wake(0);
  if(trial%2)for(unsigned i=1;i<130;i++)if(i%3==trial%3){a.wake(i);b.wake(i);}
  a.drain();b.drain();assert(a.events==b.events);
  assert(!memcmp(a._pg,b._pg,sizeof(a._pg)));
  assert(!memcmp(a._p_need,b._p_need,sizeof(a._p_need)));
  assert(!memcmp(a._p_any,b._p_any,sizeof(a._p_any)));assert(a._p_words==b._p_words);
""" + comparisons + r"""
  // Repeating a wake must preserve the already updated generation cache,
  // including the data-unchanged return and unconditional lazy fallback.
  for(unsigned i:{0u,63u,64u,129u}){a.wake(i);b.wake(i);}
  a.drain();b.drain();assert(a.events==b.events);
""" + comparisons + "\n }\n}\n"
    path = tmp_path / "proof.cpp"
    path.write_text(program)
    exe = tmp_path / "proof"
    subprocess.run(["clang++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer", str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_uniform_disabled_and_all_lazy_keep_legacy_dispatch(monkeypatch):
    monkeypatch.setattr(emit, "_PASS_PAGE", [1])
    for enabled, drained in [("0", {0}), ("1", set())]:
        monkeypatch.setenv("FLASHSIM_UNIFORM_PASSES", enabled)
        lines = []
        body = emit._emit_pass_tables(lines, "Proof", {0: 0}, drained, [["w"]], {"w": 0})
        assert body is None
        assert "&Proof::_pn0" in "\n".join(lines)
        assert "PassGuard" not in "\n".join(lines)
