"""Replay must reproduce wall-clock bounds and reject changed DMA writes."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


def test_record_replay_clock_dma_and_corruption(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    header = '''#pragma once
#include <stdint.h>
extern "C" {
typedef int (*arti_mem_read_cb)(uint64_t,uint8_t*,unsigned,uint64_t);
typedef int (*arti_mem_write_cb)(uint64_t,const uint8_t*,unsigned,uint64_t,uint64_t);
void arti_rtl_model_init();
void arti_rtl_model_set_memory_callbacks(arti_mem_read_cb,arti_mem_write_cb);
int arti_rtl_model_read(uint64_t,uint64_t*,unsigned);
int arti_rtl_model_write(uint64_t,uint64_t,unsigned);
int arti_rtl_model_check_irq(unsigned);
void arti_rtl_model_tick_n(unsigned);
void arti_rtl_model_skip_stats(uint64_t*,uint64_t*,uint64_t*,uint64_t*);
}
'''
    (tmp_path / "arti_rtl_model.h").write_text(header)
    wrapper = tmp_path / "wrapper.cpp"
    wrapper.write_text('''
static arti_mem_read_cb rd;static arti_mem_write_cb wr;
static uint64_t nt,value;
extern "C" void arti_rtl_model_init() {nt=0;value=0;}
extern "C" void arti_rtl_model_set_memory_callbacks(arti_mem_read_cb r,arti_mem_write_cb w) {rd=r;wr=w;}
extern "C" int arti_rtl_model_write(uint64_t a,uint64_t v,unsigned n) {
  uint8_t data[8]={};rd(a,data,n,7);value=v+data[0];
  data[0]=(uint8_t)value;wr(a,data,n,1,9);
  timespec begin,now;clock_gettime(CLOCK_MONOTONIC,&begin);
  do {nt++;clock_gettime(CLOCK_MONOTONIC,&now);}
  while((now.tv_sec-begin.tv_sec)*1000000000L+now.tv_nsec-begin.tv_nsec<10000);
  return 0;
}
extern "C" int arti_rtl_model_read(uint64_t,uint64_t *v,unsigned) {*v=value;return 0;}
extern "C" int arti_rtl_model_check_irq(unsigned) {return value==45;}
extern "C" void arti_rtl_model_tick_n(unsigned n) {nt+=n;}
extern "C" void arti_rtl_model_skip_stats(uint64_t *t,uint64_t *c,uint64_t *h,uint64_t *w) {
  if(t)*t=nt;if(c)*c=0;if(h)*h=0;if(w)*w=0;
}
''')
    main = tmp_path / "main.cpp"
    main.write_text('''#include "arti_rtl_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" int arti_trace_replay();
int read_mem(uint64_t,uint8_t *v,unsigned n,uint64_t) {memset(v,3,n);return 0;}
int write_mem(uint64_t,const uint8_t *v,unsigned,uint64_t,uint64_t) {return *v==45?0:-1;}
int main(int argc,char **argv) {
  if(argc>1) {setenv("ARTI_MODEL_REPLAY",argv[1],1);return arti_trace_replay();}
  arti_rtl_model_init();arti_rtl_model_set_memory_callbacks(read_mem,write_mem);
  if(arti_rtl_model_write(32,42,4))return 1;
  uint64_t v=0,t=0;arti_rtl_model_read(32,&v,4);
  if(v!=45 || !arti_rtl_model_check_irq(0))return 2;
  arti_rtl_model_tick_n(11);
  for(unsigned i=0;i<100;i++)arti_rtl_model_tick_n(1);
  arti_rtl_model_skip_stats(&t,nullptr,nullptr,nullptr);
  printf("%llu\\n",(unsigned long long)t);
}
''')
    source = Path(__file__).resolve().parents[1] / "scripts/arti_trace_model.cpp"
    binary = tmp_path / "check"
    subprocess.run([compiler,"-std=c++17","-O1","-I",str(tmp_path),
                    f'-DARTI_WRAPPER_SOURCE="{wrapper}"',str(source),str(main),
                    "-o",str(binary)],check=True)
    trace = tmp_path / "work.trace"
    env = dict(os.environ,ARTI_MODEL_TRACE=str(trace))
    recorded = subprocess.check_output([str(binary)],env=env,text=True)
    result = json.loads(subprocess.check_output([str(binary),str(trace)],text=True))
    assert result["ticks"] == int(recorded)
    assert result["calls"] == 105
    assert result["dma_reads"] == result["dma_writes"] == 1
    assert result["truncated_tail"] is False
    window_env = dict(os.environ,ARTI_REPLAY_WINDOW_START="2",
                      ARTI_REPLAY_WINDOW_CALLS="80",ARTI_REPLAY_WINDOW_REPEATS="3",
                      ARTI_REPLAY_WINDOW_PREHEAT="0")
    windows = [json.loads(line) for line in subprocess.check_output(
        [str(binary),str(trace)],env=window_env,text=True).splitlines()]
    assert len(windows) == 3
    for i,window in enumerate(windows):
        assert window["sample"] == i
        assert window["warmup_calls"] == 2
        assert window["calls"] == 80 and window["ticks"] == 88
        assert window["dma_reads"] == window["dma_writes"] == 0
    # The parent must propagate a failed child rather than reporting a
    # successful timing when a window's expected MMIO result is changed.
    raw = trace.read_bytes()
    marker = struct.pack("=6Q",6,0,45,0,0,0)
    offset = raw.index(marker)+16
    bad_window = tmp_path / "bad-window.trace"
    bad_window.write_bytes(raw[:offset]+struct.pack("=Q",44)+raw[offset+8:])
    failed = subprocess.run([str(binary),str(bad_window)],env=window_env,
                            capture_output=True,text=True)
    assert failed.returncode != 0 and "event or data differs" in failed.stderr
    # A process killed while writing a block leaves preceding complete calls
    # replayable. The report explicitly identifies the incomplete tail.
    partial = tmp_path / "partial.trace"
    partial.write_bytes(raw[:-1])
    result = json.loads(subprocess.check_output([str(binary),str(partial)],text=True))
    assert result["calls"] == 104 and result["truncated_tail"] is True
    # Locate the unique DMA write record, then corrupt its payload.
    marker = struct.pack("=6Q",9,32,4,1,9,0)
    offset = raw.index(marker)+48
    corrupt = tmp_path / "corrupt.trace"
    corrupt.write_bytes(raw[:offset]+bytes([raw[offset]^1])+raw[offset+1:])
    result = subprocess.run([str(binary),str(corrupt)],capture_output=True,text=True)
    assert result.returncode != 0
    assert "event or data differs" in result.stderr
