// Included around a generated ARTI wrapper by build_arti_trace.py.
// Replay supplies the original DMA inputs and wall clock observations, so
// wall-clock-bounded settle loops execute exactly the recorded tick count.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <chrono>
#include "arti_rtl_model.h"

namespace arti_trace {
enum Tag : uint64_t { Init=1, Read, Write, Irq, Tick, Result, Clock, DmaRead, DmaWrite };
struct Event { uint64_t tag, a, b, c, d, e; };
static FILE *file;
static bool replay, opened;
static std::vector<uint8_t> block;
static size_t cursor;
static uint64_t calls, reads, writes, bytes;
static arti_mem_read_cb read_cb;
static arti_mem_write_cb write_cb;
static const char *knobs[] = {"ARTI_MODEL_MMIO_ADVANCE", "ARTI_MODEL_IDLE_GRACE",
  "ARTI_MODEL_SETTLE_MIN", "ARTI_MODEL_IRQ_PUMP", "ARTI_MODEL_IRQ_PUMP_NS",
  "ARTI_MODEL_HOLD_BOOST", "ARTI_MODEL_CHG_TAIL", "ARTI_MODEL_SETTLE_NS"};
[[noreturn]] static void fail(const char *why) {
  fprintf(stderr,"ARTI trace mismatch at call %llu byte %zu: %s\n",
          (unsigned long long)calls,cursor,why); abort();
}
static void open() {
  if (opened) return;
  opened=true;
  const char *path=getenv("ARTI_MODEL_REPLAY");
  replay=path && *path;
  if (!replay) path=getenv("ARTI_MODEL_TRACE");
  if (!path || !*path) return;
  file=fopen(path,replay?"rb":"wb");
  if (!file) {perror(path);abort();}
  char magic[16]="ARTI_TRACE_v1";
  if (replay) {
    char got[16];
    if (fread(got,1,16,file)!=16 || memcmp(got,magic,16)) fail("file header");
  } else fwrite(magic,1,16,file);
  for (const char *name:knobs) {
    char value[64]={};
    if (replay) {
      if (fread(value,1,sizeof(value),file)!=sizeof(value)) fail("environment header");
      value[63]=0;
      if (*value) setenv(name,value,1); else unsetenv(name);
    } else {
      const char *v=getenv(name);
      if (v && strlen(v)>=sizeof(value)) fail("environment value too long");
      if(v) strcpy(value,v);
      fwrite(value,1,sizeof(value),file);
    }
  }
  setenv("ARTI_MODEL_STATS","0",1);
}
static void transfer(void *p,size_t n,bool compare=false) {
  if (!file) return;
  if (replay) {
    if (cursor+n>block.size()) fail("unexpected event or payload");
    if (compare && memcmp(p,block.data()+cursor,n)) fail("event or data differs");
    if (!compare) memcpy(p,block.data()+cursor,n);
    cursor+=n;
  } else {
    auto *s=static_cast<uint8_t *>(p);block.insert(block.end(),s,s+n);
  }
}
static void event(uint64_t tag,uint64_t a=0,uint64_t b=0,uint64_t c=0,
                  uint64_t d=0,uint64_t e=0) {
  Event v{tag,a,b,c,d,e};transfer(&v,sizeof(v),true);
}
static void begin(uint64_t tag,uint64_t a=0,uint64_t b=0,uint64_t c=0) {
  open();
  if (!replay) block.clear();
  event(tag,a,b,c);
}
static void end(int status=0,uint64_t data=0) {
  event(Result,(uint64_t)(int64_t)status,data);
  if (file) {
    if(replay) {if(cursor!=block.size()) fail("unconsumed events");}
    else {
      uint64_t n=block.size();
      if(fwrite(&n,1,sizeof(n),file)!=sizeof(n) ||
         fwrite(block.data(),1,n,file)!=n) fail("write failed");
    }
  }
  calls++;
}
static int clock(clockid_t id,struct timespec *ts) {
  if (!file) return ::clock_gettime(id,ts);
  Event v{Clock,(uint64_t)id,0,0,0,0};
  if(replay) {
    transfer(&v,sizeof(v));
    if(v.tag!=Clock || v.a!=(uint64_t)id) fail("clock request");
    ts->tv_sec=v.c;ts->tv_nsec=v.d;return (int)(int64_t)v.b;
  }
  int r=::clock_gettime(id,ts);
  v.b=(uint64_t)(int64_t)r;v.c=ts->tv_sec;v.d=ts->tv_nsec;
  transfer(&v,sizeof(v));return r;
}
static int dma_read(uint64_t addr,uint8_t *data,unsigned n,uint64_t id) {
  event(DmaRead,addr,n,id);reads++;bytes+=n;
  int status=replay?0:(read_cb?read_cb(addr,data,n,id):-1);
  Event result{Result,(uint64_t)(int64_t)status,0,0,0,0};
  transfer(&result,sizeof(result));
  if(result.tag!=Result)fail("DMA read result");
  transfer(data,n);return (int)(int64_t)result.a;
}
static int dma_write(uint64_t addr,const uint8_t *data,unsigned n,uint64_t mask,uint64_t id) {
  event(DmaWrite,addr,n,mask,id);writes++;bytes+=n;
  transfer(const_cast<uint8_t *>(data),n,true);
  int status=replay?0:(write_cb?write_cb(addr,data,n,mask,id):-1);
  Event result{Result,(uint64_t)(int64_t)status,0,0,0,0};
  transfer(&result,sizeof(result));
  if(result.tag!=Result)fail("DMA write result");
  return (int)(int64_t)result.a;
}
}

#define clock_gettime arti_trace::clock
#define arti_rtl_model_init trace_real_init
#define arti_rtl_model_read trace_real_read
#define arti_rtl_model_write trace_real_write
#define arti_rtl_model_check_irq trace_real_irq
#define arti_rtl_model_tick_n trace_real_tick_n
#define arti_rtl_model_skip_stats trace_real_stats
#define arti_rtl_model_set_memory_callbacks trace_real_callbacks
#include ARTI_WRAPPER_SOURCE
#undef clock_gettime
#undef arti_rtl_model_init
#undef arti_rtl_model_read
#undef arti_rtl_model_write
#undef arti_rtl_model_check_irq
#undef arti_rtl_model_tick_n
#undef arti_rtl_model_skip_stats
#undef arti_rtl_model_set_memory_callbacks

extern "C" void arti_rtl_model_init() {
  arti_trace::begin(arti_trace::Init);trace_real_init();arti_trace::end();
}
extern "C" void arti_rtl_model_set_memory_callbacks(arti_mem_read_cb r,arti_mem_write_cb w) {
  arti_trace::read_cb=r;arti_trace::write_cb=w;
  trace_real_callbacks(arti_trace::dma_read,arti_trace::dma_write);
}
extern "C" int arti_rtl_model_read(uint64_t a,uint64_t *v,unsigned n) {
  arti_trace::begin(arti_trace::Read,a,n);int r=trace_real_read(a,v,n);
  arti_trace::end(r,r==0?*v:0);return r;
}
extern "C" int arti_rtl_model_write(uint64_t a,uint64_t v,unsigned n) {
  arti_trace::begin(arti_trace::Write,a,v,n);int r=trace_real_write(a,v,n);
  arti_trace::end(r);return r;
}
extern "C" int arti_rtl_model_check_irq(unsigned i) {
  arti_trace::begin(arti_trace::Irq,i);int r=trace_real_irq(i);arti_trace::end(r);return r;
}
extern "C" void arti_rtl_model_tick_n(unsigned n) {
  arti_trace::begin(arti_trace::Tick,n);trace_real_tick_n(n);arti_trace::end();
}
extern "C" void arti_rtl_model_skip_stats(uint64_t *t,uint64_t *c,uint64_t *h,uint64_t *w) {
  trace_real_stats(t,c,h,w);
}
extern "C" int arti_trace_replay() {
  using namespace arti_trace;
  open();if(!file || !replay)fail("ARTI_MODEL_REPLAY must name a trace");
  arti_rtl_model_set_memory_callbacks(nullptr,nullptr);
  const auto start=std::chrono::steady_clock::now();
  bool truncated=false;
  while(true) {
    uint64_t n=0;
    size_t got=fread(&n,1,sizeof(n),file);
    if(!got)break;
    if(got!=sizeof(n)) {truncated=true;break;}
    if(n<2*sizeof(Event) || n>UINT64_C(1)<<30)fail("invalid block size");
    block.resize(n);cursor=0;
    if(fread(block.data(),1,n,file)!=n) {truncated=true;break;}
    Event v;memcpy(&v,block.data(),sizeof(v));
    uint64_t value=0;
    switch(v.tag) {
      case Init:arti_rtl_model_init();break;
      case Read:arti_rtl_model_read(v.a,&value,v.b);break;
      case Write:arti_rtl_model_write(v.a,v.b,v.c);break;
      case Irq:arti_rtl_model_check_irq(v.a);break;
      case Tick:arti_rtl_model_tick_n(v.a);break;
      default:fail("unknown API call");
    }
  }
  uint64_t ticks=0,commits=0,holds=0,wrote=0;
  trace_real_stats(&ticks,&commits,&holds,&wrote);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  printf("{\"calls\":%llu,\"ticks\":%llu,\"commits\":%llu,\"holds\":%llu,"
         "\"dma_reads\":%llu,\"dma_writes\":%llu,\"dma_bytes\":%llu,"
         "\"seconds\":%.9f,\"truncated_tail\":%s}\n",
         (unsigned long long)calls,(unsigned long long)ticks,(unsigned long long)commits,
         (unsigned long long)holds,(unsigned long long)reads,(unsigned long long)writes,
         (unsigned long long)bytes,seconds,truncated?"true":"false");
  return calls?0:1;
}
