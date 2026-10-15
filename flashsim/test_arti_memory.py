"""Exercise the generated AXI callback adapter without building a GPU model."""
import ast
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("beat_bytes", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("callback_bytes", [8, 64])
def test_memory_callbacks_masks_faults_and_sparse_writes(tmp_path, beat_bytes, callback_bytes):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    tree = ast.parse(Path(__file__).with_name("arti_model.py").read_text())
    template = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SYSTEM_SOURCE"
                            for t in node.targets))
    capture = "static void mem_capture(void)" + template.split(
        "static void mem_capture(void)", 1)[1].split(
        "\nstatic unsigned g_hold_boost;", 1)[0]
    drive = template.split("template <typename T, typename U>", 1)[1].split(
        "static void mem_capture(void)", 1)[0]
    drive = "template <typename T, typename U>" + drive
    callback_config = "#ifndef ARTI_MODEL_WRITE_CALLBACK_BYTES" + template.split(
        "#ifndef ARTI_MODEL_WRITE_CALLBACK_BYTES", 1)[1].split("using ArtiAxiData", 1)[0]
    # Compile the production functions verbatim against a narrow DUT stub.
    prefix = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <type_traits>
static constexpr unsigned M_AXI_BYTES = BEAT_BYTES;
using ArtiAxiData = std::array<uint8_t, M_AXI_BYTES>;
using Strobe = std::conditional_t<(M_AXI_BYTES <= 8), uint8_t,
               std::conditional_t<(M_AXI_BYTES <= 32), uint32_t, uint64_t>>;
struct Dut {
  bool io_m_axi_awvalid, io_m_axi_awready, io_m_axi_wvalid, io_m_axi_wready;
  bool io_m_axi_wlast, io_m_axi_arvalid, io_m_axi_arready;
  uint64_t io_m_axi_awaddr, io_m_axi_awid, io_m_axi_araddr, io_m_axi_arid;
  unsigned io_m_axi_awsize, io_m_axi_arsize, io_m_axi_arlen;
  ArtiAxiData io_m_axi_wdata;
  Strobe io_m_axi_wstrb;
  bool io_m_axi_bvalid, io_m_axi_rvalid, io_m_axi_rlast;
  unsigned io_m_axi_bresp, io_m_axi_rresp;
  uint64_t io_m_axi_bid, io_m_axi_rid;
  ArtiAxiData io_m_axi_rdata;
} dut;
static Dut *g_rtl = &dut;
struct ArtiAxiWriteBeat {ArtiAxiData data; uint64_t strb; bool last;};
struct ArtiAxiWriteResp {uint64_t id; uint8_t resp;};
struct ArtiAxiReadBeat {ArtiAxiData data; uint64_t id; uint8_t resp; bool last;};
static std::deque<ArtiAxiWriteBeat> g_wbeats;
static std::deque<ArtiAxiWriteResp> g_bresp;
static std::deque<ArtiAxiReadBeat> g_rresp;
static bool g_aw_active;
static uint64_t g_aw_addr, g_aw_id;
static unsigned g_aw_size, g_aw_beat, g_arti_idle;
static int g_write_status;
using WriteCb = int (*)(uint64_t, const uint8_t *, unsigned, uint64_t, uint64_t);
using ReadCb = int (*)(uint64_t, uint8_t *, unsigned, uint64_t);
static unsigned calls, fail_call;
static std::array<uint8_t, 1024> memory;
static int write_cb(uint64_t addr, const uint8_t *data, unsigned n,
                    uint64_t mask, uint64_t id) {
  assert(n && n <= ARTI_MODEL_WRITE_CALLBACK_BYTES && addr + n <= memory.size() && id == 7);
  assert(mask && (n == 64 || !(mask >> n)));
  ++calls;
  if (calls == fail_call) return -1;
  for (unsigned i = 0; i < n; ++i)
    if (mask & (UINT64_C(1) << i)) memory[addr + i] = data[i];
  return 0;
}
static int read_cb(uint64_t addr, uint8_t *data, unsigned n, uint64_t id) {
  assert(id >= 9 && id < 12 && addr + n <= memory.size());
  memcpy(data, memory.data() + addr, n);
  return fail_call ? -1 : 0;
}
static WriteCb g_mem_write_cb = write_cb;
static ReadCb g_mem_read_cb = read_cb;
"""
    main = r"""
static unsigned log2(unsigned n) {unsigned r=0; while(n>1) {n>>=1; ++r;} return r;}
static uint64_t all_strobes() {return UINT64_MAX >> (64 - M_AXI_BYTES);}
static void reset() {
  dut = {}; g_wbeats.clear(); g_bresp.clear(); g_rresp.clear();
  g_aw_active=false; calls=fail_call=0; memory.fill(0xaa);
  g_mem_write_cb=write_cb;
}
static void write(unsigned size, unsigned lane, uint64_t mask, bool last=true) {
  dut.io_m_axi_awvalid=dut.io_m_axi_awready=true;
  dut.io_m_axi_awaddr=256+lane; dut.io_m_axi_awid=7;
  dut.io_m_axi_awsize=log2(size);
  dut.io_m_axi_wvalid=dut.io_m_axi_wready=true;
  dut.io_m_axi_wlast=last; dut.io_m_axi_wstrb=mask;
  for(unsigned i=0;i<M_AXI_BYTES;++i) dut.io_m_axi_wdata[i]=i+1;
  mem_capture();
  dut.io_m_axi_awvalid=dut.io_m_axi_wvalid=false;
}
int main() {
  reset(); write(M_AXI_BYTES,0,all_strobes());
  assert(calls==(M_AXI_BYTES+WRITE_CALLBACK_BYTES-1)/WRITE_CALLBACK_BYTES &&
         g_bresp.size()==1 && g_bresp.front().resp==0);
  for(unsigned i=0;i<M_AXI_BYTES;++i) assert(memory[256+i]==i+1);
  reset(); unsigned sparse=M_AXI_BYTES-1;
  write(M_AXI_BYTES,0,UINT64_C(1)<<sparse);
  assert(calls==1 && g_bresp.front().resp==0);
  for(unsigned i=0;i<M_AXI_BYTES;++i) assert(memory[256+i]==(i==sparse?i+1:0xaa));
  reset(); write(4,M_AXI_BYTES-4,UINT64_C(15)<<(M_AXI_BYTES-4));
  assert(calls==1 && g_bresp.front().resp==0);
  for(unsigned i=0;i<M_AXI_BYTES;++i) assert(memory[256+i]==(i>=M_AXI_BYTES-4?i+1:0xaa));
  // Preserve lane wrap without reading beyond the host data array. A failure
  // in the first segment must survive a successful second callback.
  if(M_AXI_BYTES>=8) {
    reset(); write(8,M_AXI_BYTES-4,all_strobes());
    assert(calls==2 && g_bresp.front().resp==0);
    for(unsigned i=0;i<4;++i) {
      assert(memory[256+M_AXI_BYTES-4+i]==M_AXI_BYTES-3+i);
      assert(memory[256+M_AXI_BYTES+i]==i+1);
    }
    assert(memory[256+M_AXI_BYTES-5]==0xaa && memory[256+M_AXI_BYTES+4]==0xaa);
    reset(); fail_call=1; write(8,M_AXI_BYTES-4,all_strobes());
    assert(calls==2 && g_bresp.front().resp==2);
    for(unsigned i=0;i<4;++i) {
      assert(memory[256+M_AXI_BYTES-4+i]==0xaa);
      assert(memory[256+M_AXI_BYTES+i]==i+1);
    }
  }
  reset(); g_mem_write_cb=nullptr; write(M_AXI_BYTES,0,0);
  assert(calls==0 && g_bresp.front().resp==0);
  reset(); g_mem_write_cb=nullptr; write(M_AXI_BYTES,0,1);
  assert(g_bresp.front().resp==2);
  reset(); fail_call=1; write(M_AXI_BYTES,0,all_strobes());
  assert(g_bresp.front().resp==2); // Later successful chunks cannot erase failure.
  reset(); fail_call=1; write(M_AXI_BYTES,0,all_strobes(),false);
  assert(g_bresp.empty());
  dut.io_m_axi_wvalid=true; dut.io_m_axi_wlast=true;
  mem_capture(); dut.io_m_axi_wvalid=false;
  assert(g_bresp.front().resp==2); // Nor can later successful burst beats.
  reset();
  dut.io_m_axi_wvalid=dut.io_m_axi_wready=true; dut.io_m_axi_wlast=true;
  dut.io_m_axi_wstrb=1; dut.io_m_axi_wdata[0]=42;
  mem_capture(); dut.io_m_axi_wvalid=false;
  assert(calls==0 && g_wbeats.size()==1);
  dut.io_m_axi_awvalid=dut.io_m_axi_awready=true;
  dut.io_m_axi_awaddr=256; dut.io_m_axi_awid=7; dut.io_m_axi_awsize=2;
  mem_capture(); assert(calls==1 && memory[256]==42 && g_bresp.front().resp==0);
  reset(); memory[256+M_AXI_BYTES-4]=73;
  dut.io_m_axi_arvalid=dut.io_m_axi_arready=true;
  dut.io_m_axi_araddr=256+M_AXI_BYTES-4; dut.io_m_axi_arid=9;
  dut.io_m_axi_arsize=2; dut.io_m_axi_arlen=0;
  mem_capture(); assert(g_rresp.size()==1 && g_rresp.front().last && g_rresp.front().resp==0);
  for(unsigned i=0;i<M_AXI_BYTES;++i)
    assert(g_rresp.front().data[i]==(i==M_AXI_BYTES-4?73:(i>M_AXI_BYTES-4?0xaa:0)));
  // Multiple addresses may arrive before any read response is consumed.
  reset();
  for(unsigned j=0;j<3;++j) {
    memory[256+j*M_AXI_BYTES]=17+j;
    dut.io_m_axi_arvalid=true;dut.io_m_axi_araddr=256+j*M_AXI_BYTES;
    dut.io_m_axi_arid=9+j;dut.io_m_axi_arsize=log2(M_AXI_BYTES);dut.io_m_axi_arlen=0;
    mem_drive(); assert(dut.io_m_axi_arready);mem_capture();
  }
  dut.io_m_axi_arvalid=false;
  for(unsigned j=0;j<3;++j) {
    for(unsigned stall=0;stall<4;++stall) {
      mem_drive();assert(dut.io_m_axi_rvalid && dut.io_m_axi_rid==9+j);
      assert(dut.io_m_axi_rdata[0]==17+j && dut.io_m_axi_rlast);
    }
    g_rresp.pop_front();
  }
  mem_drive();assert(!dut.io_m_axi_rvalid);
  // Admission remains bounded even if the consumer never drains R.
  for(unsigned j=0;j<64;++j)g_rresp.push_back({});
  mem_drive();assert(!dut.io_m_axi_arready);
  g_rresp.pop_front();mem_drive();assert(dut.io_m_axi_arready);
}
"""
    source = tmp_path / "check.cpp"
    prefix = prefix.replace("using ArtiAxiData =", callback_config + "using ArtiAxiData =", 1)
    source.write_text(prefix + drive + capture + main)
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=undefined",
                    "-fno-sanitize-recover=undefined", f"-DBEAT_BYTES={beat_bytes}",
                    f"-DARTI_MODEL_WRITE_CALLBACK_BYTES={callback_bytes}",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
