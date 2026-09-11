"""ARTI-compatible C API on top of a FlashSim `GpuHostAxiDut`.

Same entry points QEMU's embedded `arti-rtl` device calls:

    void arti_rtl_model_init(void);
    int  arti_rtl_model_write(uint64_t addr, uint64_t data, unsigned size);
    int  arti_rtl_model_read(uint64_t addr, uint64_t *data, unsigned size);
    int  arti_rtl_model_check_irq(unsigned index);

`tick()` is FlashSim's single-thread posedge. Combinational AXI ready/valid
are sampled with `eval_*` after `poke_inputs`, matching Verilator `eval()`.
"""

from __future__ import annotations

from pathlib import Path

from flashsim.compile import compile_file


HEADER = """\
#ifndef ARTI_RTL_MODEL_H
#define ARTI_RTL_MODEL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void arti_rtl_model_init(void);
int arti_rtl_model_write(uint64_t addr, uint64_t data, unsigned size);
int arti_rtl_model_read(uint64_t addr, uint64_t *data, unsigned size);
typedef int (*arti_mem_read_cb)(uint64_t addr, uint8_t *data, unsigned size, uint64_t transaction_id);
typedef int (*arti_mem_write_cb)(uint64_t addr, const uint8_t *data, unsigned size, uint64_t byte_mask, uint64_t transaction_id);
void arti_rtl_model_set_memory_callbacks(arti_mem_read_cb read_cb, arti_mem_write_cb write_cb);
int arti_rtl_model_check_irq(unsigned index);
#ifdef __cplusplus
}
#endif
#endif
"""

SOURCE = r'''
// FlashSim ARTI backend for GpuHostAxi.
#include "dut.h"
#include "arti_rtl_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static GpuHostAxiDut *g_rtl = nullptr;
static arti_mem_read_cb g_mem_read_cb = nullptr;
static arti_mem_write_cb g_mem_write_cb = nullptr;
static int g_arti_debug = -1;
static unsigned g_arti_idle = 0;

static constexpr unsigned TIMEOUT_CYCLES = 1000;

#ifndef ARTI_MODEL_MMIO_ADVANCE_CYCLES
#define ARTI_MODEL_MMIO_ADVANCE_CYCLES 500000
#ifndef ARTI_MODEL_IDLE_GRACE
#define ARTI_MODEL_IDLE_GRACE 20000
#endif
#endif

extern "C" void arti_rtl_model_set_memory_callbacks(
    arti_mem_read_cb read_cb, arti_mem_write_cb write_cb)
{
  g_mem_read_cb = read_cb;
  g_mem_write_cb = write_cb;
}

static void combo(void)
{
  g_rtl->poke_inputs();
  g_rtl->eval_io_s_axi_awready();
  g_rtl->eval_io_s_axi_wready();
  g_rtl->eval_io_s_axi_bvalid();
  g_rtl->eval_io_s_axi_bresp();
  g_rtl->eval_io_s_axi_arready();
  g_rtl->eval_io_s_axi_rvalid();
  g_rtl->eval_io_s_axi_rdata();
  g_rtl->eval_io_s_axi_rlast();
  g_rtl->eval_io_s_axi_rresp();
  g_rtl->eval_io_m_irq();
}

static void idle(void)
{
  g_rtl->io_s_axi_awvalid = 0;
  g_rtl->io_s_axi_wvalid = 0;
  g_rtl->io_s_axi_bready = 0;
  g_rtl->io_s_axi_arvalid = 0;
  g_rtl->io_s_axi_rready = 0;
  g_rtl->io_cbMem_req_ready = 1;
  g_rtl->io_fbMem_req_ready = 1;
  g_rtl->io_kernelMemReq_ready = 1;
  g_rtl->io_kernelWordMemReq_ready = 1;
  g_rtl->io_kernelL1InvalidateDone_ready = 1;
  g_rtl->io_kernelGlobalAtomicRequest_ready = 1;
  g_rtl->io_texMem_req_ready = 1;
  g_rtl->io_cbMem_resp_valid = 0;
  g_rtl->io_fbMem_resp_valid = 0;
  g_rtl->io_kernelMemResp_valid = 0;
  g_rtl->io_kernelWordMemResp_valid = 0;
  g_rtl->io_kernelL1Invalidate_valid = 0;
  g_rtl->io_kernelGlobalAtomicResponse_valid = 0;
  g_rtl->io_texMem_resp_valid = 0;
}

static void tick(void)
{
  g_rtl->tick();
  combo();
  g_arti_idle++;
}

static void arti_model_settle(void)
{
  if (g_arti_debug < 0)
    g_arti_debug = getenv("ARTI_MODEL_DEBUG") ? 1 : 0;
  g_arti_idle = 0;
  for (unsigned i = 0; i < ARTI_MODEL_MMIO_ADVANCE_CYCLES; i++) {
    tick();
    if (g_rtl->io_m_irq)
      break;
    if (g_arti_idle > ARTI_MODEL_IDLE_GRACE)
      break;
  }
  if (g_arti_debug)
    fprintf(stderr, "[artidbg] settle end: irq=%u idle=%u\n",
            (unsigned)g_rtl->io_m_irq, g_arti_idle);
}

extern "C" void arti_rtl_model_init(void)
{
  if (g_rtl)
    return;
  g_rtl = new GpuHostAxiDut();
  idle();
  g_rtl->reset = 1;
  g_rtl->io_s_axi_aresetn = 0;
  combo();
  for (int i = 0; i < 8; i++)
    tick();
  g_rtl->reset = 0;
  g_rtl->io_s_axi_aresetn = 1;
  idle();
  combo();
  tick();
  tick();
}

extern "C" int arti_rtl_model_write(uint64_t addr, uint64_t data, unsigned size)
{
  if (!g_rtl || size == 0 || size > 4)
    return -1;
  uint32_t word = (uint32_t)data;
  uint32_t addr_val = (uint32_t)(addr & 0xffffffffu);
  uint8_t wstrb = (uint8_t)(((1u << size) - 1u) << (addr_val & 3u));
  idle();
  g_rtl->io_s_axi_awaddr = addr_val;
  g_rtl->io_s_axi_awlen = 0;
  g_rtl->io_s_axi_awsize = 2;
  g_rtl->io_s_axi_awburst = 1;
  g_rtl->io_s_axi_wdata = word;
  g_rtl->io_s_axi_wlast = 1;
  g_rtl->io_s_axi_wstrb = wstrb;
  g_rtl->io_s_axi_awvalid = 1;
  g_rtl->io_s_axi_wvalid = 1;
  combo();
  int saw_aw = 0, saw_w = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_awready)
      saw_aw = 1;
    if (g_rtl->io_s_axi_wready)
      saw_w = 1;
    tick();
    if (saw_aw && saw_w)
      break;
  }
  if (!saw_aw || !saw_w)
    return -1;
  idle();
  g_rtl->io_s_axi_bready = 1;
  combo();
  int saw_b = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_bvalid) {
      tick();
      saw_b = 1;
      break;
    }
    tick();
  }
  if (!saw_b)
    return -1;
  idle();
  combo();
  arti_model_settle();
  return 0;
}

extern "C" int arti_rtl_model_read(uint64_t addr, uint64_t *data, unsigned size)
{
  if (!g_rtl || !data || size == 0 || size > 4)
    return -1;
  uint32_t addr_val = (uint32_t)(addr & 0xffffffffu);
  idle();
  g_rtl->io_s_axi_araddr = addr_val;
  g_rtl->io_s_axi_arlen = 0;
  g_rtl->io_s_axi_arsize = 2;
  g_rtl->io_s_axi_arburst = 1;
  g_rtl->io_s_axi_arvalid = 1;
  combo();
  int saw_ar = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_arready) {
      tick();
      saw_ar = 1;
      break;
    }
    tick();
  }
  if (!saw_ar)
    return -1;
  idle();
  g_rtl->io_s_axi_rready = 1;
  combo();
  uint32_t rdata_val = 0;
  int saw_r = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_rvalid) {
      rdata_val = g_rtl->io_s_axi_rdata;
      saw_r = 1;
      break;
    }
    tick();
  }
  if (!saw_r)
    return -1;
  tick();
  idle();
  combo();
  *data = rdata_val;
  arti_model_settle();
  return 0;
}

extern "C" int arti_rtl_model_check_irq(unsigned index)
{
  if (!g_rtl || index != 0)
    return 0;
  tick();
  return g_rtl->io_m_irq ? 1 : 0;
}
'''

SMOKE = r'''
#include "arti_rtl_model.h"
#include <cstdio>
#include <cstdint>

static int expect_eq(const char *name, uint64_t got, uint64_t want)
{
  std::printf("%s=%08llx\n", name, (unsigned long long)got);
  if (got != want) {
    std::fprintf(stderr, "FAIL %s: got 0x%llx want 0x%llx\n",
                 name, (unsigned long long)got, (unsigned long long)want);
    return 1;
  }
  return 0;
}

int main()
{
  arti_rtl_model_init();
  int fail = 0;
  uint64_t data = 0;
  if (arti_rtl_model_read(0x00, &data, 4) != 0)
    return 2;
  fail |= expect_eq("id", data, 0x47550001ull);
  if (arti_rtl_model_write(0x18, 0x20000000ull, 4) != 0)
    return 3;
  data = 0;
  if (arti_rtl_model_read(0x18, &data, 4) != 0)
    return 4;
  fail |= expect_eq("color_base", data, 0x20000000ull);
  int irq = arti_rtl_model_check_irq(0);
  std::printf("irq=%d\n", irq);
  if (irq != 0) {
    std::fprintf(stderr, "FAIL irq: got %d want 0\n", irq);
    fail = 1;
  }
  if (fail)
    return 1;
  std::printf("PASS\n");
  return 0;
}
'''


def write_arti_model(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "arti_rtl_model.h"
    source = out_dir / "arti_rtl_model.cpp"
    smoke = out_dir / "arti_smoke.cpp"
    header.write_text(HEADER)
    source.write_text(SOURCE)
    smoke.write_text(SMOKE)
    return header, source, smoke


BUILD_EMBEDDED = r'''#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QEMU_SRC=${QEMU_SRC:?must be set}
ARTI_WORK="${ARTI_WORK:-$(cd "$SCRIPT_DIR/../../.." && pwd)/arti-work}"
QEMU_BUILD=${QEMU_BUILD:-$ARTI_WORK/qemu-arti-build}
TOP_MODULE=__TOP__

echo "=== Building FlashSim embedded RTL model for $TOP_MODULE ==="
cd "$SCRIPT_DIR"
[ -f dut.h ] || { echo "missing dut.h (run flashsim arti-model first)" >&2; exit 1; }
[ -f arti_rtl_model.cpp ] || { echo "missing arti_rtl_model.cpp" >&2; exit 1; }

# Split DUT shards (dut_*.cpp) compile at -O2 in parallel. The thin ARTI
# wrapper stays -O1-compatible; without shards we keep -O1 on the monolith
# because clang -O2 never finishes on a ~300MB single TU.
NPROC="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
CXXFLAGS_BASE=(-std=gnu++17 -fPIC -fPIE -w -Wno-parentheses-equality -fbracket-depth=4096 -I.)
OBJS=()

compile_one() {
  local src="$1" obj="$2" opt="$3"
  echo "  c++ $opt $src"
  c++ "${CXXFLAGS_BASE[@]}" "$opt" -c "$src" -o "$obj"
}

shopt -s nullglob
SHARDS=(dut_*.cpp)
if ((${#SHARDS[@]} > 0)); then
  # DUT method shards: -O2. Wrapper includes the huge class layout — keep -O1
  # or clang hangs for tens of minutes at 0% CPU on arti_rtl_model.cpp.
  for src in "${SHARDS[@]}"; do
    obj="${src%.cpp}.o"
    OBJS+=("$obj")
    compile_one "$src" "$obj" -O2 &
  done
  wait
  compile_one arti_rtl_model.cpp arti_rtl_model.o -O1
  OBJS+=(arti_rtl_model.o)
else
  compile_one arti_rtl_model.cpp arti_rtl_model.o -O1
  OBJS=(arti_rtl_model.o)
fi

rm -f libarti_rtl_model.a
ar rcs libarti_rtl_model.a "${OBJS[@]}"
ls -lh libarti_rtl_model.a

cmp -s libarti_rtl_model.a "$QEMU_SRC/hw/misc/libarti_rtl_model.a" || \
    cp libarti_rtl_model.a "$QEMU_SRC/hw/misc/"
cmp -s "$SCRIPT_DIR/arti_rtl_model.h" "$QEMU_SRC/hw/misc/arti_rtl_model.h" || \
    cp "$SCRIPT_DIR/arti_rtl_model.h" "$QEMU_SRC/hw/misc/"

echo "=== Rebuilding QEMU ==="
if [ "${SKIP_QEMU_REBUILD:-}" != "1" ]; then
    if [ ! -f "$QEMU_BUILD/build.ninja" ]; then
        echo "QEMU build directory not configured yet; run setup_env.sh first"
        exit 1
    fi
    PATH="$ARTI_WORK/qemu-build-tools/bin:$PATH" ninja -C "$QEMU_BUILD" qemu-system-aarch64
    # macOS: a plain cp of the linked binary can leave Finder xattrs that break
    # ad-hoc codesign; dyld then hangs in _dyld_start. Clear and re-sign.
    if command -v codesign >/dev/null 2>&1; then
      xattr -cr "$QEMU_BUILD/qemu-system-aarch64" 2>/dev/null || true
      codesign -s - --force --deep "$QEMU_BUILD/qemu-system-aarch64"
    fi
    echo "=== Done ==="
    ls -lh "$QEMU_BUILD/qemu-system-aarch64"
else
    echo "SKIP_QEMU_REBUILD=1, leaving QEMU build to setup_env.sh"
fi
'''


SYSTEM_SOURCE = r'''
// FlashSim ARTI backend for GpuHostSystemAxi.
#include "dut.h"
#include "arti_rtl_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>

static GpuHostSystemAxiDut *g_rtl = nullptr;
static arti_mem_read_cb g_mem_read_cb = nullptr;
static arti_mem_write_cb g_mem_write_cb = nullptr;
static int g_arti_debug = -1;
static unsigned g_arti_idle = 0;

static constexpr unsigned TIMEOUT_CYCLES = 1000;
static constexpr unsigned M_AXI_BYTES = 8;

#ifndef ARTI_MODEL_MMIO_ADVANCE_CYCLES
// Cap per host MMIO settle. Must be in the same ballpark as Verilator's
// effective settle (~IDLE_GRACE of 20k with a free-running idle counter):
// during a busy draw the guest only advances the model on MMIO/IRQ poll, so a
// much smaller FlashSim cap starves the GPU relative to Verilator and makes
// end-to-end "business" paths look slower even when per-cycle eval is faster.
// Idle exits early via gpu_active()+IDLE_GRACE, so a higher cap does not tax
// sparse/idle traffic.
#define ARTI_MODEL_MMIO_ADVANCE_CYCLES 20000
#endif
#ifndef ARTI_MODEL_IDLE_GRACE
#define ARTI_MODEL_IDLE_GRACE 16
#endif
#ifndef ARTI_MODEL_SETTLE_MIN
#define ARTI_MODEL_SETTLE_MIN 256
#endif
#ifndef ARTI_MODEL_IRQ_PUMP_CYCLES
#define ARTI_MODEL_IRQ_PUMP_CYCLES 262144
#endif
#ifndef ARTI_MODEL_IRQ_PUMP_NS
// 0 = no wall-clock cap. QEMU's BQL is held, which also pauses guest
// jiffies, so a 30s fence wait survives a ~40s host draw.
#define ARTI_MODEL_IRQ_PUMP_NS 0
#endif

#ifndef ARTI_MODEL_HOLD_BOOST
#define ARTI_MODEL_HOLD_BOOST 0
#endif

// Optional: ARTI_MODEL_STATS=1 prints settle/irq-pump tick and activity counts
// so business-path A/B can separate "cycles advanced" from "wall time".
static uint64_t g_stat_ticks;
static uint64_t g_stat_active_ticks;
static uint64_t g_stat_settles;
static uint64_t g_stat_irq_pumps;

static int stats_enabled(void)
{
  static int v = -1;
  if (v < 0)
    v = getenv("ARTI_MODEL_STATS") ? 1 : 0;
  return v;
}

static void stats_report(const char *why)
{
  if (!stats_enabled())
    return;
  // QEMU merges model stderr into the guest serial log. Printing on every
  // 100us IRQ poll floods the console and skews wall-clock A/B — throttle.
  static uint64_t last_ticks;
  static unsigned calls;
  calls++;
  if ((calls & 255u) != 0 && g_stat_ticks - last_ticks < 50000ull)
    return;
  last_ticks = g_stat_ticks;
  fprintf(stderr,
          "[artistats] %s ticks=%llu active=%llu settles=%llu irq_pumps=%llu\n",
          why,
          (unsigned long long)g_stat_ticks,
          (unsigned long long)g_stat_active_ticks,
          (unsigned long long)g_stat_settles,
          (unsigned long long)g_stat_irq_pumps);
}

static unsigned env_u(const char *name, unsigned defv)
{
  const char *e = getenv(name);
  if (!e || !*e)
    return defv;
  unsigned v = (unsigned)strtoul(e, nullptr, 0);
  return v ? v : defv;
}

/** Like env_u but an explicit 0 is a valid value (disables the knob). */
static unsigned env_u0(const char *name, unsigned defv)
{
  const char *e = getenv(name);
  if (!e || !*e)
    return defv;
  return (unsigned)strtoul(e, nullptr, 0);
}

static unsigned hold_boost_ticks(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u0("ARTI_MODEL_HOLD_BOOST", ARTI_MODEL_HOLD_BOOST);
    once = 1;
  }
  return v;
}

static unsigned mmio_advance(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u("ARTI_MODEL_MMIO_ADVANCE", ARTI_MODEL_MMIO_ADVANCE_CYCLES);
    once = 1;
  }
  return v;
}

static unsigned idle_grace(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u("ARTI_MODEL_IDLE_GRACE", ARTI_MODEL_IDLE_GRACE);
    once = 1;
  }
  return v;
}

static long irq_pump_ns(void)
{
  static long v;
  static int once;
  if (!once) {
    const char *e = getenv("ARTI_MODEL_IRQ_PUMP_NS");
    v = e ? strtol(e, nullptr, 0) : ARTI_MODEL_IRQ_PUMP_NS;
    once = 1;
  }
  return v;
}

static unsigned irq_pump_cycles(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u("ARTI_MODEL_IRQ_PUMP", ARTI_MODEL_IRQ_PUMP_CYCLES);
    once = 1;
  }
  return v;
}

struct ArtiAxiWriteBeat {
  uint64_t data;
  uint64_t strb;
  bool last;
};
struct ArtiAxiWriteResp {
  uint64_t id;
  uint8_t resp;
};
struct ArtiAxiReadBeat {
  uint64_t data;
  uint64_t id;
  uint8_t resp;
  bool last;
};

static std::deque<ArtiAxiWriteBeat> g_wbeats;
static std::deque<ArtiAxiWriteResp> g_bresp;
static std::deque<ArtiAxiReadBeat> g_rresp;
static bool g_aw_active;
static uint64_t g_aw_addr, g_aw_id;
static unsigned g_aw_size, g_aw_beat;
static int g_write_status;

extern "C" void arti_rtl_model_set_memory_callbacks(
    arti_mem_read_cb read_cb, arti_mem_write_cb write_cb)
{
  g_mem_read_cb = read_cb;
  g_mem_write_cb = write_cb;
}

static void combo_slave(void)
{
  g_rtl->eval_io_s_axi_awready();
  g_rtl->eval_io_s_axi_wready();
  g_rtl->eval_io_s_axi_bvalid();
  g_rtl->eval_io_s_axi_bresp();
  g_rtl->eval_io_s_axi_arready();
  g_rtl->eval_io_s_axi_rvalid();
  g_rtl->eval_io_s_axi_rdata();
  g_rtl->eval_io_s_axi_rlast();
  g_rtl->eval_io_s_axi_rresp();
  g_rtl->eval_io_m_irq();
}

static void combo_master(void)
{
  g_rtl->eval_io_m_axi_awvalid();
  g_rtl->eval_io_m_axi_awaddr();
  g_rtl->eval_io_m_axi_awlen();
  g_rtl->eval_io_m_axi_awsize();
  g_rtl->eval_io_m_axi_awid();
  g_rtl->eval_io_m_axi_wvalid();
  g_rtl->eval_io_m_axi_wdata();
  g_rtl->eval_io_m_axi_wstrb();
  g_rtl->eval_io_m_axi_wlast();
  g_rtl->eval_io_m_axi_bready();
  g_rtl->eval_io_m_axi_arvalid();
  g_rtl->eval_io_m_axi_araddr();
  g_rtl->eval_io_m_axi_arlen();
  g_rtl->eval_io_m_axi_arsize();
  g_rtl->eval_io_m_axi_arid();
  g_rtl->eval_io_m_axi_rready();
}

static void combo_eval(void)
{
  combo_slave();
  combo_master();
}

static void combo(void)
{
  g_rtl->poke_inputs();
  combo_eval();
}

// Hot path for settle / IRQ pump: the control slave is idle, so skip the
// s_axi eval cone. Master + IRQ still update every cycle for memoryAXI.
static void combo_pump_eval(void)
{
  combo_master();
  g_rtl->eval_io_m_irq();
}

static void combo_pump(void)
{
  g_rtl->poke_inputs();
  combo_pump_eval();
}

static void idle_slave(void)
{
  g_rtl->io_s_axi_awvalid = 0;
  g_rtl->io_s_axi_wvalid = 0;
  g_rtl->io_s_axi_bready = 0;
  g_rtl->io_s_axi_arvalid = 0;
  g_rtl->io_s_axi_rready = 0;
}

static void mem_drive(void)
{
  g_rtl->io_m_axi_awready = !g_aw_active;
  g_rtl->io_m_axi_wready = 1;
  // One outstanding read burst. Accepting unlimited ARs while R beats
  // sit in a software FIFO desynchronized the GPU's memoryAxi (depth
  // loads never retired, draws hung after the first write burst).
  g_rtl->io_m_axi_arready = g_rresp.empty();
  g_rtl->io_m_axi_bvalid = !g_bresp.empty();
  g_rtl->io_m_axi_bresp = g_bresp.empty() ? 0 : g_bresp.front().resp;
  g_rtl->io_m_axi_bid = g_bresp.empty() ? 0 : g_bresp.front().id;
  g_rtl->io_m_axi_rvalid = !g_rresp.empty();
  g_rtl->io_m_axi_rdata = g_rresp.empty() ? 0 : g_rresp.front().data;
  g_rtl->io_m_axi_rresp = g_rresp.empty() ? 0 : g_rresp.front().resp;
  g_rtl->io_m_axi_rlast = !g_rresp.empty() && g_rresp.front().last;
  g_rtl->io_m_axi_rid = g_rresp.empty() ? 0 : g_rresp.front().id;
}

static void mem_capture(void)
{
  if (g_rtl->io_m_axi_awvalid && g_rtl->io_m_axi_awready) {
    g_aw_active = true;
    g_aw_addr = g_rtl->io_m_axi_awaddr;
    g_aw_id = g_rtl->io_m_axi_awid;
    g_aw_size = 1u << g_rtl->io_m_axi_awsize;
    g_aw_beat = 0;
    g_write_status = 0;
  }
  if (g_rtl->io_m_axi_wvalid && g_rtl->io_m_axi_wready)
    g_wbeats.push_back({(uint64_t)g_rtl->io_m_axi_wdata,
                        (uint64_t)g_rtl->io_m_axi_wstrb,
                        !!g_rtl->io_m_axi_wlast});
  if (g_aw_active && !g_wbeats.empty()) {
    ArtiAxiWriteBeat w = g_wbeats.front();
    g_wbeats.pop_front();
    uint64_t addr = g_aw_addr + (uint64_t)g_aw_beat * g_aw_size;
    unsigned lane = addr & (M_AXI_BYTES - 1);
    unsigned size = g_aw_size;
    if (size > M_AXI_BYTES - lane)
      size = M_AXI_BYTES - lane;
    uint64_t data = w.data >> (lane * 8);
    uint64_t mask = w.strb >> lane;
    int status = -1;
    if (g_mem_write_cb)
      status = g_mem_write_cb(addr, (const uint8_t *)&data, size, mask, g_aw_id);
    if (status != 0)
      g_write_status = status;
    g_aw_beat++;
    g_arti_idle = 0;
    if (w.last) {
      g_bresp.push_back({g_aw_id, (uint8_t)(g_write_status ? 2 : 0)});
      g_aw_active = false;
    }
  }
  if (g_rtl->io_m_axi_arvalid && g_rtl->io_m_axi_arready) {
    uint64_t base = g_rtl->io_m_axi_araddr;
    uint64_t id = g_rtl->io_m_axi_arid;
    unsigned size = 1u << g_rtl->io_m_axi_arsize;
    unsigned beats = 1u + g_rtl->io_m_axi_arlen;
    for (unsigned beat = 0; beat < beats; beat++) {
      uint64_t addr = base + (uint64_t)beat * size;
      unsigned lane = addr & (M_AXI_BYTES - 1);
      unsigned transfer = size > M_AXI_BYTES - lane ? M_AXI_BYTES - lane : size;
      uint64_t data = 0;
      int status = g_mem_read_cb
                       ? g_mem_read_cb(addr, (uint8_t *)&data, transfer, id)
                       : -1;
      data <<= lane * 8;
      g_rresp.push_back({data, id, (uint8_t)(status ? 2 : 0), beat + 1 == beats});
    }
    g_arti_idle = 0;
  }
}

static unsigned g_hold_boost;

// Busy detection is design-independent: the DUT is live while it still
// changes state (FlashSim's _chg counts state elements written last tick) or
// while AXI traffic is outstanding on either side of the boundary. No
// DUT-internal instance path is referenced, so GPU RTL changes cannot break
// it. The hold boost is deliberately not counted as busy: making it busy
// pins every settle to its full tick budget.
static int gpu_active(void)
{
  if (g_rtl->_chg)
    return 1;
  if (g_aw_active || !g_wbeats.empty() || !g_bresp.empty() || !g_rresp.empty())
    return 1;
  if (g_rtl->io_m_axi_arvalid || g_rtl->io_m_axi_awvalid || g_rtl->io_m_axi_wvalid)
    return 1;
  return 0;
}

// Safety net for skip-eval wake edges: for a bounded number of ticks after
// the host touches the device, force every hold group and bump generations.
// Host policy only — no design-specific register names. Set
// ARTI_MODEL_HOLD_BOOST=0 to run pure skip-eval.
static void force_hold_boost(void)
{
  if (!g_hold_boost)
    return;
  g_hold_boost--;
  const unsigned n =
      (unsigned)(sizeof(g_rtl->_h_busy) / sizeof(g_rtl->_h_busy[0]));
  for (unsigned i = 0; i < n; i++)
    g_rtl->_h_busy[i] = ~0ull;
  const unsigned np =
      (unsigned)(sizeof(g_rtl->_pg) / sizeof(g_rtl->_pg[0]));
  for (unsigned i = 0; i < np; i++)
    g_rtl->_pg[i]++;
}

// combo_fn runs before the posedge (must poke: mem_drive just updated
// inputs). eval_fn runs after tick(); tick() already poke_inputs(), and
// mem_capture does not change DUT inputs, so skip a redundant poke walk.
static void tick_with(void (*combo_fn)(void), void (*eval_fn)(void))
{
  force_hold_boost();
  mem_drive();
  combo_fn();
  // Sample response handshakes before the posedge so the DUT sees each
  // R/B beat for a full cycle. Popping before tick() dropped the first
  // beat of every read burst and hung depth-tested draws.
  const bool b_fire = g_rtl->io_m_axi_bvalid && g_rtl->io_m_axi_bready;
  const bool r_fire = g_rtl->io_m_axi_rvalid && g_rtl->io_m_axi_rready;
  mem_capture();
  // Inputs unchanged since the pre-tick combo's poke; tick() would poke again.
  g_rtl->tick_nba();
  eval_fn();
  if (b_fire && !g_bresp.empty())
    g_bresp.pop_front();
  if (r_fire && !g_rresp.empty())
    g_rresp.pop_front();
  if (gpu_active()) {
    g_arti_idle = 0;
    g_stat_active_ticks++;
  } else {
    g_arti_idle++;
  }
  g_stat_ticks++;
}

static void tick(void)
{
  tick_with(combo, combo_eval);
}

// Settle and IRQ pump never drive the control slave; use the lighter combo.
static void tick_pump(void)
{
  tick_with(combo_pump, combo_pump_eval);
}

static void arti_model_settle(void)
{
  if (g_arti_debug < 0)
    g_arti_debug = getenv("ARTI_MODEL_DEBUG") ? 1 : 0;
  g_arti_idle = 0;
  g_hold_boost = hold_boost_ticks();
  unsigned i;
  g_stat_settles++;
  // Do not stop on io_m_irq: a leftover level (JOB_CONTROL, W1C races)
  // would abort UCMD_SUBMIT/doorbell before fill/draw FSMs go live.
  // Completion is visible once gpu_active() drops and idle grace expires;
  // QEMU samples the pin after MMIO and on the IRQ poll timer.
  for (i = 0; i < mmio_advance(); i++) {
    tick_pump();
    if (i + 1 >= ARTI_MODEL_SETTLE_MIN && g_arti_idle > idle_grace())
      break;
  }
  if (g_arti_debug)
    fprintf(stderr,
            "[artidbg] settle ticks=%u irq=%u idle=%u chg=%u ar=%u aw=%u "
            "w=%zu b=%zu r=%zu boost=%u\n",
            i, (unsigned)g_rtl->io_m_irq, g_arti_idle, g_rtl->_chg,
            (unsigned)g_rtl->io_m_axi_arvalid,
            (unsigned)g_rtl->io_m_axi_awvalid, g_wbeats.size(),
            g_bresp.size(), g_rresp.size(), g_hold_boost);
}

extern "C" void arti_rtl_model_init(void)
{
  if (g_rtl)
    return;
  g_rtl = new GpuHostSystemAxiDut();
  idle_slave();
  mem_drive();
  g_rtl->io_s_axi_aresetn = 0;
  combo();
  for (int i = 0; i < 8; i++)
    tick();
  g_rtl->io_s_axi_aresetn = 1;
  idle_slave();
  combo();
  tick();
  tick();
}

extern "C" int arti_rtl_model_write(uint64_t addr, uint64_t data, unsigned size)
{
  if (!g_rtl || size == 0 || size > 4)
    return -1;
  uint32_t word = (uint32_t)data;
  uint32_t addr_val = (uint32_t)(addr & 0xffffffffu);
  uint8_t wstrb = (uint8_t)(((1u << size) - 1u) << (addr_val & 3u));
  idle_slave();
  g_rtl->io_s_axi_awaddr = addr_val;
  g_rtl->io_s_axi_awlen = 0;
  g_rtl->io_s_axi_awsize = 2;
  g_rtl->io_s_axi_awburst = 1;
  g_rtl->io_s_axi_wdata = word;
  g_rtl->io_s_axi_wlast = 1;
  g_rtl->io_s_axi_wstrb = wstrb;
  g_rtl->io_s_axi_awvalid = 1;
  g_rtl->io_s_axi_wvalid = 1;
  combo();
  int saw_aw = 0, saw_w = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_awready)
      saw_aw = 1;
    if (g_rtl->io_s_axi_wready)
      saw_w = 1;
    tick();
    if (saw_aw && saw_w)
      break;
  }
  if (!saw_aw || !saw_w)
    return -1;
  idle_slave();
  g_rtl->io_s_axi_bready = 1;
  combo();
  int saw_b = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_bvalid) {
      tick();
      saw_b = 1;
      break;
    }
    tick();
  }
  if (!saw_b)
    return -1;
  idle_slave();
  combo();
  arti_model_settle();
  return 0;
}

extern "C" int arti_rtl_model_read(uint64_t addr, uint64_t *data, unsigned size)
{
  if (!g_rtl || !data || size == 0 || size > 4)
    return -1;
  uint32_t addr_val = (uint32_t)(addr & 0xffffffffu);
  idle_slave();
  g_rtl->io_s_axi_araddr = addr_val;
  g_rtl->io_s_axi_arlen = 0;
  g_rtl->io_s_axi_arsize = 2;
  g_rtl->io_s_axi_arburst = 1;
  g_rtl->io_s_axi_arvalid = 1;
  combo();
  int saw_ar = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_arready) {
      tick();
      saw_ar = 1;
      break;
    }
    tick();
  }
  if (!saw_ar)
    return -1;
  idle_slave();
  g_rtl->io_s_axi_rready = 1;
  combo();
  uint32_t rdata_val = 0;
  int saw_r = 0;
  for (unsigned i = 0; i < TIMEOUT_CYCLES; i++) {
    if (g_rtl->io_s_axi_rvalid) {
      rdata_val = g_rtl->io_s_axi_rdata;
      saw_r = 1;
      break;
    }
    tick();
  }
  if (!saw_r)
    return -1;
  tick();
  idle_slave();
  combo();
  *data = rdata_val;
  arti_model_settle();
  return 0;
}

extern "C" int arti_rtl_model_check_irq(unsigned index)
{
  if (!g_rtl || index != 0)
    return 0;
  // Idle polls must not tick: QEMU fires this every 100us under the BQL.
  // Only pump the DUT while a job/DMA FSM is actually live so fence waits
  // make progress without stalling the guest vCPU.
  if (gpu_active()) {
    const unsigned cap = irq_pump_cycles();
    const long limit = irq_pump_ns();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (g_arti_debug < 0)
      g_arti_debug = getenv("ARTI_MODEL_DEBUG") ? 1 : 0;
    if (g_arti_debug)
      fprintf(stderr, "[artidbg] irq_pump cap=%u chg=%u ar=%u aw=%u r=%zu\n",
              cap, g_rtl->_chg, (unsigned)g_rtl->io_m_axi_arvalid,
              (unsigned)g_rtl->io_m_axi_awvalid, g_rresp.size());
    g_hold_boost = hold_boost_ticks();
    g_stat_irq_pumps++;
    for (unsigned i = 0; i < cap && gpu_active(); i++) {
      tick_pump();
      if (limit > 0 && (i & 15u) == 15u) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L +
                  (t1.tv_nsec - t0.tv_nsec);
        if (ns > limit)
          break;
      }
    }
  }
  combo_pump();
  stats_report("check_irq");
  return g_rtl->io_m_irq ? 1 : 0;
}
'''


def write_embedded_model(out_dir: Path, top: str, verilog: Path | None = None,
                         frontend: str = "circt") -> None:
    """Compile `top` and emit the ARTI embedded library sources."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if verilog is not None:
        # GpuHostSystemAxi is too large for a single -O2 TU; emit method shards.
        split = 16 if top == "GpuHostSystemAxi" else 0
        compile_file(verilog, out_dir / "dut.h", frontend=frontend, split=split)
    (out_dir / "arti_rtl_model.h").write_text(HEADER)
    if top == "GpuHostSystemAxi":
        (out_dir / "arti_rtl_model.cpp").write_text(SYSTEM_SOURCE)
    elif top == "GpuHostAxi":
        (out_dir / "arti_rtl_model.cpp").write_text(SOURCE)
    else:
        raise SystemExit(f"unsupported ARTI top {top}")
    (out_dir / "arti_smoke.cpp").write_text(SMOKE)
    script = BUILD_EMBEDDED.replace("__TOP__", top)
    path = out_dir / "build_embedded.sh"
    path.write_text(script)
    path.chmod(0o755)
