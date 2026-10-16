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

import os
import re
from pathlib import Path

from flashsim.compile import compile_file
from flashsim.circt_frontend import verilog_sources


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
// Advance exactly `n` posedges with the same per-tick work a settle does.
// Exists so a benchmark can drive a fixed tick count: `check_irq` advances one
// tick on the Verilator backend but up to ARTI_MODEL_IRQ_PUMP on this one, so
// timing "one call" on each measures different amounts of simulation. QEMU
// never calls this.
void arti_rtl_model_tick_n(unsigned n);
// Skip-kernel traffic, read without stopping the model. Lets a driver time a
// specific phase (e.g. the free-running scanout) and see commit slots per tick
// and hold invocations per tick, which is what separates "the commit path is
// expensive" from "the wake set is too wide to skip anything". Design
// independent; the pointers may be NULL.
void arti_rtl_model_skip_stats(uint64_t *ticks, uint64_t *commits,
                               uint64_t *holds, uint64_t *holds_wrote);
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
# Keep pathname expansion and archive member order independent of host locale.
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QEMU_SRC=${QEMU_SRC:?must be set}
# The rest of this script runs from the generated model directory. Resolve
# caller-provided relative paths before `cd`, otherwise a successful large
# model compile can fail only at the final archive-copy step.
QEMU_SRC="$(cd "$QEMU_SRC" && pwd)"
ARTI_WORK="${ARTI_WORK:-$(cd "$SCRIPT_DIR/../../.." && pwd)/arti-work}"
QEMU_BUILD=${QEMU_BUILD:-$ARTI_WORK/qemu-arti-build}
TOP_MODULE=__TOP__
# Optional LLVM PGO profile produced by a representative host/QEMU training
# run. Resolve the caller's path before changing to the model directory.
PGO_PROFILE="${FLASHSIM_PGO_PROFILE:-}"
PGO_GENERATE="${FLASHSIM_PGO_GENERATE:-0}"
case "$PGO_GENERATE" in
  0|1) ;;
  *) echo "FLASHSIM_PGO_GENERATE must be 0 or 1" >&2; exit 1 ;;
esac
if [[ "$PGO_GENERATE" == 1 && -n "$PGO_PROFILE" ]]; then
  echo "FLASHSIM_PGO_GENERATE and FLASHSIM_PGO_PROFILE are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$PGO_PROFILE" ]]; then
  [ -f "$PGO_PROFILE" ] || { echo "FLASHSIM_PGO_PROFILE not found: $PGO_PROFILE" >&2; exit 1; }
  PGO_PROFILE="$(cd "$(dirname "$PGO_PROFILE")" && pwd)/$(basename "$PGO_PROFILE")"
  echo "PGO profile: $PGO_PROFILE"
fi

echo "=== Building FlashSim embedded RTL model for $TOP_MODULE ==="
cd "$SCRIPT_DIR"
[ -f dut.h ] || { echo "missing dut.h (run flashsim arti-model first)" >&2; exit 1; }
[ -f arti_rtl_model.cpp ] || { echo "missing arti_rtl_model.cpp" >&2; exit 1; }

# Prefer ccache compiler wrappers when present (Homebrew libexec or Linux
# distro /usr/lib/ccache). Keeps the plain `c++` invocations below cached.
for _ccache_libexec in \
    /opt/homebrew/opt/ccache/libexec \
    /usr/local/opt/ccache/libexec \
    /usr/lib/ccache \
    /usr/lib64/ccache; do
  if [ -d "$_ccache_libexec" ]; then
    export PATH="$_ccache_libexec:$PATH"
    echo "ccache     : $_ccache_libexec"
    break
  fi
done
unset _ccache_libexec

# Split DUT shards (dut_*.cpp) compile at -O2 in parallel. The thin ARTI
# wrapper stays -O1-compatible; without shards we keep -O1 on the monolith
# because clang -O2 never finishes on a ~300MB single TU.
NPROC="${FLASHSIM_COMPILE_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
SHARD_OPT="${FLASHSIM_SHARD_OPT:--O2}"
CALLBACK_WRITE_BYTES="${FLASHSIM_WRITE_CALLBACK_BYTES:-8}"
case "$CALLBACK_WRITE_BYTES" in
  1|2|4|8|16|32|64) ;;
  *) echo "FLASHSIM_WRITE_CALLBACK_BYTES must be 1, 2, 4, 8, 16, 32 or 64" >&2; exit 1 ;;
esac
case "$SHARD_OPT" in
  -O0|-O1|-O2|-O3) ;;
  *) echo "FLASHSIM_SHARD_OPT must be -O0, -O1, -O2 or -O3" >&2; exit 1 ;;
esac
# Keep this opt-in: a profile from a tiny smoke test can over-specialize the
# model and make a full-system workload slower.
OBJS=()

compile_one() {
  local src="$1" obj="$2" opt="$3"
  local CXXFLAGS_BASE=(-std=gnu++17 -fPIC -fPIE -w -Wno-parentheses-equality -fbracket-depth=4096 -I.)
  if [[ "$src" == arti_rtl_model.cpp ]]; then
    CXXFLAGS_BASE+=("-DARTI_MODEL_WRITE_CALLBACK_BYTES=$CALLBACK_WRITE_BYTES")
  fi
  echo "  c++ $opt $src"
  if [[ -n "$PGO_PROFILE" ]]; then
    c++ "${CXXFLAGS_BASE[@]}" -fprofile-instr-use="$PGO_PROFILE" "$opt" -c "$src" -o "$obj"
  elif [[ "$PGO_GENERATE" == 1 ]]; then
    c++ "${CXXFLAGS_BASE[@]}" -fprofile-instr-generate "$opt" -c "$src" -o "$obj"
  else
    c++ "${CXXFLAGS_BASE[@]}" "$opt" -c "$src" -o "$obj"
  fi
}

shopt -s nullglob
SHARDS=(dut_*.cpp)
if ((${#SHARDS[@]} > 0)); then
  # DUT method shards: -O2. Wrapper includes the huge class layout — keep -O1
  # or clang hangs for tens of minutes at 0% CPU on arti_rtl_model.cpp.
  # (Tiny in-header evals are capped via _SPLIT_INLINE_MAX_BYTES so dut_0's
  # tick_nba still finishes under -O2.)
  # dut_commit.cpp holds the dirty-list switch (~3MB for a 707-page top), so it
  # stays at -O0 to keep that one TU's compile time bounded. The splitter
  # emits dut_commit_helpers.cpp separately so the page-invalidation loop
  # uses the normal shard optimization level without optimizing the huge
  # switch. On the current 640x480 GTK model, sampling found that loop was
  # the largest active function when compiled inside the -O0 commit TU.
  # A work queue fills a slot as soon as any compiler exits. Waiting for
  # PIDs in launch order left most slots idle behind a large early shard.
  [[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || {
    echo "FLASHSIM_COMPILE_JOBS must be a positive integer" >&2; exit 1;
  }
  export -f compile_one
  export PGO_PROFILE PGO_GENERATE
  tasks=()
  for src in "${SHARDS[@]}"; do
    obj="${src%.cpp}.o"
    OBJS+=("$obj")
    opt="$SHARD_OPT"
    [[ "$src" != "dut_commit.cpp" ]] || opt=-O0
    tasks+=("$src" "$obj" "$opt")
  done
  # Compile the largest sources first to reduce the final serial tail.
  # Keep OBJS in filename order so scheduling does not reorder the archive.
  python3 -c 'import os,sys
v=sys.argv[1:]
tasks=[v[i:i+3] for i in range(0,len(v),3)]
for task in sorted(tasks,key=lambda t:os.path.getsize(t[0]),reverse=True):
    for arg in task:sys.stdout.buffer.write(os.fsencode(arg)+b"\0")' "${tasks[@]}" | \
    xargs -0 -n 3 -P "$NPROC" bash -c 'compile_one "$@"' _
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
    # QEMU's final linker is Objective-C on macOS and C on Linux. The model
    # archive alone cannot pull in LLVM's profile runtime; configure the
    # language actually used for the final QEMU link. Only touch an empty
    # option (or one left by this script) to preserve caller link flags.
    PGO_LINK_MARKER="$QEMU_BUILD/.flashsim-pgo-link-arg"
    if [[ "$PGO_GENERATE" == 1 ]]; then
      if [[ "$(uname -s)" == Darwin ]]; then
        PGO_LINK_KEY=objc_link_args
      else
        PGO_LINK_KEY=c_link_args
      fi
      PGO_LINK_CURRENT="$(meson introspect "$QEMU_BUILD" --buildoptions | \
        python3 -c 'import json,sys; k=sys.argv[1]; print(" ".join(next(x["value"] for x in json.load(sys.stdin) if x["name"] == k)))' "$PGO_LINK_KEY")"
      if [[ -n "$PGO_LINK_CURRENT" && "$PGO_LINK_CURRENT" != -fprofile-instr-generate ]]; then
        echo "QEMU $PGO_LINK_KEY already has link flags: $PGO_LINK_CURRENT" >&2
        exit 1
      fi
      printf '%s\n' "$PGO_LINK_KEY" > "$PGO_LINK_MARKER"
      meson configure "$QEMU_BUILD" "-D$PGO_LINK_KEY=-fprofile-instr-generate"
    elif [[ -f "$PGO_LINK_MARKER" ]]; then
      PGO_LINK_KEY="$(cat "$PGO_LINK_MARKER")"
      case "$PGO_LINK_KEY" in
        objc_link_args|c_link_args) ;;
        *) echo "invalid PGO link marker: $PGO_LINK_MARKER" >&2; exit 1 ;;
      esac
      meson configure "$QEMU_BUILD" "-D$PGO_LINK_KEY="
      rm "$PGO_LINK_MARKER"
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
if [[ "$PGO_GENERATE" == 1 ]]; then
  echo "Run representative QEMU training with LLVM_PROFILE_FILE set, then"
  echo "merge the .profraw files with llvm-profdata and rebuild with"
  echo "FLASHSIM_PGO_PROFILE=/path/to/merged.profdata."
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
#include <array>
#include <algorithm>

static GpuHostSystemAxiDut *g_rtl = nullptr;
static arti_mem_read_cb g_mem_read_cb = nullptr;
static arti_mem_write_cb g_mem_write_cb = nullptr;
static int g_arti_debug = -1;
static unsigned g_arti_idle = 0;
// Set only around settle / IRQ pump loops. Those loops' only input writer is
// mem_drive(), so an unchanged drive can skip the input-image compare.
static int g_settle_quiet = 0;

static constexpr unsigned TIMEOUT_CYCLES = 1000;
// Scalar and limb-array DUT ports have the exact byte size of the AXI bus.
static constexpr unsigned M_AXI_BYTES =
    sizeof(((GpuHostSystemAxiDut *)nullptr)->io_m_axi_wdata);
static_assert(M_AXI_BYTES >= 4 && M_AXI_BYTES <= 64 &&
              (M_AXI_BYTES & (M_AXI_BYTES - 1)) == 0, "unsupported AXI beat size");
static_assert(sizeof(((GpuHostSystemAxiDut *)nullptr)->io_m_axi_rdata) == M_AXI_BYTES,
              "AXI read/write data widths differ");
#ifndef ARTI_MODEL_WRITE_CALLBACK_BYTES
// Keep the measured eight-byte default. Hosts can opt into complete beats
// with -DARTI_MODEL_WRITE_CALLBACK_BYTES=64 after workload qualification.
#define ARTI_MODEL_WRITE_CALLBACK_BYTES 8
#endif
static constexpr unsigned WRITE_CALLBACK_BYTES = ARTI_MODEL_WRITE_CALLBACK_BYTES;
static_assert(WRITE_CALLBACK_BYTES && WRITE_CALLBACK_BYTES <= 64 &&
              (WRITE_CALLBACK_BYTES & (WRITE_CALLBACK_BYTES - 1)) == 0,
              "write callback limit must be a power of two from 1 to 64");
using ArtiAxiData = std::array<uint8_t, M_AXI_BYTES>;

#ifndef ARTI_MODEL_MMIO_ADVANCE_CYCLES
// Cap per host MMIO settle. Match Verilator's ARTI default (500k): while the
// GPU keeps DMA/_chg live, idle stays reset and the job can finish inside one
// MMIO settle instead of being sliced across 100us IRQ polls (BQL tax).
// Idle exits early via gpu_active()+IDLE_GRACE, so sparse MMIO is unaffected.
#define ARTI_MODEL_MMIO_ADVANCE_CYCLES 500000
#endif
#ifndef ARTI_MODEL_IDLE_GRACE
// FlashSim detects activity via _chg + AXI queues (stricter than Verilator's
// mem-traffic-only idle reset), so a small grace is enough to drain bubbles
// without burning Verilator's 20k idle ticks on every quiet MMIO.
#define ARTI_MODEL_IDLE_GRACE 16
#endif
#ifndef ARTI_MODEL_SETTLE_MIN
// A quiet control access needs only a short observation window: FlashSim's
// activity detector already requires AXI quiescence plus IDLE_GRACE.  Keeping
// this at 128 halves the fixed cost of idle QEMU MMIO polls while active GPU
// work continues until the externally visible queues drain.
#define ARTI_MODEL_SETTLE_MIN 128
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
#ifndef ARTI_MODEL_CHG_TAIL
// Ticks of continued state change that still count as "busy" after the last
// externally visible event. This is what stops a free-running display counter
// (one state commit per tick, zero AXI traffic) from pinning every MMIO settle
// at the full tick cap. Raise it if a job that finishes entirely inside the
// DUT ever needs more than this to drain; the guest still makes progress in
// slices either way, because each poll advances a fresh settle.
#define ARTI_MODEL_CHG_TAIL 4096
#endif
#ifndef ARTI_MODEL_SETTLE_NS
// Wall-clock ceiling on one MMIO settle, 0 = only the tick cap applies. Keeps a
// never-quiescing design from pinning QEMU's BQL for the whole tick budget on
// every guest register access.
#define ARTI_MODEL_SETTLE_NS 0
#endif

// Optional: ARTI_MODEL_STATS=1 prints settle/irq-pump tick and activity counts
// so business-path A/B can separate "cycles advanced" from "wall time".
// ARTI_MODEL_STATS=2 additionally drops the print throttle, which is what you
// want when reading per-tick skip traffic off a real workload: the throttled
// samples miss whole phases (e.g. the raster/scanout burst) entirely.
static uint64_t g_stat_ticks;
static uint64_t g_stat_active_ticks;
static uint64_t g_stat_settles;
static uint64_t g_stat_irq_pumps;

static int stats_level(void)
{
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("ARTI_MODEL_STATS");
    v = e ? (int)strtol(e, nullptr, 0) : 0;
  }
  return v;
}

static int stats_enabled(void)
{
  return stats_level() > 0;
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
  if (stats_level() < 2 &&
      (calls & 255u) != 0 && g_stat_ticks - last_ticks < 50000ull)
    return;
  last_ticks = g_stat_ticks;
  // Skip-kernel traffic per tick: commit slots and hold invocations separate
  // "the commit/bump path costs too much" from "the wake set is too wide to
  // skip anything". Both are cumulative and design-independent.
  const uint64_t nt = g_rtl ? g_rtl->_nt : 0;
  const uint64_t cc = g_rtl ? g_rtl->_cc : 0;
  const uint64_t hc = g_rtl ? g_rtl->_hc : 0;
  const uint64_t hb = g_rtl ? g_rtl->_hb : 0;
  fprintf(stderr,
          "[artistats] %s ticks=%llu active=%llu settles=%llu irq_pumps=%llu "
          "cmt=%llu holds=%llu wrote=%llu per_tick=%.2f/%.2f/%.2f\n",
          why,
          (unsigned long long)g_stat_ticks,
          (unsigned long long)g_stat_active_ticks,
          (unsigned long long)g_stat_settles,
          (unsigned long long)g_stat_irq_pumps,
          (unsigned long long)cc, (unsigned long long)hc,
          (unsigned long long)hb,
          nt ? (double)cc / (double)nt : 0.0,
          nt ? (double)hc / (double)nt : 0.0,
          nt ? (double)hb / (double)nt : 0.0);
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

static unsigned chg_tail(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u("ARTI_MODEL_CHG_TAIL", ARTI_MODEL_CHG_TAIL);
    once = 1;
  }
  return v;
}

static long settle_ns(void)
{
  static long v;
  static int once;
  if (!once) {
    const char *e = getenv("ARTI_MODEL_SETTLE_NS");
    v = e ? strtol(e, nullptr, 0) : ARTI_MODEL_SETTLE_NS;
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

static unsigned settle_min(void)
{
  static unsigned v;
  static int once;
  if (!once) {
    v = env_u("ARTI_MODEL_SETTLE_MIN", ARTI_MODEL_SETTLE_MIN);
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
  ArtiAxiData data;
  uint64_t strb;
  bool last;
};
struct ArtiAxiWriteResp {
  uint64_t id;
  uint8_t resp;
};
struct ArtiAxiReadBeat {
  ArtiAxiData data;
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
// s_axi eval cone. Master + IRQ still update every cycle for memory AXI.
static void combo_pump_eval(void)
{
  combo_master();
  g_rtl->eval_io_m_irq();
}

static void combo_pump(void)
{
  // mem_drive() always runs immediately before this and may change m_axi_*
  // ready/valid inputs. Always poke so the DUT sees those updates even when
  // the control slave is idle. Skipping poke here left write-through depth
  // stores invisible to the next DEPTH_LOAD submission under FlashSim while
  // Verilator (always-poke eval) passed.
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

template <typename T, typename U>
static bool drive_in(T& dst, U val) {
  T next = static_cast<T>(val);
  if (dst == next)
    return false;
  dst = next;
  return true;
}

static bool mem_drive(void)
{
  bool ch = false;
  ch |= drive_in(g_rtl->io_m_axi_awready, !g_aw_active);
  ch |= drive_in(g_rtl->io_m_axi_wready, 1);
  // Tagged RTL read contexts can overlap independent bursts. Preserve the
  // queued head under R backpressure and bound admission by pending beats.
  // A newly accepted cache-line burst adds at most sixteen further beats.
  ch |= drive_in(g_rtl->io_m_axi_arready, g_rresp.size() < 64);
  ch |= drive_in(g_rtl->io_m_axi_bvalid, !g_bresp.empty());
  ch |= drive_in(g_rtl->io_m_axi_bresp, g_bresp.empty() ? 0 : g_bresp.front().resp);
  ch |= drive_in(g_rtl->io_m_axi_bid, g_bresp.empty() ? 0 : g_bresp.front().id);
  ch |= drive_in(g_rtl->io_m_axi_rvalid, !g_rresp.empty());
  const ArtiAxiData zero{};
  const auto &data = g_rresp.empty() ? zero : g_rresp.front().data;
  if (memcmp(&g_rtl->io_m_axi_rdata, data.data(), M_AXI_BYTES)) {
    memcpy(&g_rtl->io_m_axi_rdata, data.data(), M_AXI_BYTES);
    ch = true;
  }
  ch |= drive_in(g_rtl->io_m_axi_rresp, g_rresp.empty() ? 0 : g_rresp.front().resp);
  ch |= drive_in(g_rtl->io_m_axi_rlast, !g_rresp.empty() && g_rresp.front().last);
  ch |= drive_in(g_rtl->io_m_axi_rid, g_rresp.empty() ? 0 : g_rresp.front().id);
  return ch;
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
  if (g_rtl->io_m_axi_wvalid && g_rtl->io_m_axi_wready) {
    ArtiAxiWriteBeat w{};
    ArtiAxiData strobe{};
    memcpy(w.data.data(), &g_rtl->io_m_axi_wdata, M_AXI_BYTES);
    // One strobe bit per data byte, including a four-byte AXI bus.
    memcpy(strobe.data(), &g_rtl->io_m_axi_wstrb, (M_AXI_BYTES + 7) / 8);
    memcpy(&w.strb, strobe.data(), (M_AXI_BYTES + 7) / 8);
    w.last = !!g_rtl->io_m_axi_wlast;
    g_wbeats.push_back(w);
  }
  if (g_aw_active && !g_wbeats.empty()) {
    ArtiAxiWriteBeat w = g_wbeats.front();
    g_wbeats.pop_front();
    uint64_t addr = g_aw_addr + (uint64_t)g_aw_beat * g_aw_size;
    unsigned lane = addr & (M_AXI_BYTES - 1);
    unsigned size = g_aw_size;
    int status = 0;
    unsigned done = 0;
    unsigned data_offset = lane;
    while (done < size) {
      unsigned chunk = std::min({WRITE_CALLBACK_BYTES, size - done, M_AXI_BYTES - data_offset});
      // The callback accepts an entire AXI beat. Avoid shifting by 64 for a
      // full-width mask; unaligned transfers still stop at the bus boundary.
      uint64_t chunk_mask = chunk == 64 ? UINT64_MAX : (UINT64_C(1) << chunk) - 1;
      uint64_t mask = (w.strb >> data_offset) & chunk_mask;
      // Zero strobes have no memory effect. Keep errors sticky across all
      // enabled chunks; a later success must not turn SLVERR into OKAY.
      if (mask && (!g_mem_write_cb ||
          g_mem_write_cb(addr + done, w.data.data() + data_offset, chunk,
                         mask, g_aw_id) != 0))
        status = -1;
      done += chunk;
      data_offset += chunk;
      if (data_offset == M_AXI_BYTES)
        data_offset = 0;
    }
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
      ArtiAxiReadBeat r{};
      int status = g_mem_read_cb
                       ? g_mem_read_cb(addr, r.data.data() + lane, transfer, id)
                       : -1;
      r.id = id;
      r.resp = status ? 2 : 0;
      r.last = beat + 1 == beats;
      g_rresp.push_back(r);
    }
    g_arti_idle = 0;
  }
}

static unsigned g_hold_boost;
// Budget, in ticks, that state changes may spend keeping the model "busy"
// after the last externally visible event. Refilled by any AXI beat. See
// gpu_active().
static unsigned g_chg_tail;

// Busy detection is design-independent: the DUT is live while it still
// changes state (FlashSim's _chg counts state elements written last tick) or
// while AXI traffic is outstanding on either side of the boundary. No
// DUT-internal instance path is referenced, so GPU RTL changes cannot break
// it. The hold boost is deliberately not counted as busy: making it busy
// pins every settle to its full tick budget.
// Externally visible work in flight: AXI traffic on either side of the
// boundary. Design independent — no DUT register or instance name is touched.
static int axi_busy(void)
{
  if (g_aw_active || !g_wbeats.empty() || !g_bresp.empty() || !g_rresp.empty())
    return 1;
  return g_rtl->io_m_axi_arvalid || g_rtl->io_m_axi_awvalid ||
         g_rtl->io_m_axi_wvalid;
}

static int gpu_active(void)
{
  if (axi_busy())
    return 1;
  // `_chg` on its own means "a flop moved", not "the GPU owes us work". A
  // free-running scanout commits one state element per tick and issues no AXI
  // traffic at all, so treating it as busy pinned every settle at the full
  // MMIO_ADVANCE cap: measured 500002 ticks per guest MMIO read with the
  // scanout running versus 258 with it off, a 1940x penalty per host read that
  // the guest's vblank polling loop then pays thousands of times. Let state
  // changes hold the model open only for a bounded tail after the last
  // externally visible event (AXI beat, master valid), so a job still drains
  // but a free-running counter cannot.
  // Both conditions matter. Without the tail a free-running counter pins every
  // settle at the cap; without the `_chg` term a model that has simply gone
  // quiet would still burn its whole tail before the idle grace could fire.
  return g_rtl->_chg != 0 && g_chg_tail < chg_tail();
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
  // The dispatch loop walks _h_any (one bit per hold word), so force-busting
  // _h_busy alone would leave every hold asleep.
  const unsigned na = (unsigned)(sizeof(g_rtl->_h_any) / sizeof(g_rtl->_h_any[0]));
  for (unsigned i = 0; i < na; i++)
    g_rtl->_h_any[i] = ~0ull;
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
  const bool inputs_changed = mem_drive();
  // Settle and the IRQ pump do not touch the control slave. When the memory
  // side also holds its previous values, the input image is unchanged and
  // poke can skip the compare.
  if (g_settle_quiet && !inputs_changed)
    g_rtl->_in_fresh = 1;
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
  {
    // The tail counts down ticks with no externally visible event. It must
    // advance on every such tick, not only while `_chg` is set: gating it on
    // `_chg` makes the limit unreachable exactly when the design goes quiet,
    // which is when the settle most needs to exit.
    //
    // axi_busy() is sampled once here and reused by the active test below;
    // it reads five DUT valids plus three queue depths, and this runs on
    // every tick of every settle.
    const bool busy = axi_busy();
    if (busy)
      g_chg_tail = 0;
    else if (g_chg_tail < chg_tail())
      g_chg_tail++;
    if (busy || (g_rtl->_chg != 0 && g_chg_tail < chg_tail())) {
      g_arti_idle = 0;
      g_stat_active_ticks++;
    } else {
      g_arti_idle++;
    }
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
  // The `_chg` budget is scoped to one settle: "may I keep pumping?" is a
  // question about this burst, so every settle starts with a full budget.
  // Leaving it global lets it saturate during the first burst and then starve
  // every later one, which silently caps a long AXI-free job at SETTLE_MIN.
  g_chg_tail = 0;
  g_hold_boost = hold_boost_ticks();
  unsigned i;
  g_stat_settles++;
  // Do not stop on io_m_irq: a leftover level (JOB_CONTROL, W1C races)
  // would abort UCMD_SUBMIT/doorbell before fill/draw FSMs go live.
  // Completion is visible once gpu_active() drops and idle grace expires;
  // QEMU samples the pin after MMIO and on the IRQ poll timer.
  const long ns_budget = settle_ns();
  struct timespec s0, s1;
  if (ns_budget > 0)
    clock_gettime(CLOCK_MONOTONIC, &s0);
  g_settle_quiet = 1;
  for (i = 0; i < mmio_advance(); i++) {
    tick_pump();
    if (i + 1 >= settle_min() && g_arti_idle > idle_grace())
      break;
    // Wall-clock bound, independent of the activity policy. A pathological
    // design that never quiesces would otherwise hold QEMU's BQL for the whole
    // tick cap on every MMIO access.
    if (ns_budget > 0 && (i & 0x3fu) == 0x3fu) {
      clock_gettime(CLOCK_MONOTONIC, &s1);
      const long ns = (s1.tv_sec - s0.tv_sec) * 1000000000L +
                      (s1.tv_nsec - s0.tv_nsec);
      if (ns > ns_budget)
        break;
    }
  }
  g_settle_quiet = 0;
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
  g_chg_tail = 0;
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

extern "C" void arti_rtl_model_tick_n(unsigned n)
{
  if (!g_rtl)
    return;
  for (unsigned i = 0; i < n; i++)
    tick();
}

extern "C" void arti_rtl_model_skip_stats(uint64_t *ticks, uint64_t *commits,
                                          uint64_t *holds,
                                          uint64_t *holds_wrote)
{
  if (!g_rtl)
    return;
  if (ticks)
    *ticks = g_rtl->_nt;
  if (commits)
    *commits = g_rtl->_cc;
  if (holds)
    *holds = g_rtl->_hc;
  if (holds_wrote)
    *holds_wrote = g_rtl->_hb;
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
    g_chg_tail = 0;  // same scoping as arti_model_settle
    g_stat_irq_pumps++;
    g_settle_quiet = 1;
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
    g_settle_quiet = 0;
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
        split = _dut_shard_count(top, verilog)
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


def _dut_shard_count(top: str, verilog: Path) -> int:
    """Choose a shard count without changing the generated RTL.

    The four-CU ``GpuHostSystemAxi`` top is materially larger than the
    original one-CU snapshot.  Keeping the old fixed 16-way split makes each
    C++ translation unit unnecessarily large and raises peak compiler memory.
    The generated top names each CU explicitly, so counting those instance
    names is a stable elaboration-size signal and does not depend on a GPU
    configuration being imported into FlashSim.

    ``FLASHSIM_DUT_SHARDS`` is an escape hatch for CI machines with a fixed
    memory/parallelism budget.  A value of zero restores the unsplit path,
    which is useful for small tops and for debugging.
    """
    if top != "GpuHostSystemAxi":
        return 0
    override = os.environ.get("FLASHSIM_DUT_SHARDS")
    if override is not None:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError("FLASHSIM_DUT_SHARDS must be an integer") from exc
        if value < 0:
            raise ValueError("FLASHSIM_DUT_SHARDS must be >= 0")
        return value
    try:
        text = "\n".join(
            source.read_text(errors="ignore")
            for source in verilog_sources(verilog)
        )
    except OSError:
        return 16
    cu_ids = {int(m.group(1)) for m in re.finditer(r"_computeUnits_(\d+)_", text)}
    if len(cu_ids) >= 4:
        return 32
    return 16
