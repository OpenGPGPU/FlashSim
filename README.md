# FlashSim

Cycle-accurate RTL simulation that must match Verilator at DUT ports, and beat
it by skipping inactive combinational cones. The skip kernel is frozen at the
go/no-go; `GpuHostAxi` / `GpuHostSystemAxi` are the QEMU/ARTI control tops.

## Goal

| Constraint | Rule |
|---|---|
| Gold model | Same Verilog, same stimulus, same per-cycle `dout` as Verilator |
| Speed (unit) | Low-activity ≥ 2× **1-thread** Verilator; busy ≥ 0.8×. Also report `--threads 4`. |
| Speed (ARTI) | Real QEMU/Linux GPU paths (bind, sparse MMIO, DRM jobs) must beat the Verilator-linked QEMU on **wall clock**. Microbench-only wins do not count. |
| CIRCT | Official release binaries, not a git submodule |

CIRCT lowers RTL (`firtool` / `circt-opt` / `circt-verilog`). FlashSim imports
HW/Comb/Seq, restores skip-friendly mux/`if` form, and emits C++. Skip caches
use ESSENT-style coarsened generation counters. GPU-sized nets split hold-style
updates into skipped `noinline` methods, flatten nested reset/set `if`s, wrap
idle `x <= y` copies as holds, wake those methods from changed leaves
(GSIM-style), and commit through a write worklist. Const updates are
self-gated, arrays are copy-on-write, and hold dispatch is a 64-bit bitmask
so idle ticks do not memcpy, re-clear, or scan hundreds of skip flags. Do
not fork CIRCT unless a pass must land in-tree. `--frontend native` is a
Verilog-subset fallback when CIRCT is not installed.

ARTI A/B (needs a probe initramfs under `$WORK/initramfs.cpio.gz`; build via
`arti-work` probe setup or copy from a prior run):

```bash
./scripts/bench_arti_backends.sh
ROUNDS=3 ./scripts/bench_arti_backends.sh              # quieter mean wall
ARTI_MODEL_STATS=1 ./scripts/bench_arti_backends.sh   # tick/active counters
FLASHSIM_QEMU=/tmp/qemu-fs-new ./scripts/bench_arti_backends.sh
```

**GpuHostSystemAxi / OpenGPU probe** (real QEMU/Linux MMIO + job-queue self-test;
this is the go/no-go wall clock, not microbench MHz).

Measured 2026-09-17, quiet machine, `ROUNDS=3`,
`FLASHSIM_QEMU=/tmp/qemu-fs-new` (dmux-chunk embed at `b658f9b`) vs
`qemu-system-aarch64.verilator`:

| Backend | Mean wall | Per-round wall | Guest self-test |
|---|---|---|---|
| FlashSim | **6.3 s** | 8 / 6 / 5 s | 4.57 / 3.38 / 3.55 s |
| Verilator-linked QEMU | **19.0 s** | 19 / 19 / 19 s | 11.22 / 11.61 / 11.02 s |
| Speedup | **3.0×** | (VL wall / FS wall) | |

First FlashSim round is often cold (I-cache / codesign); warm rounds land
around 5–6 s wall / ~3.4–3.6 s guest (~3.2–3.8×). Re-run `ROUNDS=3` before
claiming a regression — single-run noise is ±1 s.

Recent GPU emit/opt wins on this path (correctness-checked vs Verilator on
unit benches):

- **Top-gate demand hoist** — hold entry evals only wires needed by outer
  gates/NBAs; mega coalescer SSA stays inside the arm.
- **Absorb `v & (…\|v\|…)`** — instruction-cache one-hot gates (~7 KB → ~110 B).
- **Sibling `&`/`\|` prefix CSE** — L2 long-if spines (12k → 2.5k nodes).
- **Mux chunk on NBA RHS** — coalescer `readData` is 8×63-deep priority muxes
  inline in holds; split into skip-cached `_dmux` chunks (SPLIT=32 / CHUNK=16)
  so early arms skip later evals.

Tried and **reverted** (no net win on quiet ARTI): cond-ranked mega promote,
hold-bucket 64, fanout-ranked promote, SETTLE_MIN=128 default, noinline outline
of coalescer hold arms, global dmux SPLIT=16 / CHUNK=8, Concat-part-only hoist
of vectorTlb `writeData` 16-deep lanes, heavy-hold isolation by ternary size,
threading `demanded` through wide/ternary emit (I-cache only), deferring deep
mux select demands out of `_emit_compute` / Concat entry (noisy win, quiet lose),
vectorDataCache 32-bucket skip partitions (+1.0 s wall vs dmux baseline),
L2 SSA shard colocation by `tid%65` (wall tie), L2 `always_inline` (clang -O2 hang),
vector ALU 32-bucket skip partitions (multiplyAlu/…; +1.0 s wall).

## Setup

```bash
# Verilator gold (already used as the functional oracle)
verilator --version

# Optional CIRCT toolchain, pinned to firtool-1.158.0
python3 -m flashsim setup-circt
python3 -m flashsim which-circt
```

Release tarballs go to `third_party/circt-release/` (gitignored). Override with
`FLASHSIM_CIRCT`.

## Experiment

```bash
python3 -m flashsim experiment
python3 -m flashsim experiment --frontend native   # skip CIRCT
python3 -m flashsim qemu-smoke                     # ARTI MMIO C API on GpuHostAxi
python3 -m flashsim arti-model rtl/gpu/host-axi/GpuHostAxi.sv -o build/GpuHostAxi
# GPU_SIM=flashsim ../gpu/scripts/run_arti_gpu.sh  # QEMU embeds FlashSim instead of Verilator
```

`experiment` uses CIRCT when `circt-verilog` is on `PATH` (or under
`third_party/circt-release/`). This generates circuits, compiles FlashSim and
Verilator, diffs 4096 cycles, then measures 1e6 cycles. The PASS gate is vs
1-thread Verilator (QEMU in-process eval is single-thread). The summary also
builds Verilator `--threads 4`, which is how people usually run a standalone
sim. More threads is not always faster; on GpuSystem `--threads 8` was slower
than 1 thread.

Skip / activity benches:

- `gated_pipe` — expensive `assign` mix sampled only when valid
- `sticky_input` — mix always sampled, `din` rarely changes
- `busy_alu` — mix always live, `din` changes every cycle
- `counter` — tiny, always-active control

Structural benches (CIRCT frontend; they exercise memory, `icmp`, hierarchy,
and register arrays):

- `sync_fifo` — 8-deep sync FIFO, `seq.firmem`, multiple outputs
- `cmp_acc` — `comb.icmp` / conditional update
- `hier_pipe` — `hw.instance` flattened into the parent
- `mini_rf` — 16×32 register file (`hw.array` + `seq.firreg`)

GPU slices from `rtl/gpu/` (Chisel → SystemVerilog, still compared to Verilator):

- `DrawContextFifo` — depth-4 draw-context queue, `clock`/`reset`, Decoupled ports
- `TriangleRasterizer` — 128×128 triangle setup/scan, 64-bit edges, widths up to 67 bits
- `WarpScheduler` — 4-warp round-robin issue/resume/finish
- `GpuCommandRouter` — host command/completion queues (491-bit packed descriptors)
- `BankedSharedMemory` — 4-lane 256-byte shared memory with atomics
- `GpuFrontend` — launch / fetch / decode pipe (open-loop IMEM)
- `InstructionCache` — 16-set 2-way fetch cache with MSHR, 8-byte lines
- `ScalarBackend` — integer execute pipe, RF, scoreboard (512-bit cache ports idle)
- `FrontendICache` — GpuFrontend fetch port wired to the 16×2 instruction cache
- `VectorBackend` — RVV execute pipe, vector RF, integer/FPU ALUs (memory/tex idle)
- `FpuBackend` — scalar FP32 issue / FMA / exact (memory idle)
- `FrontendScalar` — frontend fetch/decode wired to I$ and the scalar execute pipe
- `FrontendScalarFpu` — same closed loop plus the scalar FP32 FMA/exact pipe
- `Gpu` — one FlashSim-sized compute unit: kernel launch, GpuCore (scalar+vector+FPU+shared mem), 16×2 I$
- `GpuSystem` — command processor + one CU + shared L2 + idle DMA; DRAM refill at 64-byte lines
- `GpuHostAxi` — ARTI/QEMU control top: AXI4 slave + `m_irq`; one ID-register read, mem ports idle

`gated_pipe` and `sticky_input` must be ≥ 2× 1-thread Verilator. Other benches
must not be slower than 0.8× 1-thread. All must match Verilator at the dumped
ports.

Measured on this machine 2026-09-17 (CIRCT `firtool-1.158.0`, Verilator 5.050,
Apple clang `-O3`, 12 stages × 8 mix rounds, 1e6 cycles, ports match for 4096
cycles). GPU rows include vs 4-thread Verilator:

| bench | vs vlt 1T | vs vlt 4T | note |
|---|---|---|---|
| gated_pipe | **5.1×** | (tiny; MT overhead) | mix only runs when valid |
| sticky_input | **5.7×** | (tiny; MT overhead) | mix skipped while `din` unchanged |
| busy_alu | **1.2×** | | every mixer live every cycle |
| counter | **14×** | | tiny DUT |
| sync_fifo | **3.0×** | | idle cycles skip the write cone |
| cmp_acc | **3.7×** | | compares and gated updates |
| hier_pipe | **14×** | | inlined `add1` instance |
| mini_rf | **10×** | | sticky reads skip the file |
| DrawContextFifo | **1.4×** | | GPU slice; gated enq/retire |
| TriangleRasterizer | **2.4×** | | GPU slice; idle setup vs scan |
| WarpScheduler | **2.4×** | | SIMT issue idle when no eligible warp |
| GpuCommandRouter | **1.1×** | | packed queues; skip while engines idle |
| BankedSharedMemory | **1.3×** | | 4-lane 256 B SRAM + atomics |
| GpuFrontend | **1.5×** | | launch / fetch / decode; gated payload cones |
| InstructionCache | **1.0×** | | 16×2 fetch cache; 1-bit valid bits → `if` |
| FrontendICache | **1.7×** | | frontend fetch wired to 16×2 I$ |
| ScalarBackend | **3.6×** | | integer execute / RF / scoreboard; wide ports idle |
| VectorBackend | **38×** | | RVV execute / vector RF / ALUs; memory idle |
| FpuBackend | **11×** | | scalar FP32 issue / FMA / exact; wide ports idle |
| FrontendScalar | **11×** | | frontend+I$ closed through scalar ALU/redirect |
| FrontendScalarFpu | **16×** | | same loop plus scalar FP32 FMA |
| Gpu | **87×** | | closed CU; idle after warp finish |
| GpuSystem | **77×** | **364×** | 1 CU + L2 + DMA; FS ~5.5 MHz vs 1T 0.07 / 4T 0.015 MHz |
| GpuHostAxi | **17×** | **144×** | AXI ID read, then idle; FS ~32 MHz vs 1T 1.8 / 4T 0.22 MHz |

```bash
python3 -m flashsim compile build/benches/counter.v -o /tmp/counter.h
python3 -m flashsim compile rtl/gpu/draw-fifo/DrawContextFifo.sv -o /tmp/fifo.h --frontend circt
```
