# FlashSim

Cycle-accurate RTL simulation that must match Verilator at DUT ports, and beat
it by skipping inactive combinational cones. The skip kernel is frozen at the
go/no-go; `GpuHostAxi` is the QEMU/ARTI control top now in the experiment.

## Goal

| Constraint | Rule |
|---|---|
| Gold model | Same Verilog, same stimulus, same per-cycle `dout` as Verilator |
| Speed | Low-activity ≥ 2× **1-thread** Verilator; busy ≥ 0.8×. Also report `--threads 4`. |
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

Measured on this machine (CIRCT `firtool-1.158.0`, Verilator 5.050, Apple clang
`-O3`, 12 stages × 8 mix rounds, 1e6 cycles, ports match for 4096 cycles).
GPU rows include vs 4-thread Verilator where measured:

| bench | vs vlt 1T | vs vlt 4T | note |
|---|---|---|---|
| gated_pipe | ~6.7× | n/a (tiny DUT; threads add overhead) | mix only runs when valid |
| sticky_input | ~3.1× | n/a | mix skipped while `din` is unchanged |
| busy_alu | ~1.2× | n/a | every mixer live every cycle |
| counter | ~50× | n/a | tiny DUT |
| sync_fifo | ~4.0× | | idle cycles skip the write cone |
| cmp_acc | ~11× | | compares and gated updates |
| hier_pipe | ~28× | | inlined `add1` instance |
| mini_rf | ~8.6× | | sticky reads skip the file |
| DrawContextFifo | ~1.6× | | GPU slice; gated enq/retire |
| TriangleRasterizer | ~2.7× | | GPU slice; idle setup vs scan |
| WarpScheduler | ~2.6× | | SIMT issue idle when no eligible warp |
| GpuCommandRouter | ~1.3× | | packed queues; skip while engines idle |
| BankedSharedMemory | ~1.4× | | 4-lane 256 B SRAM + atomics |
| GpuFrontend | ~1.1× | | launch / fetch / decode; gated payload cones |
| InstructionCache | ~1.1× | | 16×2 fetch cache; 1-bit valid bits lowered to `if` |
| ScalarBackend | ~2.7× | | integer execute / RF / scoreboard; 512-bit cache ports idle |
| FrontendICache | ~1.3× | | frontend fetch wired to 16×2 I$; open-loop line refill |
| VectorBackend | ~2.5× | | RVV execute / vector RF / integer+FPU ALUs; memory idle |
| FpuBackend | ~2.0× | | scalar FP32 issue / FMA / exact; 512-bit cache ports idle |
| FrontendScalar | ~1.2× | | frontend+I$ closed through scalar ALU/redirect |
| FrontendScalarFpu | ~2.5× | | same loop plus scalar FP32 FMA; mixed addi/fadd IMEM |
| Gpu | ~69× | | closed CU; idle after warp finish (8T Verilator was slower than 1T) |
| GpuSystem | ~53× | **~43×** | 1 CU + L2 + DMA; FlashSim ~5.9 MHz vs 1T 0.11 / 4T 0.14 MHz |
| GpuHostAxi | ~22× | **~167×** | AXI ID read, then idle; FlashSim ~39 MHz vs 1T 1.7 / 4T 0.23 MHz |

```bash
python3 -m flashsim compile build/benches/counter.v -o /tmp/counter.h
python3 -m flashsim compile rtl/gpu/draw-fifo/DrawContextFifo.sv -o /tmp/fifo.h --frontend circt
```
