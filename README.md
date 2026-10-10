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

Large `GpuHostSystemAxi` models are emitted as C++ method shards and compiled
in parallel. Multi-CU RTL is detected from the generated source and uses a
32-way split by default; set `FLASHSIM_DUT_SHARDS` to pin the split count for
machines with a smaller memory or compiler parallelism budget. Set
`FLASHSIM_COMPILE_JOBS` to limit parallel shard compilation (including the
ARTI `build_embedded.sh` path); completed shard objects are reused between the
correctness and performance builds. Set
`FLASHSIM_REUSE_EMIT=1` when rerunning a benchmark against an unchanged
generated DUT to reuse the emitted C++ and skip the Python lowering pass. Set
`FLASHSIM_SHARD_OPT=-O0`/`-O1` for a quick multi-CU smoke build; the default is
`-O2`; the generated `dut_commit.cpp` shard is kept at O0 because it is a
large dirty-list switch with little effect on the normal runtime path.

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

- **O(dirty) dirty-flag reset** — the write worklist cleared its whole flag
  array every tick (`memset(_w, 0, 13369)` on `GpuHostSystemAxi`) to consume
  the one or two slots a live tick actually dirties. Resetting by walking the
  previous tick's list instead is O(dirty), not O(signals). Measured: idle
  ARTI path 328 → 175 ns/tick, scanout-active 540 → 426 ns/tick, unit benches
  +1.32x geomean (Gpu 1.44x, VectorBackend 1.32x, GpuHostAxi 1.33x,
  GpuSystem 1.18x).
- **Top-gate demand hoist** — hold entry evals only wires needed by outer
  gates/NBAs; mega coalescer SSA stays inside the arm.
- **Absorb `v & (…|v|…)`** — instruction-cache one-hot gates (~7 KB → ~110 B).
- **Sibling `&`/`|` prefix CSE** — L2 long-if spines (12k → 2.5k nodes).
- **Mux chunk on NBA RHS** — coalescer `readData` is 8×63-deep priority muxes
  inline in holds; split into skip-cached `_dmux` chunks (SPLIT=32 / CHUNK=16)
  so early arms skip later evals.

### Measuring the active phase

The unit benches and the ARTI wall clock are different machines. Every
speedup number above comes from a stimulus that leaves the GPU idle, and
holding the command stream open does not change that: `FLASHSIM_HOT=1`
resubmits the same kernel every cycle and measures 0.99–1.02x cold, because one
CU running `addi; fadd.s; cease` idles between launches either way. The
fixed-function scanout is what dominates the ARTI present phase, and it is
free-running by construction.

`arti_rtl_model_skip_stats()` exposes per-tick skip traffic (ticks, commit
slots, hold invocations, holds that wrote) so a driver can separate "the commit
path is expensive" from "the wake set is too wide to skip anything" without
knowing anything about the design. Driving the scanout registers over the
model's own MMIO path with it gives, on this machine:

| phase | ns/tick | commit/tick | holds/tick | holds that wrote/tick |
|---|---|---|---|---|
| quiescent | 152 | 0.00 | 0.00 | 0.00 |
| scanout running (64×64) | 385 | 1.00 | 2.00 | 1.00 |

Note the skip kernel is doing its job in the active phase — 2 hold
invocations and 1 commit per tick while the scanout runs — so the active cost
is the tick skeleton, not wasted re-evaluation.

### Where the ARTI wall clock is actually lost

Both ARTI backends expose the same C API and were built from the same RTL, so
linking one driver against each gives a like-for-like per-tick cost. Measured
on the scanout workload above (`TICKS` advanced exactly, not by timing
`check_irq`, which advances one tick on the Verilator backend but up to
`ARTI_MODEL_IRQ_PUMP` on this one):

| workload | FlashSim | Verilator (8 regions) | ratio |
|---|---|---|---|
| scanout running | **405 ns/tick** | 11,492 ns/tick | **~28×** |
| scanout disabled | **153 ns/tick** | 14,737 ns/tick | **~96×** |

The Verilator figure is the ARTI backend built by the normal flow, so it has 8
compiled eval regions; its runtime pool defaults to all cores
(`VerilatedContext::m_threads = getProcessDefaultParallelism()`), not to
`ARTI_VERILATOR_THREADS`, which only affects codegen.

Verilator's cost is flat at ~11.5 µs whether or not the GPU is doing anything,
because it re-evaluates the whole cone every cycle; FlashSim varies 3× with
activity. **So the end-to-end ARTI loss is not per-tick cost** — it is the
number of cycles advanced per host access. One guest MMIO read advances:

| scanout | ticks per MMIO read |
|---|---|
| disabled | 258 (settle early-exits at `SETTLE_MIN`) |
| running | 500,002 (the full `ARTI_MODEL_MMIO_ADVANCE_CYCLES` cap) |

`gpu_active()` was true whenever `_chg != 0`, and a free-running scanout commits
one state element per tick and issues **zero** AXI traffic, so activity never
went false and **every** settle burned its full 500k budget. Activity is now
`axi_busy() || (_chg && tail < ARTI_MODEL_CHG_TAIL)`: externally visible AXI
traffic keeps a real job alive indefinitely, while state changes get only a
bounded, per-settle budget to drain. The budget is reset at the start of every
settle — leaving it global lets it saturate once and starve every later settle,
silently capping a long AXI-free job at `SETTLE_MIN`.

| ticks advanced per guest MMIO read | before | after |
|---|---|---|
| scanout disabled | 258 | 258 (unchanged) |
| scanout running | 500,002 | **4,114** (122×) |

`ARTI_MODEL_SETTLE_NS` (default 0) additionally bounds one settle in wall-clock
time, so no activity policy can hold QEMU's BQL for a whole tick cap.

Note what this does and does not buy: it cuts *latency per host read* and the
per-read tick count, not the total simulated cycles a job needs. Total wall
clock is still (cycles the job needs) × (ns/tick). That is measured end to end
below.

### End to end: the Debian present workload

The scanout micro-benchmark above is fixed-function and leaves most of the GPU
cold. The real workload is `opengpu_pipe_blit` then `opengpu_pipe_present` on the
`debian-320x240` display profile, which has both shader cores on. Both backends
come from the same RTL and run the same ISO, timed over the serial console:

| stage | FlashSim | Verilator (8 regions) | ratio |
|---|---|---|---|
| blit | **0.130 s** | 22.777 s | **175×** |
| present | **129.606 s** | 353.030 s | **2.72×** |
| GPU test phase | **129.737 s** | 375.807 s | **2.90×** |
| whole run | 169.925 s | 422.461 s | 2.49× |

Both report `OPENGPU PIPE PRESENT PASS: 320x240 painted=38160 tint` and
`BENCH_RC=0`, so the two backends agree on the rendered pixels, not just on the
timing. The whole-run figure is the weakest one: past ~40 s it is measuring
Linux boot, not the GPU.

Present is 99.9% of the GPU test phase, so this is where the remaining work is,
and it is a real remaining loss rather than a measurement artifact. Two earlier
comparisons that looked far better were not: they ran a `pipe_present` that
failed immediately with `EOPNOTSUPP` on *both* backends, so they timed how long
each took to give up. The display profile turns the vertex core on by default,
and the driver rejects any submit whose vertex flag does not match the hardware
(`driver/opengpu_compute.c`), so `pipe_present` has to go through the vertex
path to draw anything at all.

The present phase is bound by per-tick cost, not by how fast the host can drive
the model. The ARTI device polls the model from a `QEMU_CLOCK_HOST` timer and
gives the model one tick per poll, so the poll rate looks like a floor on
wall clock. It is not — sweeping it over a 16× range moves the phase by 1.2%,
which is noise:

| IRQ poll interval | present |
|---|---|
| 25 µs | 137.83 s |
| 100 µs (default) | 139.44 s |
| 400 µs | 138.35 s |

Sweeping it required relinking the driver, which first produced a binary linked
against a stale model archive and a present phase that never finished; the
control run is what caught that. A model relink has to check that the archive in
`hw/misc/` is the one the build actually produced — `build_embedded.sh` only
copies it when the bytes differ, so a tree can keep serving an older model.

### Where the tick actually goes

Profiling the ARTI model under `sample` (scanout registers driven over MMIO,
same RTL as the integration build):

| | total samples | `tick_nba` | `host_host_*` evals |
|---|---|---|---|
| scanout running | 6614 | 33.3% | **52.3%** |
| scanout disabled | 6822 | 79.9% | **0.0%** |

The fixed-function cone is `host_host_core_rp_*` (ROP / clipper / texture).
`skip_partition_key` bucketed it by `host_host` alone, so **91 wires shared one
generation counter**: every leaf the display path touches invalidated all 91
`eval_*` methods, every tick. Switch the scanout off and they skip perfectly.
Splitting that cone by submodule plus a hash bucket takes the largest partition
from 91 wires to 23, costs 18 extra entries in the per-tick guard walk
(94 -> 112 partitions), and measured 1.32x on the scanout-active path and 1.73x
idle — but re-profiling moved `host_host_*` only 52.3% -> 47.6%, so the bucket
size was a symptom, not the cause. Those wires read a common set of leaves, so
one scanout state change invalidates every sub-bucket: the cone is genuinely
recomputing wide pixel work each cycle, and the lever there is the code the
compiler sees, not the invalidation granularity.

### Attributing the tick skeleton

`tick_nba` is 33% of a scanout-active tick and 80% of an idle one. Rebuilt one
shard with line tables and re-profiled; the self time resolves to:

| share of tick_nba | what |
|---|---|
| 15.6% | the hold dispatch scan: `for (w = 0; w < 70u; w++) bits = _h_need[w] \| _h_busy[w];` — every word, every tick, for a design that dispatches ~0.8 holds per tick |
| ~7% | guarded eval sites `if (X__ok != _pg[p]) eval_X();` |
| 4.7% | `if (__we0 \| __we1 \| ... \| __we108)` — a 109-way OR of mem-write enables, evaluated unconditionally before knowing whether any write can fire |
| 2.2% | `_commit()` |
| 0.6% | the O(dirty) dirty-flag reset |

The dispatch scan is now a summary-of-words bitmap (`_h_any`, one bit per hold
word), so the loop runs `ceil(nw/64)` iterations — 2 instead of 70 — and a word
is only touched when awake. Measured 1.13x idle, 1.05x scanout-active.

That change also exposed a gap in the test suite worth recording: the 14-bench
Verilator byte-diff **passed 14/14 on a build where the inline wake path
updated `_h_need` without arming `_h_any`** — every hold woken through a set of
at most `_BUMP_INLINE` holds simply never ran, and the ARTI scanout reported
inactive. The unit benches never wake a hold that way. `test_emit_wake.py` now
asserts the invariant directly (and the assertion is checked to fail when the
fix is removed).

Tried and **reverted** (no net win, measured not guessed):

- **Guard-homogeneous hold fusion.** Buckets are split by write partition but
  woken by guard partition, and on `GpuHostSystemAxi` 4334 hold methods share
  only 742 guard signatures (one signature is shared by 1024 methods), so
  fusing them cut the method count to 1432 and the wake-edge count 3.3x — and
  measured 0.97x geomean over 15 interleaved rounds (Gpu 0.93x, GpuSystem
  1.04x, GpuHostAxi 0.91x, VectorBackend 0.99x). Waking one statement then
  re-runs its whole fused bucket, and that coarser granularity costs more than
  the saved calls. The mismatch is real; a coarser bucket is the wrong lever.
- **Packed 64-bit word wake masks** (`FLASHSIM_WAKE_PACK`, off by default) —
  3.5x fewer wake-construction ops, measured 1.003x geomean, because commit
  only runs ~1 slot per tick so the whole path is under 1% of a cycle.
- **`-O2` for `dut_commit.cpp`** — `-O1`/`-O2` there compile in 50/70s, not
  "hours", so the stated rationale for the `-O0` is wrong. On the `vertex_draw`
  replay it is neutral: commit runs 0.67 slots/tick there. But "neutral" is
  workload-specific, and on the *scanout* path that a present actually spends
  its time in it is a large regression. Measured on the 320x240 model with the
  scanout free-running (1.00 commit/tick, the free-running-display case), Apple
  clang 16, M4 Max, ns/tick over three runs each:

  | `dut_commit.cpp` | idle | scanout active |
  |---|---|---|
  | `-O0` (current) | 180.7 | **199.4** |
  | `-O1` | 111.5 (-38%) | 190.1 (**+33% worse**) |
  | `-O2` | 114.3 (-37%) | 234.8 (**+64% worse**) |

  So `-O0` stays, and for a different reason than "compile time": raising it
  buys a large idle win and pays a much larger active loss, because the commit
  body bloats the hot tick path. Quote the workload with any commit-path
  number. (`-O3` on the shards is also a slight loss on this codegen: idle
  181.1 → 186.6, active 199.8 → 205.6.)
- cond-ranked mega promote, hold-bucket 64, fanout-ranked promote, SETTLE_MIN=128 default, noinline outline
  of coalescer hold arms, global dmux SPLIT=16 / CHUNK=8, Concat-part-only hoist
  of vectorTlb `writeData` 16-deep lanes, heavy-hold isolation by ternary size,
  threading `demanded` through wide/ternary emit (I-cache only), deferring deep
  mux select demands out of `_emit_compute` / Concat entry (noisy win, quiet lose),
  vectorDataCache 32-bucket skip partitions (+1.0 s wall vs dmux baseline),
  L2 SSA shard colocation by `tid%65` (wall tie), L2 `always_inline` (clang -O2 hang),
  vector ALU 32-bucket skip partitions (multiplyAlu/…; +1.0 s wall).

Also tried against the present workload above. These were measured by recording
a full guest run's model calls and replaying them against a candidate model,
checking every return value and every written byte against the recording, so a
behavioural regression shows up as a mismatch rather than as a faster number.
The recording was taken on the `vertex_draw` workload: 21.6 M ticks, 1.31 G hold
invocations, and the baseline candidate replays it at 7.85 µs/tick with zero
mismatches.

- **The giant `sharers` hold ladders.** Sixteen holds are ~6,100 lines each with
  864 `sharers_n =` assignments, and "body lines × executions" makes them look
  like half the cost. They are **0.8%**. They are nested `if`s: the outer
  condition is a large disjunction, so when it is false the whole ladder is
  skipped in O(1), and lines are not executions. Profiling the standalone replay
  binary instead of QEMU gives `tick_nba` 87.65%, all `_eager*` 10.42%, all
  holds 0.8%. Cost proxies only work on straight-line code.
- **`-O3` on the shards, and `-O2` for `dut_commit.cpp`** — 7.85 → 7.79 and
  7.85 → 7.76 µs/tick. Both noise against this replay, and the `-O2` costs 10.5
  minutes of compile. Re-measured against the scanout path rather than this
  replay, both are small losses — see the `dut_commit.cpp` entry above.
- **Smaller `_EAGER_CHUNK` (1024 → 256) for instruction locality** — exactly
  0%. A 10,711-line pass becomes four chunks of ≤2,689 and nothing moves.
- **Skipping the sequential guard list when nothing was invalidated.** Every
  guard is "has my page moved since I last ran", so a tick that invalidates
  nothing can skip all 878 of them behind one compare. Measured 7.85 → 7.81:
  pages are invalidated on essentially every tick (27.4 of 4,240 passes re-run),
  so the gate never fires.
- **Trimming the coarse `_up` wake channel.** 94 passes are woken when one L2
  state decode flips; 73 genuinely depend on it. The 21 spurious ones are small,
  so the reachable win is under 1%.
- **Within-function CSE of repeated wire expressions.** 21.5% of assignments are
  structurally repeated inside their function, but 53.6% of those repeats are
  single-operator (`!x`), where sharing costs the load it saves. The profitable
  bucket is the two-operator one, and it adds ~66k locals to a function that is
  already 24,000 cycles.
- **Bit-parallel evaluation.** The design does have lane structure — 91
  isomorphic families, 89 of them 256-way, from the L2 tag ways — but they cover
  only **10.5%** of wires (19.4% inside the hottest passes). Arithmetic would
  drop to ~0.9×, i.e. a 1.11× ceiling, in exchange for widening the expression
  layer's type system and validating lane correspondence.

That list is the shape of the remaining problem. Change detection is close to
optimal: 27.4 of 4,240 passes re-run per tick (0.6%), and the ones that re-run
have to. One L2 state decode bit has 111,005 transitive consumers and gates 4,083
of the 4,322 wires in the hottest pass, so that cone genuinely recomputes when
it changes — the dependency is dense and real, not an artifact of coarse
invalidation. At ~0.9 cycles per wire the emitted code is close to what scalar
bitwise evaluation costs.

Going further therefore means changing what has to be evaluated, not how fast it
is evaluated: `l2.slices[*].sharers` and `valid` are dynamically indexed
register vectors that firtool expands into hundreds of per-entry enable chains,
and modelling them as array-indexed writes is a frontend change. Nothing in
this file can substitute for it.

### PGO is available and is the only flag-level win left

The embedded build (`arti_model.py`'s `BUILD_EMBEDDED` template) uses plain
`-O2` on the shards, `-O0` on `dut_commit.cpp`, `-O1` on the wrapper, and no
`-march`/`-mcpu`. Of those, only the optimisation *level* has anything left in
it, and it is negative: `-O3` on the shards is a slight loss (see above), and
`-mcpu=native` measured inside the noise on the scanout path (active
210.0 → 203.3 ns/tick, idle unchanged) while costing build portability.

Two-phase PGO is the one thing that does pay. Compiling the shards with
`-fprofile-generate`, running the model, merging with `llvm-profdata`, then
recompiling with `-fprofile-use` measured (Apple clang 16, M4 Max, ns/tick,
three runs each, scanout free-running plus a quiet-MMIO phase):

| model | phase | `-O2` | PGO | delta |
|---|---|---:|---:|---:|
| 320x240 (shader cores on) | idle | 180.7 | 166.6 | **-7.8%** |
| 320x240 | scanout active | 199.4 | 184.2 | **-7.6%** |
| GpuHostSystemAxi 64x64 | idle | 123.9 | 115.5 | **-6.8%** |
| GpuHostSystemAxi 64x64 | scanout active | 143.4 | 132.6 | **-7.6%** |

It is deliberately *not* wired in, for two reasons. It doubles shard compile
time (two full passes over ~90 MB of emitted C++), and — the real blocker —
it needs a training workload, and `arti_smoke.cpp` is not one: it reads
`GPU_ID`, writes `color_base` and checks an IRQ, so a profile from it would
describe almost nothing. Training has to cover the paths that dominate a real
run (the free-running scanout, the raster/fragment path, the shader path), and
picking that workload is a design decision, not a build default. PGO also does
not threaten the byte-identical emitter output claimed below, since it changes
only the compiler's decisions, not the emitted C++.

Anyone wiring it up should train on more than the scanout alone: the scanout
path and the shader path barely share a hot branch, and a scanout-only profile
is what the numbers above actually are.

### Reproducible builds

The emitter used to be **non-deterministic**: two runs on identical input
produced C++ differing by ~64k lines, because set iteration over strings
follows `PYTHONHASHSEED` and that order reached the output through the signal
dict, the SSA-promotion caps, the deep-mux chunk budget, the invalidation
table indices and `eval_*` emission order. That guarantees a ccache miss on
every ARTI rebuild and makes a binary unreproducible from source. Fixed at the
sources (`IdSet` in `ir.py` iterates sorted; explicit `sorted()` at the cap and
index sites); `Gpu` and `GpuSystem` are now byte-identical across
`PYTHONHASHSEED=0/999/4242`. Cost of the fix: 0.994x geomean on unit benches,
with `GpuHostAxi` 0.949x because the promotion cap now breaks ties by name
instead of arbitrarily.

`hw.mlir` is cached in the *output directory* and freshness was judged by
source mtime alone, so compiling a second design into a directory that already
had a cache found every source older than the cache and silently reused the
wrong design. It is now keyed on the top module via a `hw.mlir.top` sidecar
(`compile.py:_mlir_matches`).


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

Those 2026-09-17 numbers predate the O(dirty) dirty-flag reset. Re-measured
2026-09-28 on the same benches (6 stages × 4 mix rounds, so absolute MHz is not
comparable run-to-run — use the A/B ratios in "Recent GPU emit/opt wins"):

| bench | vs vlt 1T | vs vlt 4T |
|---|---|---|
| GpuFrontend | **1.28×** | 15.6× |
| InstructionCache | **1.07×** | 12.7× |
| VectorBackend | **38.0×** | 61.9× |
| Gpu | **94.7×** | 62.3× |
| GpuSystem | **56.2×** | 67.4× |
| GpuHostAxi | **20.4×** | 230.9× |

```bash
python3 -m flashsim compile build/benches/counter.v -o /tmp/counter.h
python3 -m flashsim compile rtl/gpu/draw-fifo/DrawContextFifo.sv -o /tmp/fifo.h --frontend circt
```
