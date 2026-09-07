# GPU RTL snapshot

Generated from the sibling `gpu` Chisel tree (`../gpu` from the FlashSim
root) with CIRCT `firtool-1.158.0`. Slices plus the ARTI/QEMU control top
(`GpuHostAxi`).

```bash
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$(git rev-parse --show-toplevel)/third_party/circt-release/firtool-1.158.0/bin:$PATH"
cd ../gpu
sbt "runMain opengpu.elaboration.EmitPpaRtl raster-quad ../FlashSim/rtl/gpu/raster-quad"
sbt "runMain opengpu.elaboration.EmitPpaRtl draw-fifo ../FlashSim/rtl/gpu/draw-fifo"
sbt "runMain opengpu.elaboration.EmitPpaRtl warp-scheduler ../FlashSim/rtl/gpu/warp-scheduler"
sbt "runMain opengpu.elaboration.EmitPpaRtl command-router ../FlashSim/rtl/gpu/command-router"
sbt "runMain opengpu.elaboration.EmitPpaRtl shared-mem ../FlashSim/rtl/gpu/shared-mem"
sbt "runMain opengpu.elaboration.EmitPpaRtl gpu-frontend ../FlashSim/rtl/gpu/frontend"
sbt "runMain opengpu.elaboration.EmitPpaRtl icache ../FlashSim/rtl/gpu/icache"
sbt "runMain opengpu.elaboration.EmitPpaRtl scalar-pipe ../FlashSim/rtl/gpu/scalar-pipe"
sbt "runMain opengpu.elaboration.EmitPpaRtl frontend-icache ../FlashSim/rtl/gpu/frontend-icache"
sbt "runMain opengpu.elaboration.EmitPpaRtl vector-pipe ../FlashSim/rtl/gpu/vector-pipe"
sbt "runMain opengpu.elaboration.EmitPpaRtl fpu-pipe ../FlashSim/rtl/gpu/fpu-pipe"
sbt "runMain opengpu.elaboration.EmitPpaRtl frontend-scalar ../FlashSim/rtl/gpu/frontend-scalar"
sbt "runMain opengpu.elaboration.EmitPpaRtl frontend-scalar-fpu ../FlashSim/rtl/gpu/frontend-scalar-fpu"
sbt "runMain opengpu.elaboration.EmitPpaRtl gpu ../FlashSim/rtl/gpu/gpu"
sbt "runMain opengpu.elaboration.EmitPpaRtl gpu-system ../FlashSim/rtl/gpu/gpu-system"
sbt "runMain opengpu.elaboration.EmitGpuHostAxi ../FlashSim/rtl/gpu/host-axi"
```

| directory | Chisel top | why this slice |
|---|---|---|
| `raster-quad/` | `TriangleRasterizer` 128×128 | graphics fixed-function; 64-bit edge values |
| `draw-fifo/` | `DrawContextFifo` depth 4 | draw-context queue; arrays and Decoupled handshake |
| `warp-scheduler/` | `WarpScheduler` 4 warps × 4 lanes | SIMT round-robin issue / resume / finish |
| `command-router/` | `GpuCommandRouter` | host command and completion queues |
| `shared-mem/` | `BankedSharedMemory` 4 lanes, 256 B | banked SRAM + atomics |
| `frontend/` | `GpuFrontend` 4 warps × 4 lanes | fetch, decode, SIMT control |
| `icache/` | `InstructionCache` 16×2, 8 B lines | frontend fetch cache + MSHR |
| `scalar-pipe/` | `ScalarBackend` 4 warps × 4 lanes | integer execute, RF, scoreboard |
| `frontend-icache/` | `FrontendICache` 4 warps × 4 lanes | frontend fetch wired to 16×2 I$ |
| `vector-pipe/` | `VectorBackend` 4 warps × 4 lanes | RVV integer/FPU execute, vector RF |
| `fpu-pipe/` | `FpuBackend` 4 warps × 4 lanes | scalar FP32 issue / FMA / exact |
| `frontend-scalar/` | `FrontendScalar` 4 warps × 4 lanes | frontend+I$ closed through scalar execute |
| `frontend-scalar-fpu/` | `FrontendScalarFpu` 4 warps × 4 lanes | same loop plus scalar FP32 FMA |
| `gpu/` | `Gpu` 4 warps × 4 lanes | closed compute unit: kernel dispatch, core, I$, FPU |
| `gpu-system/` | `GpuSystem` 1 CU | command processor, L2, DMA engines; DRAM at 64 B lines |
| `host-axi/` | `GpuHostAxi` 16×16, 2 warps × 4 lanes | ARTI/QEMU AXI4 slave + irq; graphics mem ports idle |

`python3 -m flashsim experiment --frontend circt` compiles these tops and diffs
ports against Verilator. Ignore the extra `verification/` SV layers. Multi-file
blocks (`command-router/`) pass every `*.sv` next to the top.
