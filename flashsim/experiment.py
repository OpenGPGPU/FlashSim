from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from flashsim.benches import CIRCT_ONLY, write_benches
from flashsim.circt_frontend import CirctError, import_verilog_file, verilog_sources
from flashsim.compile import compile_file, _resolve_frontend
from flashsim.harness import flashsim_main, verilator_main
from flashsim.toolchain import repo_root

CHECK_CYCLES = 4096
PERF_CYCLES = 1_000_000

# Low-activity benches must beat Verilator by this factor.
# High-activity and structural benches must not fall below it.
CRITERIA = {
    "gated_pipe": 2.0,
    "sticky_input": 2.0,
    "busy_alu": 0.8,
    "counter": 0.8,
    "sync_fifo": 0.8,
    "cmp_acc": 0.8,
    "hier_pipe": 0.8,
    "mini_rf": 0.8,
    "DrawContextFifo": 0.8,
    "TriangleRasterizer": 0.8,
    "WarpScheduler": 0.8,
    "GpuCommandRouter": 0.8,
    "BankedSharedMemory": 0.8,
    "GpuFrontend": 0.8,
    "InstructionCache": 0.8,
    "FrontendICache": 0.8,
    "ScalarBackend": 0.8,
    "VectorBackend": 0.8,
    "FpuBackend": 0.8,
    "FrontendScalar": 0.8,
    "FrontendScalarFpu": 0.8,
    "Gpu": 0.8,
    "GpuSystem": 0.8,
    "GpuHostAxi": 0.8,
}

GPU_BENCHES = {
    "DrawContextFifo": Path("rtl/gpu/draw-fifo/DrawContextFifo.sv"),
    "TriangleRasterizer": Path("rtl/gpu/raster-quad/TriangleRasterizer.sv"),
    "WarpScheduler": Path("rtl/gpu/warp-scheduler/WarpScheduler.sv"),
    "GpuCommandRouter": Path("rtl/gpu/command-router/GpuCommandRouter.sv"),
    "BankedSharedMemory": Path("rtl/gpu/shared-mem/BankedSharedMemory.sv"),
    "GpuFrontend": Path("rtl/gpu/frontend/GpuFrontend.sv"),
    "InstructionCache": Path("rtl/gpu/icache/InstructionCache.sv"),
    "FrontendICache": Path("rtl/gpu/frontend-icache/FrontendICache.sv"),
    "ScalarBackend": Path("rtl/gpu/scalar-pipe/ScalarBackend.sv"),
    "VectorBackend": Path("rtl/gpu/vector-pipe/VectorBackend.sv"),
    "FpuBackend": Path("rtl/gpu/fpu-pipe/FpuBackend.sv"),
    "FrontendScalar": Path("rtl/gpu/frontend-scalar/FrontendScalar.sv"),
    "FrontendScalarFpu": Path("rtl/gpu/frontend-scalar-fpu/FrontendScalarFpu.sv"),
    "Gpu": Path("rtl/gpu/gpu/Gpu.sv"),
    "GpuSystem": Path("rtl/gpu/gpu-system/GpuSystem.sv"),
    "GpuHostAxi": Path("rtl/gpu/host-axi/GpuHostAxi.sv"),
}


@dataclass
class BenchResult:
    name: str
    correct: bool
    flash_mhz: float
    vlt_mhz: float
    speedup: float
    passed: bool
    detail: str = ""
    vlt_mt_mhz: float = 0.0
    speedup_mt: float = 0.0
    vlt_threads: int = 1


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def verilator_thread_count() -> int:
    """Eval-region target for the Verilator reference build.

    This is a *codegen* knob, not a runtime thread count: Verilator picks its
    own pool size at runtime (`VerilatedContext::m_threads =
    getProcessDefaultParallelism()`), so `--threads N` here only controls how
    the model is partitioned. It must match what the ARTI build requests, or
    the "vs N T" column compares against a configuration nobody runs.
    """
    return 8


def _need(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required tool not found: {name}")
    return path


def _compile_flash(
    work: Path, module: str, mode: str, cycles: int, *, harness: str | None = None
) -> Path:
    cxx = _need("c++")
    src = work / f"fs_{mode}.cpp"
    src.write_text(flashsim_main(harness or module, mode))
    out = work / f"fs_{mode}"
    common = [
        cxx,
        "-O3",
        "-std=c++17",
        "-Wno-parentheses-equality",
        f"-DCYCLES={cycles}ull",
        "-I",
        str(work),
    ]
    shards = sorted(work.glob("dut_*.cpp"))
    if shards:
        jobs = int(os.environ.get("FLASHSIM_COMPILE_JOBS", "4"))
        jobs = max(1, min(jobs, len(shards)))
        stamp = work / ".flashsim_shards.flags"
        shard_flags = list(common)
        shard_opt = os.environ.get("FLASHSIM_SHARD_OPT", "-O2")
        if shard_opt not in {"-O0", "-O1", "-O2", "-O3"}:
            raise ValueError("FLASHSIM_SHARD_OPT must be -O0, -O1, -O2 or -O3")
        # The commit shard is a large dirty-list switch.  Optimizing it adds
        # a disproportionate amount of compile time and has little runtime
        # effect because the list is short on the normal GPU path.  Keep it
        # at O0, matching build_embedded.sh, while the eval shards use the
        # requested optimization level.
        def shard_opt_for(shard: Path) -> str:
            return "-O0" if shard.name == "dut_commit.cpp" else shard_opt

        # CYCLES only appears in the tiny wrapper, so exclude it from the
        # shard cache key.  This lets check and perf modes share the expensive
        # generated-DUT objects while still invalidating on compiler flags.
        shard_key = "\n".join(
            [flag for flag in shard_flags if not flag.startswith("-DCYCLES=")]
            + ["shard_opt:" + shard_opt]
            + ["dut_commit.cpp:" + shard_opt_for(Path("dut_commit.cpp"))]
        )
        reuse = False
        try:
            header = work / "dut.h"
            reuse = stamp.read_text() == shard_key and all(
                shard.with_suffix(".o").is_file()
                and shard.with_suffix(".o").stat().st_mtime >= shard.stat().st_mtime
                and shard.with_suffix(".o").stat().st_mtime >= header.stat().st_mtime
                for shard in shards
            )
        except OSError:
            pass

        def compile_shard(shard: Path) -> subprocess.CompletedProcess[str]:
            obj = shard.with_suffix(".o")
            if reuse:
                return subprocess.CompletedProcess([], 0, "", "")
            flags = list(shard_flags)
            flags[1] = shard_opt_for(shard)
            # Match the ARTI build by default: the generated DUT shards are
            # large enough that -O3 adds substantial compile time.  A lower
            # level is useful for a quick multi-CU runtime smoke measurement.
            return _run([*flags, "-c", str(shard), "-o", str(obj)])

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(compile_shard, shards))
        failures = [r for r in results if r.returncode != 0]
        if failures:
            detail = "\n".join(r.stderr for r in failures)
            raise SystemExit(f"flashsim shard compile failed for {module}/{mode}:\n{detail}")
        if not reuse:
            stamp.write_text(shard_key)
        flags = [*common, str(src), *[str(s.with_suffix(".o")) for s in shards], "-o", str(out)]
        proc = _run(flags)
    else:
        proc = _run([*common, str(src), "-o", str(out)])
    if proc.returncode != 0:
        raise SystemExit(f"flashsim compile failed for {module}/{mode}:\n{proc.stderr}")
    return out


def _compile_verilator(
    work: Path,
    vfile: Path,
    module: str,
    mode: str,
    cycles: int,
    threads: int = 1,
    harness: str | None = None,
) -> Path:
    verilator = _need("verilator")
    src = work / f"vl_{mode}.cpp"
    src.write_text(verilator_main(harness or module, mode))
    mdir = work / f"vl_{mode}_t{threads}_obj" if threads > 1 else work / f"vl_{mode}_obj"
    proc = _run(
        [
            verilator,
            "--cc",
            "--exe",
            "--build",
            "-j",
            "0",
            "-O3",
            "--x-assign",
            "fast",
            "--x-initial",
            "fast",
            "--noassert",
            "--threads",
            str(threads),
            "-Wno-fatal",
            "--quiet-stats",
            "--top-module",
            module,
            "-Mdir",
            str(mdir),
            "-CFLAGS",
            f"-O3 -std=c++17 -DCYCLES={cycles}ull",
            *[str(p) for p in verilog_sources(vfile)],
            str(src),
        ]
    )
    if proc.returncode != 0:
        raise SystemExit(f"verilator compile failed for {module}/{mode}:\n{proc.stdout}\n{proc.stderr}")
    bin_path = mdir / f"V{module}"
    if not bin_path.exists():
        raise SystemExit(f"verilator did not produce {bin_path}")
    return bin_path


def _parse_mhz(text: str) -> float:
    for line in text.strip().splitlines()[::-1]:
        if "mhz=" in line:
            return float(line.split("mhz=")[1].split()[0])
    raise ValueError(f"no mhz= in output:\n{text}")


def run_bench(
    name: str,
    vfile: Path,
    work: Path,
    frontend: str = "auto",
    *,
    harness: str | None = None,
) -> BenchResult:
    work.mkdir(parents=True, exist_ok=True)
    out_h = work / "dut.h"
    if not (
        os.environ.get("FLASHSIM_REUSE_EMIT") == "1"
        and _compiled_dut_fresh(vfile, out_h)
    ):
        compile_file(vfile, out_h, frontend=frontend)

    fs_check = _compile_flash(work, name, "check", CHECK_CYCLES, harness=harness)
    vl_check = _compile_verilator(
        work, vfile, name, "check", CHECK_CYCLES, harness=harness
    )
    fs_dump = work / "fs_check.txt"
    vl_dump = work / "vl_check.txt"
    fs_out = _run([str(fs_check)])
    vl_out = _run([str(vl_check)])
    if fs_out.returncode != 0:
        return BenchResult(name, False, 0, 0, 0, False, fs_out.stderr)
    if vl_out.returncode != 0:
        return BenchResult(name, False, 0, 0, 0, False, vl_out.stderr)
    fs_dump.write_text(fs_out.stdout)
    vl_dump.write_text(vl_out.stdout)
    correct = filecmp.cmp(fs_dump, vl_dump, shallow=False)

    fs_perf = _compile_flash(work, name, "perf", PERF_CYCLES, harness=harness)
    vl_perf = _compile_verilator(
        work, vfile, name, "perf", PERF_CYCLES, threads=1, harness=harness
    )
    mt = verilator_thread_count()
    vl_mt = None
    if mt > 1:
        vl_mt = _compile_verilator(
            work,
            vfile,
            name,
            "perf_mt",
            PERF_CYCLES,
            threads=mt,
            harness=harness,
        )
    fs_p = _run([str(fs_perf)])
    vl_p = _run([str(vl_perf)])
    if fs_p.returncode != 0 or vl_p.returncode != 0:
        return BenchResult(name, correct, 0, 0, 0, False, fs_p.stderr + vl_p.stderr)
    flash_mhz = _parse_mhz(fs_p.stdout)
    vlt_mhz = _parse_mhz(vl_p.stdout)
    speedup = flash_mhz / vlt_mhz if vlt_mhz else 0.0
    vlt_mt_mhz = 0.0
    speedup_mt = 0.0
    if vl_mt is not None:
        mt_p = _run([str(vl_mt)])
        if mt_p.returncode == 0:
            vlt_mt_mhz = _parse_mhz(mt_p.stdout)
            speedup_mt = flash_mhz / vlt_mt_mhz if vlt_mt_mhz else 0.0
    need = CRITERIA[name]
    passed = correct and speedup >= need
    detail = f"need>={need:.1f}x vs 1T"
    if mt > 1 and vlt_mt_mhz:
        detail += f"; vs {mt}T={speedup_mt:.2f}x"
    return BenchResult(
        name,
        correct,
        flash_mhz,
        vlt_mhz,
        speedup,
        passed,
        detail,
        vlt_mt_mhz=vlt_mt_mhz,
        speedup_mt=speedup_mt,
        vlt_threads=mt if vl_mt is not None else 1,
    )


def _compiled_dut_fresh(vfile: Path, out_h: Path) -> bool:
    """Whether a previously emitted DUT can be reused for a benchmark run."""
    if not out_h.is_file():
        return False
    try:
        shards = list(out_h.parent.glob("dut_*.cpp"))
        outputs = [out_h, *shards]
        stamp = max(path.stat().st_mtime for path in outputs)
        tracked = [
            *verilog_sources(vfile),
            Path(__file__),
            Path(__file__).with_name("compile.py"),
            Path(__file__).with_name("emit.py"),
            Path(__file__).with_name("opt.py"),
            Path(__file__).with_name("ir.py"),
        ]
        if any(src.stat().st_mtime > stamp for src in tracked):
            return False
        # A split output must contain at least one source shard.  Monolithic
        # outputs are valid for small designs and need no extra check.
        return not shards or all(shard.is_file() for shard in shards)
    except OSError:
        return False


def run_experiment(stages: int = 12, rounds: int = 8, frontend: str = "auto") -> list[BenchResult]:
    root = repo_root()
    bench_dir = root / "build" / "benches"
    paths = write_benches(bench_dir, stages=stages, rounds=rounds)
    for name, rel in GPU_BENCHES.items():
        src = root / rel
        if src.exists():
            paths[name] = src
    results: list[BenchResult] = []
    print(f"frontend={frontend}  verilator_threads={verilator_thread_count()}")
    kind = _resolve_frontend(frontend)
    for name, vfile in paths.items():
        if kind == "native" and name in CIRCT_ONLY:
            print(f"== {name} ==")
            print("SKIP  native frontend cannot parse this bench")
            continue
        print(f"== {name} ==")
        result = run_bench(name, vfile, root / "build" / name, frontend=frontend)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status}  correct={result.correct}  "
            f"flash={result.flash_mhz:.3f} MHz  vlt1={result.vlt_mhz:.3f} MHz  "
            f"{result.speedup:.2f}x  {result.detail}"
        )
        if result.detail and not result.correct:
            print(result.detail)
    return results


def print_summary(results: list[BenchResult]) -> int:
    print()
    mt = max((r.vlt_threads for r in results), default=1)
    print(
        f"{'bench':<22} {'correct':<8} {'flash':>8} {'vlt1':>8} {'vs1T':>7} "
        f"{'vlt'+str(mt)+'T':>8} {'vsMT':>7} {'gate':>8}"
    )
    go = True
    for r in results:
        gate = "PASS" if r.passed else "FAIL"
        if not r.passed:
            go = False
        mt_mhz = f"{r.vlt_mt_mhz:8.3f}" if r.vlt_mt_mhz else f"{'—':>8}"
        mt_sp = f"{r.speedup_mt:6.2f}x" if r.vlt_mt_mhz else f"{'—':>7}"
        print(
            f"{r.name:<22} {str(r.correct):<8} {r.flash_mhz:8.3f} {r.vlt_mhz:8.3f} "
            f"{r.speedup:6.2f}x {mt_mhz} {mt_sp} {gate:>8}"
        )
    print()
    print("Gate is vs 1-thread Verilator (QEMU in-process eval is single-thread).")
    print(f"vlt{mt}T is Verilator --threads {mt}, the usual standalone sim setting.")
    if go:
        print("GO: activity skipping beats 1-thread Verilator on the agreed criteria.")
        return 0
    print("NO-GO: fix correctness or activity skipping before continuing.")
    return 1
