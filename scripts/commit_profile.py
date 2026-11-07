#!/usr/bin/env python3
"""Profile _commit hot writes, then bench the hoisted build.

Usage (from repo root):
  python3 scripts/commit_profile.py GpuSystem [--cycles 1000000]

Steps: generate a FLASHSIM_COMMIT_PROFILE=1 build, run perf, print
top IDs, rebuild with FLASHSIM_COMMIT_HOT=<top names>, run perf again,
and print warm-median MHz for both (first run discarded as cold).
The HOT list is per-model/per-workload; do not hardcode it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def _run_perf(work: Path, exe: Path, runs: int = 5) -> list[float]:
    from flashsim.experiment import _parse_mhz

    out = []
    for _ in range(runs):
        proc = subprocess.run([str(exe)], text=True, capture_output=True)
        out.append(_parse_mhz(proc.stdout))
    return out


def _build_and_profile(name: str, cycles: int) -> tuple[str, Path]:
    from flashsim.compile import compile_file
    from flashsim.experiment import GPU_BENCHES, _compile_flash

    vfile = ROOT / GPU_BENCHES[name]
    work = ROOT / "build" / name
    env = dict(os.environ)
    env["FLASHSIM_COMMIT_PROFILE"] = "1"
    subprocess.run(
        [sys.executable, "-c",
         f"from pathlib import Path; from flashsim.compile import compile_file; compile_file(Path({str(vfile)!r}), Path({str(work / 'dut.h')!r}), frontend='circt')"],
        env=env,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )
    exe = _compile_flash(work, name, "perf", cycles)
    src = (work / "fs_perf.cpp").read_text()
    dump = (
        "\n  {\n"
        "    std::vector<std::pair<unsigned long long,int>> tops;\n"
        "    for (unsigned k=0;k<sizeof(dut._commit_counts)/sizeof(dut._commit_counts[0]);k++)\n"
        "      if (dut._commit_counts[k]) tops.push_back({dut._commit_counts[k], (int)k});\n"
        "    std::sort(tops.begin(), tops.end(), [](auto &a, auto &b){return a.first>b.first;});\n"
        '    printf("COMMIT_PROFILE total=%llu top=%zu\\n", (unsigned long long)dut._cc, tops.size());\n'
        "    for (size_t t=0;t<tops.size()&&t<10;t++)\n"
        '      printf("  [%d] %s=%llu\\n", tops[t].second, dut._commit_names[tops[t].second], tops[t].first);\n'
        "  }\n"
    )
    patched = src.replace("return 0;", dump + "  return 0;")
    if "#include <vector>" not in patched:
        patched = patched.replace("#include <cstdio>", "#include <cstdio>\n#include <vector>\n#include <algorithm>\n#include <utility>")
    (work / "fs_perf.cpp").write_text(patched)
    objs = [str(s.with_suffix(".o")) for s in sorted(work.glob("dut_*.cpp"))]
    cxx = "/usr/bin/c++"
    common = [cxx, "-O3", "-std=c++17", f"-DCYCLES={cycles}ull", "-I", str(work),
              str(work / "fs_perf.cpp"), *objs, "-o", str(work / "fs_perf_prof")]
    link = subprocess.run(common, text=True, capture_output=True)
    if link.returncode:
        raise SystemExit(f"profile link failed:\n{link.stderr[-3000:]}")
    proc = subprocess.run([str(work / "fs_perf_prof")], text=True, capture_output=True)
    print(proc.stdout[-3000:])
    top_names: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("["):
            try:
                name_part = line.split("]")[1].strip().split("=")[0]
            except IndexError:
                continue
            if name_part:
                top_names.append(name_part)
    top_names = sorted(set(top_names[:2]))
    return ",".join(top_names), exe


def _build_with_hot(name: str, hot: str, cycles: int):
    from flashsim.compile import compile_file
    from flashsim.experiment import GPU_BENCHES, _compile_flash

    vfile = ROOT / GPU_BENCHES[name]
    work = ROOT / "build" / name
    env = dict(os.environ)
    env["FLASHSIM_COMMIT_HOT"] = hot
    subprocess.run(
        [sys.executable, "-c",
         f"from pathlib import Path; from flashsim.compile import compile_file; compile_file(Path({str(vfile)!r}), Path({str(work / 'dut.h')!r}), frontend='circt')"],
        env=env,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )
    return _compile_flash(work, name, "perf", cycles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--cycles", type=int, default=1_000_000)
    args = parser.parse_args()
    hot, _ = _build_and_profile(args.name, args.cycles)
    if not hot:
        print("no hot writes found; nothing to hoist")
        return 0
    print(f"hot={hot}")
    # Baseline (profile build already has counters, which add a few
    # increments per commit); rebuild clean baseline for a fair median.
    from flashsim.compile import compile_file
    from flashsim.experiment import GPU_BENCHES, _compile_flash

    vfile = ROOT / GPU_BENCHES[args.name]
    work = ROOT / "build" / args.name
    env = dict(os.environ)
    env.pop("FLASHSIM_COMMIT_PROFILE", None)
    env["FLASHSIM_COMMIT_HOT"] = ""
    subprocess.run(
        [sys.executable, "-c",
         f"from pathlib import Path; from flashsim.compile import compile_file; compile_file(Path({str(vfile)!r}), Path({str(work / 'dut.h')!r}), frontend='circt')"],
        env=env,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
    )
    base_exe = _compile_flash(work, args.name, "perf", args.cycles)
    hot_exe = _build_with_hot(args.name, hot, args.cycles)
    base = _run_perf(work, base_exe)
    hot_runs = _run_perf(work, hot_exe)
    base_med = _median(base[1:])
    hot_med = _median(hot_runs[1:])
    print(f"baseline warm median {base_med:.3f} MHz (all {base})")
    print(f"hot={hot} warm median {hot_med:.3f} MHz (all {hot_runs})")
    if base_med:
        print(f"delta {(hot_med / base_med - 1) * 100:+.2f}%")
    print(json.dumps({"name": args.name, "hot": hot, "baseline_median": base_med, "hot_median": hot_med}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
