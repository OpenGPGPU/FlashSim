from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flashsim.arti_model import write_embedded_model
from flashsim.compile import compile_file
from flashsim.experiment import print_summary, run_experiment
from flashsim.qemu import run_qemu_smoke
from flashsim.toolchain import CIRCT_RELEASE, find_bin, setup_circt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flashsim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup-circt", help=f"download CIRCT {CIRCT_RELEASE} release binaries")
    p_compile = sub.add_parser("compile", help="compile Verilog subset to C++")
    p_compile.add_argument("verilog", type=Path)
    p_compile.add_argument("-o", "--output", type=Path, required=True)
    p_compile.add_argument("--frontend", choices=("auto", "circt", "native"), default="auto")
    p_exp = sub.add_parser("experiment", help="compare FlashSim against Verilator")
    p_exp.add_argument("--stages", type=int, default=12)
    p_exp.add_argument("--rounds", type=int, default=8)
    p_exp.add_argument("--frontend", choices=("auto", "circt", "native"), default="auto")
    p_qemu = sub.add_parser(
        "qemu-smoke",
        help="ARTI MMIO C API on GpuHostAxi: ID read, COLOR_BASE write/read",
    )
    p_qemu.add_argument("--frontend", choices=("auto", "circt", "native"), default="auto")
    p_arti = sub.add_parser(
        "arti-model",
        help="compile a GPU top and emit the ARTI embedded C API + QEMU build script",
    )
    p_arti.add_argument("verilog", type=Path)
    p_arti.add_argument("-o", "--output", type=Path, required=True)
    p_arti.add_argument("--top", default=None, help="defaults to the Verilog file stem")
    p_arti.add_argument("--frontend", choices=("auto", "circt", "native"), default="auto")
    sub.add_parser("which-circt", help="print CIRCT tool paths")

    args = parser.parse_args(argv)
    if args.cmd == "setup-circt":
        setup_circt()
        return 0
    if args.cmd == "which-circt":
        for name in ("firtool", "circt-opt", "arcilator", "circt-verilog"):
            print(f"{name}: {find_bin(name)}")
        return 0
    if args.cmd == "compile":
        compile_file(args.verilog, args.output, frontend=args.frontend)
        print(f"wrote {args.output}")
        return 0
    if args.cmd == "experiment":
        results = run_experiment(stages=args.stages, rounds=args.rounds, frontend=args.frontend)
        return print_summary(results)
    if args.cmd == "qemu-smoke":
        return run_qemu_smoke(frontend=args.frontend)
    if args.cmd == "arti-model":
        top = args.top or args.verilog.stem
        write_embedded_model(args.output, top, verilog=args.verilog, frontend=args.frontend)
        print(f"wrote FlashSim ARTI model for {top} to {args.output}")
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
