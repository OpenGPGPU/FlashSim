#!/usr/bin/env python3
"""Build an opt-in recorder/replayer without regenerating the large DUT.

The output archive replaces only the ARTI wrapper. Link it into QEMU and set
ARTI_MODEL_TRACE to record; replay with `replay TRACE`. The original model and
QEMU staging files are untouched. Trace files are local native-endian data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--write-callback-bytes", type=int, choices=(1, 2, 4, 8, 16, 32, 64),
                   help="match the callback segmentation of the recorded wrapper")
    args = p.parse_args()
    model, out = args.model.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("arti_trace_model.cpp").resolve()
    wrapper = (model / "arti_rtl_model.cpp").read_text()
    default = re.search(r"#define ARTI_MODEL_WRITE_CALLBACK_BYTES\s+(\d+)", wrapper)
    if args.write_callback_bytes is not None and default is None and args.write_callback_bytes != 8:
        p.error("this older wrapper has fixed eight-byte write callbacks; regenerate it first")
    callback_bytes = args.write_callback_bytes or (int(default[1]) if default else 8)
    callback_flags = ([f"-DARTI_MODEL_WRITE_CALLBACK_BYTES={callback_bytes}"]
                      if default else [])
    obj = out / "arti_rtl_model.o"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fPIC", "-fPIE", "-w",
                    "-fbracket-depth=4096", *callback_flags, "-I", str(model),
                    f'-DARTI_WRAPPER_SOURCE="{model / "arti_rtl_model.cpp"}"',
                    "-c", str(source), "-o", str(obj)], check=True)
    archive = out / "libarti_rtl_model.a"
    shutil.copy2(model / archive.name, archive)
    subprocess.run(["ar", "r", str(archive), str(obj)], check=True)
    shutil.copy2(model / "arti_rtl_model.h", out)
    main_cpp = out / "main.cpp"
    main_cpp.write_text('''#include <cstdlib>
extern "C" int arti_trace_replay();
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  setenv("ARTI_MODEL_REPLAY",argv[1],1);
  return arti_trace_replay();
}
''')
    subprocess.run(["c++", "-std=c++17", "-O2", str(main_cpp), str(archive),
                    "-o", str(out / "replay")], check=True)
    inputs = [model / "libarti_rtl_model.a", model / "dut.h",
              model / "arti_rtl_model.cpp", source]
    manifest = {}
    for path in inputs:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        manifest[str(path)] = digest.hexdigest()
    (out / "trace-build-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out / "trace-configuration.json").write_text(json.dumps(
        {"write_callback_bytes": callback_bytes, "callback_compile_flags": callback_flags},
        indent=2)+"\n")
    print(out / "replay", flush=True)


if __name__ == "__main__":
    main()
