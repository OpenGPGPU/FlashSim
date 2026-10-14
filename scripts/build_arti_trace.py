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
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    model, out = args.model.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("arti_trace_model.cpp").resolve()
    obj = out / "arti_rtl_model.o"
    subprocess.run(["c++", "-std=c++17", "-O1", "-fPIC", "-fPIE", "-w",
                    "-fbracket-depth=4096", "-I", str(model),
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
    manifest = {str(path): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                for path in inputs}
    (out / "trace-build-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(out / "replay", flush=True)


if __name__ == "__main__":
    main()
