"""Build workers must stay bounded, fill idle slots, and propagate failure."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from flashsim.arti_model import BUILD_EMBEDDED


def build_fixture(tmp_path, fail=""):
    if not shutil.which("ar") or not shutil.which("xargs"):
        pytest.skip("archive/build tools unavailable")
    model=tmp_path/"model with spaces"
    model.mkdir()
    for name in ("dut.h","arti_rtl_model.h","arti_rtl_model.cpp",
                 "dut_0.cpp","dut_1.cpp","dut_2.cpp","dut_commit.cpp",
                 "dut_commit_helpers.cpp"):
        (model/name).write_text("// fixture\n")
    (model/"dut_commit_helpers.cpp").write_text("// largest source\n"*100)
    script=model/"build_embedded.sh"
    script.write_text(BUILD_EMBEDDED.replace("__TOP__","Fixture"))
    bindir=tmp_path/"bin"
    bindir.mkdir()
    compiler=bindir/"c++"
    compiler.write_text("#!"+sys.executable+"\n"+'''import os,sys,time,json,fcntl
from pathlib import Path
src=sys.argv[sys.argv.index("-c")+1]
def record(event):
    with open(os.environ["BUILD_EVENTS"],"a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps({"event":event,"source":src,"args":sys.argv[1:],"time":time.monotonic()})+"\\n")
record("start")
time.sleep(.7 if src=="dut_0.cpp" else .05)
record("end")
if src==os.environ.get("FAIL_SOURCE"):sys.exit(7)
Path(sys.argv[sys.argv.index("-o")+1]).write_bytes(b"fixture object")
''')
    compiler.chmod(0o755)
    stage=tmp_path/"stage"
    (stage/"hw/misc").mkdir(parents=True)
    events=tmp_path/"events.jsonl"
    env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ["PATH"],
             CCACHE_DISABLE="1",QEMU_SRC=str(stage),SKIP_QEMU_REBUILD="1",
             FLASHSIM_COMPILE_JOBS="2",BUILD_EVENTS=str(events),FAIL_SOURCE=fail)
    # Test queue and profile argument quoting without invoking a real compiler.
    profile=tmp_path/"profile with spaces.profdata"
    profile.touch()
    env["FLASHSIM_PGO_PROFILE"]=str(profile)
    env.pop("FLASHSIM_PGO_GENERATE",None)
    result=subprocess.run(["bash",str(script)],env=env,capture_output=True,text=True)
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    return model,stage,result,rows,profile


def test_build_queue_reuses_a_slot_before_the_oldest_job_finishes(tmp_path):
    model,stage,result,rows,profile=build_fixture(tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert (stage/"hw/misc/libarti_rtl_model.a").exists()
    active=peak=0
    for row in rows:
        active+=1 if row["event"]=="start" else -1
        peak=max(peak,active)
        assert 0<=active<=2
    assert peak==2 and active==0
    assert rows[0]["source"]=="dut_commit_helpers.cpp"
    members=subprocess.check_output(["ar","-t",str(model/"libarti_rtl_model.a")],text=True).splitlines()
    expected=[p.with_suffix(".o").name for p in sorted(model.glob("dut_*.cpp"))]+["arti_rtl_model.o"]
    assert [name for name in members if not name.startswith("__.SYMDEF")]==expected
    times={(row["source"],row["event"]):row["time"] for row in rows}
    assert times["dut_2.cpp","start"]<times["dut_0.cpp","end"]
    for row in rows:
        if row["event"]!="start":continue
        assert "-fprofile-instr-use="+str(profile) in row["args"]
        expected="-O0" if row["source"]=="dut_commit.cpp" else "-O1" if row["source"]=="arti_rtl_model.cpp" else "-O2"
        assert expected in row["args"]
    assert (model/"libarti_rtl_model.a").exists()


def test_failed_compilation_cannot_publish_an_archive(tmp_path):
    model,stage,result,rows,profile=build_fixture(tmp_path,fail="dut_1.cpp")
    assert result.returncode!=0
    assert not (model/"libarti_rtl_model.a").exists()
    assert not (stage/"hw/misc/libarti_rtl_model.a").exists()
    assert not any(row["source"]=="arti_rtl_model.cpp" for row in rows)
