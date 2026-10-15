"""Disabled memory ports may defer cones without losing subsequent updates."""
import shutil
import subprocess

import pytest

from flashsim.emit import emit_cpp, split_dut_methods
from flashsim.ir import AlwaysFF, Assign, BinOp, Const, Id, MemRead, MemWrite, Module, Signal


def test_port_cones_share_preedge_data_and_wake_after_disabled_ticks(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    signals = {name: Signal(name,width,kind,depth=depth)
               for name,width,kind,depth in [
                   ("clock",1,"input",0),("data",32,"input",0),
                   ("addr",3,"input",0),("en0",1,"input",0),("en1",1,"input",0),
                   ("x",32,"output",0),("y",32,"output",0),
                   ("enabled0",1,"wire",0),("enabled1",1,"wire",0),
                   ("m",32,"mem",8)]}
    mod = Module(name="PortDemand",signals=signals,
        assigns=[Assign("x",BinOp("+",Id("data"),Const(7))),
                 Assign("y",BinOp("+",MemRead("m",Id("addr")),Id("x"))),
                 Assign("enabled0",Id("en0")),Assign("enabled1",Id("en1"))],
        always=AlwaysFF("clock",[]),ports=[n for n,s in signals.items() if s.kind in ("input","output")],
        mem_writes=[MemWrite("m",Id("addr"),Id("x"),Id("enabled0")),
                    MemWrite("m",Id("addr"),Id("y"),Id("enabled1"))])
    (tmp_path / "dut.h").write_text(emit_cpp(mod))
    main = tmp_path / "main.cpp"
    main.write_text('''#include "dut.h"
int main() {
  PortDemandDut d;unsigned expected[8]={};
  for(unsigned i=0;i<256;i++) {
    d.data=i*11;d.addr=(i/4)%8;
    // Long disabled intervals, port alternation, and simultaneous writes to
    // the same address. Port 1 must read pre-edge memory even if port 0 writes.
    d.en0=(i%13==0)||(i%5==0);d.en1=(i%13==0)||(i%7==0);
    unsigned x=d.data+7,y=expected[d.addr]+x;
    if(d.en0)expected[d.addr]=x;
    if(d.en1)expected[d.addr]=y;
    d.tick();
    for(unsigned j=0;j<8;j++)if(d.m[j]!=expected[j])return i+1;
  }
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler,"-std=c++17","-O2",str(main),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_outlined_ports_read_preedge_memory_before_ordered_stores(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    signals = {name: Signal(name,width,kind,depth=depth)
               for name,width,kind,depth in [
                   ("clock",1,"input",0),("addr",2,"input",0),("en",1,"input",0),
                   ("enabled",1,"wire",0),("old",32,"output",0),("m",32,"mem",4)]}
    mod = Module(name="ManyPorts",signals=signals,
        assigns=[Assign("enabled",Id("en")),Assign("old",MemRead("m",Id("addr")))],
        always=AlwaysFF("clock",[]),ports=["clock","addr","en","old"],
        mem_writes=[MemWrite("m",Id("addr"),BinOp("+",Id("old"),Const(i)),Id("enabled"))
                    for i in range(64)])
    header,parts = split_dut_methods(emit_cpp(mod),n_shards=2)
    assert "dut_mem_ports.cpp" in dict(parts)
    assert "void _mem_port_63();" in header
    (tmp_path / "dut.h").write_text(header)
    sources = []
    for name,source in parts:
        path = tmp_path / name
        path.write_text(source)
        sources.append(str(path))
    main = tmp_path / "main.cpp"
    main.write_text('''#include "dut.h"
int main() {
  ManyPortsDut d;unsigned expected[4]={};
  for(unsigned i=0;i<100;i++) {
    d.addr=i%4;d.en=i%5!=0;
    if(d.en)expected[d.addr]+=63;
    d.tick();
    for(unsigned j=0;j<4;j++)if(d.m[j]!=expected[j])return i+1;
  }
}
''')
    binary = tmp_path / "check"
    subprocess.run([compiler,"-std=c++17","-O2",str(main),*sources,"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_enable_regions_are_not_demanded_twice_before_commit(tmp_path):
    # More than 128 cached outputs activates eager regions. Keep a direct
    # input enable as well, so eliminating duplicate region calls must not
    # accidentally reuse the previous tick's enable value.
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    signals = {"clock": Signal("clock",1,"input"),
               "en": Signal("en",1,"input"),
               "data": Signal("data",32,"input"),
               "m": Signal("m",32,"mem",depth=4),
               "enabled": Signal("enabled",1,"wire")}
    assignments = [Assign("enabled",Id("en"))]
    for i in range(129):
        name = f"out{i}"
        signals[name] = Signal(name,32,"output")
        assignments.append(Assign(name,BinOp("+",Id("data"),Const(i))))
    mod = Module(name="EnableDemand",signals=signals,assigns=assignments,
                 always=AlwaysFF("clock",[]),ports=["clock","en","data"],
                 mem_writes=[MemWrite("m",Const(0),Id("data"),Id("enabled")),
                             MemWrite("m",Const(1),Id("data"),Id("en"))])
    source = emit_cpp(mod)
    assert "_demand_seq()" in source
    assert "_demand_mem();" not in source
    (tmp_path/"dut.h").write_text(source)
    main = tmp_path/"main.cpp"
    main.write_text('''#include "dut.h"
int main() {
  EnableDemandDut d;unsigned expected=0;
  for(unsigned i=0;i<100;i++) {
    d.en=(i%7)==0;d.data=i;
    if(d.en)expected=i;
    d.tick();
    if(d.m[0]!=expected || d.m[1]!=expected)return 1;
    for(unsigned j=0;j<4;j++)if(j>1 && d.m[j])return 2;
  }
}
''')
    binary=tmp_path/"check"
    subprocess.run([compiler,"-std=c++17","-O2",str(main),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_drained_enable_passes_still_prepare_and_wake_memory(tmp_path):
    from flashsim.ir import If, NbAssign
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    signals = {"clock":Signal("clock",1,"input"),"en":Signal("en",1,"input"),
               "data":Signal("data",32,"input"),"enabled":Signal("enabled",1,"wire"),
               "m":Signal("m",32,"mem",depth=4)}
    assignments = [Assign("enabled",Id("en"))]
    body = []
    for i in range(256):
        reg=f"r{i}"
        signals[reg]=Signal(reg,32,"reg")
        body.append(If(Id("enabled"),[NbAssign(reg,BinOp("+",Id("data"),Const(i)))]))
    for i in range(129):
        name=f"out{i}"
        signals[name]=Signal(name,32,"output")
        assignments.append(Assign(name,BinOp("+",Id("data"),Const(i))))
    mod=Module(name="DrainedEnable",signals=signals,assigns=assignments,
               always=AlwaysFF("clock",body),ports=["clock","en","data"],
               mem_writes=[MemWrite("m",Const(0),Id("data"),Id("enabled"))])
    source=emit_cpp(mod)
    assert "kPasses[]" in source
    assert "_demand_mem();" not in source
    assert "_demand_seq();" not in source
    assert "_mem_prepare()" in source
    (tmp_path/"dut.h").write_text(source)
    main=tmp_path/"main.cpp"
    main.write_text('''#include "dut.h"
int main() {
  DrainedEnableDut d;unsigned expected=0;bool written=false;
  for(unsigned i=0;i<100;i++) {
    d.en=(i%7)==0;d.data=i;
    if(d.en){expected=i;written=true;}
    d.tick();
    if(d.m[0]!=expected)return 1;
    if(d.r0!=(written?expected:0))return 2;
    if(d.r255!=(written?expected+255:0))return 3;
  }
}
''')
    binary=tmp_path/"check"
    subprocess.run([compiler,"-std=c++17","-O2",str(main),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
