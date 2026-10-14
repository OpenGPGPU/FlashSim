"""Disabled memory ports may defer cones without losing subsequent updates."""
import shutil
import subprocess

import pytest

from flashsim.emit import emit_cpp
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
