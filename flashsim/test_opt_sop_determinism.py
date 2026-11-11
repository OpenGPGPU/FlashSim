"""SOP term sets must not leak Python hash order into generated shards."""
import os
from pathlib import Path
import random
import subprocess
import sys


def test_sop_order_is_reproducible_and_preserves_truth_table():
    code = '''from flashsim.opt import _merge_sop_terms
from flashsim.ir import BinOp, Id, UnaryOp, Signal
a,b,c,d,e,f=map(Id,"abcdef")
x=BinOp("|",BinOp("|",BinOp("|",BinOp("&",a,b),BinOp("&",a,UnaryOp("!",b))),BinOp("&",c,f)),BinOp("&",d,e))
s={n:Signal(n,1,"input") for n in "abcdef"};s["out"]=Signal("out",1,"output")
y=_merge_sop_terms({"out":x},s)["out"]
def value(expr, inputs):
    if isinstance(expr,Id):return inputs[expr.name]
    if isinstance(expr,UnaryOp):return not value(expr.a,inputs)
    if isinstance(expr,BinOp):
        av,bv=value(expr.a,inputs),value(expr.b,inputs)
        return av & bv if expr.op=="&" else av | bv
    raise AssertionError(expr)
for bits in range(64):
    inputs={n:bool(bits & (1<<i)) for i,n in enumerate("abcdef")}
    assert value(x,inputs)==value(y,inputs)
print(repr(y))
# Exercise the >64-term simplifier as well as the small absorption pass.
names=["g"]+["p"+str(i) for i in range(7)]
s={n:Signal(n,1,"input") for n in names};s["out"]=Signal("out",1,"output")
terms=[]
for bits in range(65):
    term=Id("g")
    for i in range(7):
        atom=Id("p"+str(i))
        term=BinOp("&",term,atom if bits & (1<<i) else UnaryOp("!",atom))
    terms.append(term)
x=terms[0]
for term in terms[1:]:x=BinOp("|",x,term)
y=_merge_sop_terms({"out":x},s)["out"]
for bits in range(256):
    inputs={n:bool(bits & (1<<i)) for i,n in enumerate(names)}
    assert value(x,inputs)==value(y,inputs)
print(repr(y))
'''
    root = Path(__file__).resolve().parents[1]
    results = [subprocess.check_output(
        [sys.executable,"-c",code],cwd=root,text=True,
        env=dict(os.environ,PYTHONHASHSEED=str(seed),PYTHONPATH=str(root)))
        for seed in range(8)]
    assert len(set(results)) == 1


def test_sop_varied_term_sizes_preserve_truth_tables():
    from flashsim.ir import BinOp, Const, Id, Signal, UnaryOp
    from flashsim.opt import _merge_sop_terms

    names = 'abcdef'
    signals = {name: Signal(name, 1, 'input') for name in names}
    signals['out'] = Signal('out', 1, 'output')

    def value(expr, bits):
        if isinstance(expr, Const):
            return bool(expr.value)
        if isinstance(expr, Id):
            return bool(bits & (1 << names.index(expr.name)))
        if isinstance(expr, UnaryOp):
            return not value(expr.a, bits)
        if isinstance(expr, BinOp):
            a, b = value(expr.a, bits), value(expr.b, bits)
            return a and b if expr.op == '&' else a or b
        raise AssertionError(expr)

    rng = random.Random(8937)
    for _ in range(20):
        terms = []
        for _ in range(32):
            literals = [Id(name) if rng.randrange(2) else UnaryOp('!', Id(name))
                        for name in rng.sample(names, rng.randint(1, len(names)))]
            term = literals[0]
            for literal in literals[1:]:
                term = BinOp('&', term, literal)
            terms.append(term)
        original = terms[0]
        for term in terms[1:]:
            original = BinOp('|', original, term)
        optimized = _merge_sop_terms({'out': original}, signals)['out']
        for bits in range(1 << len(names)):
            assert value(original, bits) == value(optimized, bits)
