"""Forced evaluation must wake both schedulers, including their tail words."""
import shutil
import subprocess

import pytest

from flashsim.arti_model import SYSTEM_SOURCE as SOURCE
from flashsim.emit import _force_wake_method


@pytest.mark.parametrize("nparts,nholds,npasses", [
    (0, 0, 0), (3, 1, 1), (5, 63, 65), (2, 64, 64),
    (5, 65, 63), (1, 4096, 4097), (7, 4097, 4161),
])
def test_force_wake_rearms_valid_bits_and_can_drain(tmp_path, nparts, nholds, npasses):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    fields = []
    checks = []
    if nparts:
        fields += [f"uint32_t _pg[{nparts}]={{}};", "uint32_t _pg_gen=0;"]
        checks += [f"for(unsigned i=0;i<{nparts};i++)if(d._pg[i]!=round)return 1;",
                   "if(d._pg_gen!=round)return 2;"]
    for count, pending, summary, counter in (
        (nholds, "_h_busy", "_h_any", "_h_words"),
        (npasses, "_p_need", "_p_any", "_p_words"),
    ):
        if not count:
            continue
        nw = (count + 63) // 64
        ns = (nw + 63) // 64
        fields += [f"uint64_t {pending}[{nw}]={{}},{summary}[{ns}]={{}};",
                   f"unsigned {counter}=0;"]
        checks += [f"if(d.{counter}!={ns})return 3;",
                   f"for(unsigned bit=0;bit<{nw*64};bit++)"
                   f"if(bool(d.{pending}[bit/64]&(1ull<<(bit%64)))!=(bit<{count}))return 4;",
                   f"for(unsigned bit=0;bit<{ns*64};bit++)"
                   f"if(bool(d.{summary}[bit/64]&(1ull<<(bit%64)))!=(bit<{nw}))return 5;",
                   f"for(unsigned w=0;w<{nw};w++){{"
                   f"d.{pending}[w]=0;d.{summary}[w/64]&=~(1ull<<(w%64));"
                   f"if(!d.{summary}[w/64])--d.{counter};}}",
                   f"if(d.{counter})return 6;"]
    text = "#include <cstdint>\nstruct Dut{\n" + "\n".join(fields)
    text += "\n" + "\n".join(_force_wake_method(nparts, nholds, npasses, bool(nparts)))
    text += "\n};\nint main(){Dut d;for(unsigned round=1;round<=3;round++){"
    text += "d._force_wake_all();\n" + "\n".join(checks) + "\n}}\n"
    source = tmp_path / "check.cpp"
    source.write_text(text)
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_wrapper_force_wake_accepts_new_and_older_headers(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    begin = SOURCE.index("template <typename Model>")
    end = SOURCE.index("static void force_hold_boost(void)", begin)
    text = "#include <cstdint>\n" + SOURCE[begin:end] + r'''
struct Modern {
    unsigned calls=0;
    void _force_wake_all(){++calls;}
};
struct Legacy {
    uint64_t _h_busy[2]={},_h_any[1]={};
    uint32_t _pg[3]={};
};
int main(){
    Modern modern;Legacy legacy;
    force_model_wake(&modern,0);force_model_wake(&legacy,0);
    if(modern.calls!=1)return 1;
    if(legacy._h_busy[0]!=~0ull||legacy._h_busy[1]!=~0ull||legacy._h_any[0]!=~0ull)return 2;
    for(auto page:legacy._pg)if(page!=1)return 3;
}
'''
    source = tmp_path / "wrapper.cpp"
    source.write_text(text)
    binary = tmp_path / "wrapper"
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
