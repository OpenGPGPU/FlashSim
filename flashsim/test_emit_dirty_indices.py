"""Large DUTs must preserve dirty state indices across the 16-bit boundary."""
import shutil
import subprocess

import pytest

from flashsim.emit import _dirty_tracking_declarations


@pytest.mark.parametrize("count", [65536, 65537, 66748])
def test_dirty_indices_do_not_alias_and_can_be_rearmed(tmp_path, count):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    # Exercise the exact declarations used by emit_cpp, without compiling
    # tens of thousands of unrelated register-assignment bodies.
    declarations = "\n".join(_dirty_tracking_declarations(count))
    main = tmp_path / "main.cpp"
    main.write_text("#include <cstdint>\nstruct Dut {\n" + declarations +
        "\n};\n" + r"""
static Dut d;
int main() {
  const unsigned count = sizeof(d._w);
  for (unsigned round=0;round<3;round++) {
    // Includes indices 0, 65535, 65536, and all later GPU states. Inserting
    // twice must preserve one entry per state, even after the next tick.
    for(unsigned i=0;i<count;i++) { d._note(i); d._note(i); }
    if(d._nw!=count)return 1;
    for(unsigned i=0;i<count;i++)
      if(d._wl[i]!=i || !d._w[i])return 2;
    for(unsigned i=0;i<d._nw;i++)d._w[d._wl[i]]=0;
    d._nw=0;
    // Change only the high entry next: it must not alias its low 16 bits.
    d._note(count-1);
    if(d._nw!=1 || d._wl[0]!=count-1 || !d._w[count-1])return 3;
    if(count>65536 && d._w[(count-1)&65535])return 4;
    d._w[d._wl[0]]=0;d._nw=0;
  }
}
""")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=address,undefined", str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
