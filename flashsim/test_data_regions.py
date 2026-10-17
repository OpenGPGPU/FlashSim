"""Keep large generated storage layout and cross-shard execution unchanged."""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashsim.data_regions import group_primitive_storage
from flashsim.emit import _bulk_state_constructor


def model_source():
    lines = ['#include <cstdint>\n#include <cstring>\nstruct ProbeDut {\n']
    names = []
    for i in range(16):
        declarations = [
            (f'uint8_t byte{i} = {i + 1}', [f'byte{i}']),
            (f'uint16_t pair{i}[2] = {{{i + 3}, {i + 5}}}', [f'pair{i}']),
            (f'uint32_t matrix{i}[2][3] = {{{{{i + 7}}}, {{{i + 9}}}}}', [f'matrix{i}']),
            (f'uint64_t a{i} = {i + 11}, b{i} = {i + 13}', [f'a{i}', f'b{i}']),
            (f'unsigned __int128 wide{i} = ((unsigned __int128)1 << 100) | {i + 1}', [f'wide{i}']),
        ]
        for text, fields in declarations:
            lines.append('  ' + text + ';\n')
            names.extend(fields)
    # The unsupported prefix declaration remains in the derived tail, along
    # with its explicit alignment and nonzero constructor initialization.
    lines += ['  alignas(32) uint64_t guard[3] = {1, 2, 3};\n',
              '  uint8_t tail = 77;\n', '  void _work() {\n']
    for i in range(16):
        lines += [f'    byte{i} ^= {i + 17};\n',
                  f'    pair{i}[1] += byte{i};\n',
                  f'    matrix{i}[1][2] += pair{i}[1];\n',
                  f'    a{i} += matrix{i}[1][2];\n',
                  f'    wide{i} ^= ((unsigned __int128)a{i} << 67);\n']
    lines += ['    tail ^= byte0;\n', '  }\n', '};\n']
    return _bulk_state_constructor(''.join(lines), 'ProbeDut', min_fields=1), names + ['guard', 'tail']


def compiler():
    found = shutil.which('c++')
    if not found:
        pytest.skip('C++ compiler unavailable')
    return found


def compile_and_run(tmp_path, code):
    source = tmp_path / 'probe.cpp'
    source.write_text(code)
    exe = tmp_path / 'probe'
    subprocess.run([compiler(), '-std=gnu++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    source, '-o', exe], check=True, capture_output=True, text=True)
    subprocess.run([exe], check=True)


def test_mixed_offsets_constructor_copy_and_member_execution(tmp_path):
    original, names = model_source()
    grouped = group_primitive_storage(original, 'ProbeDut', 5)
    assert 'ProbeDut_DataRegion_' in grouped
    preamble = '#include <cstdint>\n#include <cstring>\n#include <type_traits>\n'
    checks = ''.join(
        f'if(__builtin_offsetof(A,{name}) != __builtin_offsetof(B,{name})) return 3;\n'
        for name in names)
    code = (preamble + 'namespace Before {\n' + original + '}\nnamespace After {\n'
            + grouped + '}\n#pragma GCC diagnostic push\n'
            + '#pragma GCC diagnostic ignored "' + '-W' + 'invalid-' + 'offsetof' + '"\n'
            + 'int main(){using A=Before::ProbeDut;using B=After::ProbeDut;'
            + 'static_assert(sizeof(A)==sizeof(B));static_assert(alignof(A)==alignof(B));'
            + 'static_assert(std::is_trivially_copyable<A>::value);'
            + 'static_assert(std::is_trivially_copyable<B>::value);A a;B b;'
            + 'if(std::memcmp(&a,&b,sizeof(A)))return 1;'
            + checks + 'for(int i=0;i<20;i++){a._work();b._work();'
            + 'if(std::memcmp(&a,&b,sizeof(A)))return 2;}A ca=a;B cb=b;'
            + 'return std::memcmp(&ca,&cb,sizeof(A))!=0;}\n#pragma GCC diagnostic pop\n')
    compile_and_run(tmp_path, code)


def test_region_layout_guards_reject_packed_storage(tmp_path):
    original, _ = model_source()
    source = tmp_path / 'packed.cpp'
    source.write_text(group_primitive_storage(original, 'ProbeDut', 5))
    result = subprocess.run([compiler(), '-std=gnu++17', '-fpack-struct=1', '-c',
                             source, '-o', tmp_path / 'packed.o'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'data region' in result.stderr


def test_disabled_small_and_nonbulk_headers_stay_identical():
    original, _ = model_source()
    assert group_primitive_storage(original, 'ProbeDut') == original
    assert group_primitive_storage(original, 'ProbeDut', 10000) == original
    implicit = 'struct Small {\n  uint64_t a[32];\n};\n'
    assert group_primitive_storage(implicit, 'Small', 1) == implicit
    with pytest.raises(ValueError, match='>= 0'):
        group_primitive_storage(original, 'ProbeDut', -1)


def test_reserved_name_check_only_rejects_actual_region_collisions():
    original, _ = model_source()
    with pytest.raises(ValueError, match='already exists'):
        group_primitive_storage('struct ProbeDut_DataRegion_0 {};\n' + original,
                                'ProbeDut', 5)
    unrelated = 'struct ProbeDut_DataRegion_999999 {};\n' + original
    assert 'struct ProbeDut : ProbeDut_DataRegion_' in group_primitive_storage(
        unrelated, 'ProbeDut', 5)


@pytest.mark.parametrize('split', [0, 2])
def test_compile_file_groups_after_splitting_without_changing_method_sources(tmp_path, monkeypatch, split):
    import flashsim.compile as compile_module
    original, _ = model_source()
    monkeypatch.setattr(compile_module, 'compile_verilog',
                        lambda *args, **kwargs: (SimpleNamespace(name='Probe'), original))
    verilog = tmp_path / 'Probe.v'
    verilog.write_text('module Probe; endmodule\n')
    flat = tmp_path / 'flat' / 'dut.h'
    grouped = tmp_path / 'grouped' / 'dut.h'
    monkeypatch.delenv('FLASHSIM_DATA_REGION_FIELDS', raising=False)
    compile_module.compile_file(verilog, flat, frontend='native', split=split)
    monkeypatch.setenv('FLASHSIM_DATA_REGION_FIELDS', '5')
    compile_module.compile_file(verilog, grouped, frontend='native', split=split)
    assert 'struct ProbeDut : ProbeDut_DataRegion_' in grouped.read_text()
    assert {p.name: p.read_text() for p in flat.parent.glob('dut_*.cpp')} == {
        p.name: p.read_text() for p in grouped.parent.glob('dut_*.cpp')}
    main = grouped.parent / 'main.cpp'
    main.write_text('#include "dut.h"\nint main(){ProbeDut p;p._work();'
                    'return p.pair0[1] != 21 || p.tail != 93;}\n')
    subprocess.run([compiler(), '-std=gnu++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    main, *grouped.parent.glob('dut_*.cpp'), '-o', grouped.parent / 'run'],
                   check=True, capture_output=True, text=True)
    subprocess.run([grouped.parent / 'run'], check=True)


@pytest.mark.parametrize('value', ['-1', 'not-an-integer'])
def test_invalid_configuration_fails_before_frontend_work(tmp_path, monkeypatch, value):
    from flashsim.compile import compile_file
    monkeypatch.setenv('FLASHSIM_DATA_REGION_FIELDS', value)
    with pytest.raises(ValueError, match='FLASHSIM_DATA_REGION_FIELDS'):
        compile_file(tmp_path / 'does-not-exist.v', tmp_path / 'dut.h', frontend='native')
