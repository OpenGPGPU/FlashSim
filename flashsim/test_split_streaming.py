"""Large-model splitting preserves source lines without a full line list."""
from pathlib import Path
import random
from types import SimpleNamespace
import weakref

import pytest

from flashsim import emit
from flashsim.emit import _iter_lines, split_dut_methods


def test_streamed_lines_match_python_splitlines():
    rng = random.Random(8374)
    characters = 'ab\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029\ud800'
    for text in ['', '\r\n', '\n\n', 'tail', 'a\rb\r\n',
                 'a' * (1024 * 1024 - 1) + '\r\nb',
                 'a' * (1024 * 1024 + 3) + '\u2028b',
                 ('a\v' * (1024 * 1024 // 2)) + '\r\nb',
                 *(''.join(rng.choices(characters, k=rng.randrange(200)))
                   for _ in range(200))]:
        assert list(_iter_lines(text)) == text.splitlines(keepends=True)


def test_split_keeps_non_method_source_and_unusual_line_endings():
    source = 'struct SampleDut {\r\n  // a\u2028  int value;\r\n};'
    header, parts = split_dut_methods(source, n_shards=2)
    assert header == source
    assert parts == [('dut_0.cpp', '#include "dut.h"\n\n'),
                     ('dut_1.cpp', '#include "dut.h"\n\n')]


@pytest.mark.parametrize('min_fields', [3, 10])
@pytest.mark.parametrize('spill', [False, True])
@pytest.mark.parametrize('file_parts', [False, True])
def test_explicit_split_preserves_constructor_and_releases_input(
        monkeypatch, tmp_path, min_fields, spill, file_parts):
    source = [
        '#include <cstdint>\r', 'struct SampleDut {',
        '  uint32_t first = 0u, other = 7;',
        '  alignas(16) uint8_t data[2][2] = {{1, 2}, {3, 4}};',
        '  uint32_t unchanged;', '  void _work() {',
        *['    first += other; // é'] * 30,
        '  }', '', '};', '',
    ]
    cpp = emit._bulk_state_constructor(
        '\n'.join([*source, '']), 'SampleDut', min_fields=min_fields)
    expected = split_dut_methods(cpp, n_shards=4)

    class Line(str):
        pass

    raw_lines = [Line(line) for line in source]
    refs = [weakref.ref(line) for line in raw_lines]
    split = emit._split_dut_lines
    temporary_file = emit.TemporaryFile
    files = []

    def open_spool(**options):
        assert options['dir'] == tmp_path
        stream = temporary_file(**options)
        files.append(stream)
        return stream

    def check_split(cls, lines, n_shards, **kwargs):
        assert not raw_lines
        assert all(ref() is None for ref in refs)
        assert len(files) == int(spill)
        return split(cls, lines, n_shards, **kwargs)

    def emit_lines(mod, output):
        output.extend(raw_lines[:3])
        del raw_lines[:3]
        assert all(ref() is None for ref in refs[:3])
        output.extend(raw_lines)
        raw_lines.clear()

    monkeypatch.setattr(emit, '_CPP_BLOCK_LINES', 3)
    monkeypatch.setattr(emit, '_CPP_SPOOL_BYTES', 16 if spill else 64 * 1024 * 1024)
    monkeypatch.setattr(emit, '_BULK_CONSTRUCTOR_MIN_FIELDS', min_fields)
    monkeypatch.setattr(emit, '_emit_cpp_lines', emit_lines)
    monkeypatch.setattr(emit, '_split_dut_lines', check_split)
    monkeypatch.setattr(emit, 'TemporaryFile', open_spool)
    output_dir = tmp_path / "parts" if file_parts else None
    if output_dir is not None:
        output_dir.mkdir()
    monkeypatch.setattr(emit, '_SPLIT_BUFFER_CHARS', 80)
    header, parts = emit.emit_split_cpp(
        SimpleNamespace(name='Sample'), 4, spool_dir=tmp_path, output_dir=output_dir)
    assert all(isinstance(body, Path) == file_parts for _, body in parts)
    actual = [(name, body.read_bytes().decode() if file_parts else body)
              for name, body in parts]
    assert (header, actual) == expected
    assert all(stream.closed for stream in files)
    assert list(tmp_path.iterdir()) == ([output_dir] if file_parts else [])
