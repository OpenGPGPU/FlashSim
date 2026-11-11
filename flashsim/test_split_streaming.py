"""Large-model splitting preserves source lines without a full line list."""
import random

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
