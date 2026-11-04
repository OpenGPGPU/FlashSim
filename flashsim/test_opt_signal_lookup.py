"""Generated temporary widths use a live overlay of existing signals."""

from collections.abc import Mapping

import pytest

from flashsim.ir import BinOp, Id, Signal
from flashsim.opt import _fill_missing_wire_signals


class NoBulkCopy(Mapping):
    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        raise AssertionError("signal table must not be copied")


def test_generated_widths_see_prior_temporaries_and_result_overrides():
    original = NoBulkCopy({
        "source": Signal("source", 16, "input"),
        "shadow": Signal("shadow", 32, "wire"),
    })
    with pytest.raises(AssertionError, match="must not be copied"):
        {**original}
    result = {"shadow": Signal("shadow", 8, "wire")}
    assigns = {
        "first": Id("shadow"),
        "second": BinOp("+", Id("first"), Id("source")),
        "third": Id("second"),
    }
    _fill_missing_wire_signals(original, result, assigns)
    assert result["first"].width == 8
    assert result["second"].width == result["third"].width == 16
