"""Partition one generated nonblocking clocked process for CIRCT import.

This is an explicit, narrow preparation step for generated RTL, not a general
Verilog parser. It rejects unsupported statements and detectable side effects;
the caller must also establish the documented edge and driver assumptions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
from io import StringIO
import os
from pathlib import Path
import re
import tempfile


_OPEN = "  always @(posedge clock) begin"
_CLOSE = "  end // always @(posedge)"
_NAME = r"[A-Za-z_]\w*"
_OUTSIDE_NAME = re.compile(rf"\b({_NAME})\b")
_MEMBER = re.compile(rf"\.\s*{_NAME}")


def _pure(expression: str) -> None:
    # Functions, system calls, writes, delays and escaped identifiers can
    # make evaluating a guard or RHS more than once observable.
    if (any(c in expression for c in '"`$\\;#@.')
            or re.search(rf"\b{_NAME}\s*\(", expression)
            or re.search(r"\b(?:new|randomize)\b", expression)
            or re.search(r"\+\+|--|(?<![=!<>])=(?!=)", expression)):
        raise ValueError("unsupported side effect or call in clocked expression")


class _Parser:
    def __init__(self, body: str):
        if any(token in body for token in ('"', '/*', '`')):
            raise ValueError("unsupported comment, string or macro in clocked process")
        self.body = re.sub(r"//[^\n]*", "", body)
        self.pos = 0
        self.counts: Counter[str] = Counter()

    def skip(self) -> None:
        while self.pos < len(self.body) and self.body[self.pos].isspace():
            self.pos += 1

    def word(self, token: str) -> bool:
        self.skip()
        end = self.pos + len(token)
        return (self.body[self.pos:end] == token and
                (end == len(self.body) or not
                 (self.body[end].isalnum() or self.body[end] == "_")))

    def eat(self, token: str) -> None:
        if not self.word(token):
            raise ValueError(f"expected {token} in clocked process")
        self.pos += len(token)

    def statement(self, depth: int = 0):
        if depth > 256:
            raise ValueError("clocked process nesting exceeds 256")
        self.skip()
        if self.word("begin"):
            self.eat("begin")
            children = []
            while not self.word("end"):
                if self.pos >= len(self.body):
                    raise ValueError("unterminated begin block")
                children.append(self.statement(depth + 1))
            self.eat("end")
            return ["block", children, 0]
        if self.word("if"):
            self.eat("if")
            self.skip()
            if self.pos >= len(self.body) or self.body[self.pos] != "(":
                raise ValueError("missing clocked if condition")
            start = self.pos
            nesting = 0
            while self.pos < len(self.body):
                char = self.body[self.pos]
                nesting += (char == "(") - (char == ")")
                self.pos += 1
                if nesting == 0:
                    break
            if nesting:
                raise ValueError("unterminated clocked if condition")
            condition = self.body[start:self.pos]
            _pure(condition)
            yes = self.statement(depth + 1)
            no = None
            if self.word("else"):
                self.eat("else")
                no = self.statement(depth + 1)
            return ["if", condition, yes, no, 0]
        end = self.body.find(";", self.pos)
        if end < 0:
            raise ValueError("unterminated clocked assignment")
        text = self.body[self.pos:end + 1].strip()
        self.pos = end + 1
        match = re.fullmatch(rf"({_NAME})\s*<=\s*(.+);", text, re.S)
        if not match:
            raise ValueError("only plain nonblocking assignments are supported")
        _pure(match[2])
        self.counts[match[1]] += 1
        return ["assign", text, match[1], 0]

    def parse(self):
        children = []
        self.skip()
        while self.pos < len(self.body):
            children.append(self.statement())
            self.skip()
        return ["block", children, 0]


def _outside_writes(prefix: str, suffix: str, targets: set[str],
                    assume_default_initialization: bool,
                    assume_no_instance_drivers: bool) -> None:
    # Firtool's optional RANDOMIZE_REG_INIT assigns the same registers at
    # time zero. The caller must explicitly assert that it and custom init
    # macros are disabled before that guarded region can be ignored.
    marker = "`ifdef RANDOMIZE_REG_INIT"
    if assume_default_initialization and marker in suffix:
        if suffix.count(marker) != 1:
            raise ValueError("unexpected random initialization guards")
        start = suffix.index(marker)
        end = suffix.find("`endif", start)
        if end < 0:
            raise ValueError("unterminated random initialization guard")
        region = suffix[start:end]
        if (region.count("`ifdef") != 1 or "`ifndef" in region or
                "`else" in region or "`elsif" in region):
            raise ValueError("nested random initialization guard")
        suffix = suffix[:start] + suffix[end:]
    outside = re.sub(r"/\*.*?\*/|//[^\n]*", "", prefix + suffix, flags=re.S)
    if ('"' in outside or
            re.search(r"\b(?:alias|bind|force|release|function|task)\b", outside) or
            re.search(r"\$(?:deposit|force|release)\b", outside)):
        raise ValueError("unsupported outside-process construct")
    if not assume_no_instance_drivers:
        for port in re.finditer(rf"\.{_NAME}\s*\(", outside):
            start, pos, depth = port.end(), port.end(), 1
            while pos < len(outside) and depth:
                depth += (outside[pos] == "(") - (outside[pos] == ")")
                pos += 1
            if depth:
                raise ValueError("unterminated instance port expression")
            if any(name[1] in targets for name in _OUTSIDE_NAME.finditer(outside, start, pos - 1)):
                raise ValueError("clocked target reaches an instance port; verify it is not driven")
    for match in _OUTSIDE_NAME.finditer(outside):
        if match[1] not in targets:
            continue
        before = match.start() - 1
        while before >= 0 and outside[before].isspace():
            before -= 1
        if before >= 1 and outside[before - 1:before + 1] in ("++", "--"):
            raise ValueError(f"clocked target {match[1]} may be written elsewhere")
        pos = match.end()
        while pos < len(outside) and outside[pos].isspace():
            pos += 1
        # A second process may write any bit or part select of the same
        # target, including a select containing another select.
        while pos < len(outside) and outside[pos] in "[.":
            if outside[pos] == ".":
                member = _MEMBER.match(outside, pos)
                if not member:
                    raise ValueError("unexpected outside-process member")
                pos = member.end()
            else:
                depth = 1
                pos += 1
                while pos < len(outside) and depth:
                    depth += (outside[pos] == "[") - (outside[pos] == "]")
                    pos += 1
                if depth:
                    raise ValueError("unterminated outside-process select")
            while pos < len(outside) and outside[pos].isspace():
                pos += 1
        remaining = outside[pos:pos + 4]
        if (remaining.startswith(("<=", "++", "--", "+=", "-=", "*=", "/=",
                                  "&=", "|=", "^=", "%=", "<<=", ">>=",
                                  "<<<=", ">>>=", "**="))
                or (remaining.startswith("=") and not remaining.startswith("=="))):
            raise ValueError(f"clocked target {match[1]} may be written elsewhere")
    braces = []
    for pos, char in enumerate(outside):
        if char == "{":
            braces.append(pos)
        elif char == "}" and braces:
            start = braces.pop()
            end = pos + 1
            while end < len(outside) and outside[end].isspace():
                end += 1
            following = outside[end:end + 2]
            if following == "<=" or (following.startswith("=") and following != "=="):
                for name in _OUTSIDE_NAME.finditer(outside, start + 1, pos):
                    if name[1] in targets:
                        raise ValueError(f"clocked target {name[1]} may be written elsewhere")


def partition_clocked_process(source: str, groups: int = 128, *,
                              assume_default_initialization: bool = False,
                              assume_no_instance_drivers: bool = False,
                              assume_stable_edge_inputs: bool = False) -> tuple[str, dict]:
    """Split a single generated clock process while keeping each target ordered.

    The opt-in initialization assumption means RANDOMIZE_REG_INIT and custom
    initializer macros are disabled and the first clock follows reset setup.
    If targets reach instance ports, the caller must verify none are driven.
    All guards and RHS inputs must remain stable throughout the clock event.
    """
    if not 1 <= groups <= 256:
        raise ValueError("groups must be between 1 and 256")
    if not assume_stable_edge_inputs:
        raise ValueError("confirm that clock-edge inputs are stable")
    if source.count(_OPEN) != 1 or source.count(_CLOSE) != 1:
        raise ValueError("expected exactly one generated posedge process")
    prefix, rest = source.split(_OPEN, 1)
    body, suffix = rest.split(_CLOSE, 1)
    parser = _Parser(body)
    root = parser.parse()
    if not parser.counts:
        raise ValueError("clocked process has no assignments")
    _outside_writes(prefix, suffix, set(parser.counts), assume_default_initialization,
                    assume_no_instance_drivers)
    bins = [(0, n) for n in range(groups)]
    heapq.heapify(bins)
    mapping = {}
    for target, weight in parser.counts.most_common():
        total, group = heapq.heappop(bins)
        mapping[target] = group
        heapq.heappush(bins, (total + weight, group))

    # Pass recursion explicitly so the mapping is released by reference counting,
    # including when a caller has disabled cyclic GC for a large compilation.
    def annotate(node, *, _recur):
        if node[0] == "assign":
            mask = 1 << mapping[node[2]]
        elif node[0] == "block":
            mask = 0
            for child in node[1]:
                mask |= _recur(child, _recur=_recur)
        else:
            mask = _recur(node[2], _recur=_recur)
            if node[3] is not None:
                mask |= _recur(node[3], _recur=_recur)
        node[-1] = mask
        return mask

    annotate(root, _recur=annotate)

    def emit(node, mask, out, depth=1, *, _recur):
        if not node[-1] & mask:
            return
        indent = "  " * depth
        if node[0] == "assign":
            out.write(indent + node[1] + "\n")
        elif node[0] == "block":
            for child in node[1]:
                _recur(child, mask, out, depth, _recur=_recur)
        else:
            out.write(indent + "if " + node[1] + " begin\n")
            _recur(node[2], mask, out, depth + 1, _recur=_recur)
            if node[3] is not None and node[3][-1] & mask:
                out.write(indent + "end else begin\n")
                _recur(node[3], mask, out, depth + 1, _recur=_recur)
            out.write(indent + "end\n")

    out = StringIO()
    out.write(prefix)
    for group in range(groups):
        if root[-1] & (1 << group):
            out.write("  always @(posedge clock) begin\n")
            emit(root, 1 << group, out, _recur=emit)
            out.write("  end\n")
    out.write(suffix)
    return out.getvalue(), {"targets": len(parser.counts),
                            "assignments": sum(parser.counts.values()),
                            "groups": root[-1].bit_count()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--groups", type=int, default=128)
    ap.add_argument("--assume-default-initialization", action="store_true",
                    help="assume random/custom initializers are disabled and reset is set before the first clock")
    ap.add_argument("--assume-no-instance-drivers", action="store_true",
                    help="assert that instance output/inout ports do not drive any partitioned target")
    ap.add_argument("--assume-stable-edge-inputs", action="store_true",
                    help="assert that all guard and RHS inputs remain stable throughout the clock event")
    args = ap.parse_args()
    if args.source.resolve() == args.output.resolve():
        ap.error("source and output must be different files")
    result, counts = partition_clocked_process(
        args.source.read_text(), args.groups,
        assume_default_initialization=args.assume_default_initialization,
        assume_no_instance_drivers=args.assume_no_instance_drivers,
        assume_stable_edge_inputs=args.assume_stable_edge_inputs)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix="." + args.output.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(result)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
