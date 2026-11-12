"""Split already generated DUT methods into smaller, verifiable C++ files.

This transforms only generated DUT source files. It does not modify the input
or claim that the derived files were emitted directly from RTL. The manifest
and ``verify`` command bind every output to the original bytes and transformer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from flashsim import emit

INCLUDE = b'#include "dut.h"\n'
GUARD = b'#pragma once\n'
NUMBERED = re.compile(r"dut_(\d+)\.cpp")
METHOD = re.compile(rb"^void ([A-Za-z_]\w*)::([A-Za-z_]\w*)\(\) \{\n$")
STORE_DECL = re.compile(rb"^  __attribute__\(\(noinline\)\) void _mem_store_part_(\d+)\(\);\n$")
STORE_CALL = re.compile(rb"^    if \((__we\d+(?: \|\| __we\d+)*)\) _mem_store_part_(\d+)\(\);\n$")
STORE_PORT = re.compile(rb"^    if \(__we(\d+)\) \{\n", re.MULTILINE)
STORE_CONTEXT = re.compile(
    rb"\b(?:__func__|__FUNCTION__|__PRETTY_FUNCTION__|__builtin_FUNCTION|"
    rb"__FILE__|__BASE_FILE__|__FILE_NAME__|__LINE__|__COUNTER__|"
    rb"__DATE__|__TIME__|__TIMESTAMP__|__builtin_FILE|__builtin_LINE|"
    rb"__builtin_COLUMN|__builtin_source_location|source_location|_Pragma)\b|"
    rb"(?m:^[ \t]*#)|\bassert\s*\("
)
STORE_HEADER_DIRECTIVE = re.compile(
    rb'^\s*#\s*(?:pragma\s+once|include\s+<(?:cstdint|cstring|stdexcept|string|cstdio)>|'
    rb'define\s+FLASHSIM_DATA_REGIONS\s+1|'
    rb'pragma\s+GCC\s+diagnostic\s+(?:push|pop|ignored\s+"-Winvalid-offsetof"))\s*$'
)
OPTNONE = (b"#if defined(__clang__)\n"
           b"__attribute__((optnone, noinline))\n"
           b"#else\n"
           b'#error "this resharded model requires Clang"\n'
           b"#endif\n")
OPTNONE_LINES = OPTNONE.splitlines(keepends=True)
MANIFEST = "flashsim-reshard.json"


class _LocalStores:
    """Move emitted store helpers into equivalent local noinline closures."""

    def __init__(self, source: Path, class_name: str | None = None):
        helper = source / "dut_mem_stores.cpp"
        if not helper.is_file():
            raise ValueError("local store helpers require outlined memory stores")
        text = helper.read_bytes()
        first = re.match(re.escape(INCLUDE + b"\n") +
                         rb"void ([A-Za-z_]\w*)::_mem_store_part_0\(\) \{\n", text)
        if not first:
            raise ValueError("unexpected outlined store helper class")
        self.class_name = first.group(1).decode("ascii")
        if class_name is not None and self.class_name != class_name:
            raise ValueError("outlined store helpers use a different DUT class")
        class_name = self.class_name
        prefix = rb"^void " + re.escape(class_name.encode()) + rb"::_mem_store_part_(\d+)\(\) \{\n"
        pattern = re.compile(prefix + rb"(.*?)^\}\n\n", re.MULTILINE | re.DOTALL)
        matches = list(pattern.finditer(text))
        self.parts = {int(match.group(1)): match.group(2) for match in matches}
        if (not matches or len(self.parts) != len(matches) or
                sorted(self.parts) != list(range(len(self.parts))) or
                pattern.sub(b"", text) != INCLUDE + b"\n"):
            raise ValueError("unexpected outlined store helper contents")
        if any(STORE_CONTEXT.search(body) or b"_mem_store_part_" in body
               for body in self.parts.values()):
            raise ValueError("outlined store helpers use context-sensitive syntax")
        self.seen: set[int] = set()

    def line(self, line: bytes, method: str | None) -> bytes:
        if b"_mem_store_part_" not in line:
            return line
        match = STORE_CALL.fullmatch(line)
        if not match or method != "_mem_store":
            raise ValueError("unexpected outlined store helper reference")
        index = int(match.group(2))
        if index not in self.parts or index in self.seen:
            raise ValueError("missing or repeated outlined store helper call")
        body = self.parts[index]
        guards = [int(item) for item in re.findall(rb"__we(\d+)", match.group(1))]
        ports = [int(item) for item in STORE_PORT.findall(body)]
        if not ports or guards != ports:
            raise ValueError("outlined store guards do not match helper ports")
        self.seen.add(index)
        return (b"    if (" + match.group(1) + b") [this]() __attribute__((noinline)) {\n" +
                body + b"    }();\n")

    def finish(self) -> None:
        if sorted(self.seen) != sorted(self.parts):
            raise ValueError("not all outlined store helpers were called")

    def header_lines(self, stream):
        removed = []
        for line in stream:
            if line.lstrip().startswith(b"#") and not STORE_HEADER_DIRECTIVE.fullmatch(line):
                raise ValueError("unsupported preprocessor directive in localized store header")
            match = STORE_DECL.fullmatch(line)
            if match:
                removed.append(int(match.group(1)))
            elif b"_mem_store_part_" in line:
                raise ValueError("unexpected outlined store helper declaration")
            else:
                yield line
        if removed != sorted(self.parts):
            raise ValueError("outlined store declarations do not match helpers")


def _no_other_store_references(inputs: list[Path]) -> None:
    """Reject references outside the numbered methods or helper definitions."""
    for path in inputs:
        if path.name in ("dut.h", "dut_mem_stores.cpp") or NUMBERED.fullmatch(path.name):
            continue
        tail = b""
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                data = tail + block
                if b"_mem_store_part_" in data:
                    raise ValueError("unexpected store helper reference in " + path.name)
                tail = data[-32:]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _files(source: Path) -> list[Path]:
    header = source / "dut.h"
    shards = sorted(source.glob("dut_*.cpp"))
    if not header.is_file() or not any(NUMBERED.fullmatch(p.name) for p in shards):
        raise ValueError("source needs dut.h and numbered DUT shards")
    if any(p.is_symlink() or not p.is_file() for p in [header, *shards]):
        raise ValueError("DUT inputs must be regular files")
    return [header, *shards]


def _large_methods(source: Path, threshold: int) -> list[str]:
    """Find whole methods larger than threshold without buffering their bodies."""
    if not threshold:
        return []
    found = []
    name = None
    size = 0
    with source.open("rb") as stream:
        for line in stream:
            if name is None:
                match = METHOD.fullmatch(line)
                if match:
                    name = match.group(2).decode("ascii")
                    size = len(line)
            else:
                size += len(line)
                if line == b"}\n":
                    if size > threshold:
                        found.append(name)
                    name = None
    if name is not None or len(found) != len(set(found)):
        raise ValueError("unterminated or duplicate large method in " + source.name)
    return found


def _split_methods(source: Path, out: Path, maximum: int,
                   optnone: set[str], stores: _LocalStores | None = None) -> tuple[list[str], str | None]:
    number = NUMBERED.fullmatch(source.name).group(1)
    outputs: list[str] = []
    current = None
    size = 0
    in_method = False
    found = False
    class_name = None
    method_name = None

    def start() -> None:
        nonlocal current, size, found
        name = source.name if not outputs else f"dut_extra_{number}_{len(outputs):04d}.cpp"
        if (out / name).exists():
            raise ValueError("derived shard name already exists: " + name)
        outputs.append(name)
        current = (out / name).open("xb")
        current.write(INCLUDE)
        size = len(INCLUDE)
        found = False

    try:
        start()
        with source.open("rb") as stream:
            if stream.readline() != INCLUDE:
                raise ValueError("unexpected DUT shard include: " + source.name)
            for line in stream:
                if not in_method:
                    match = METHOD.fullmatch(line)
                    if match:
                        name = match.group(1).decode("ascii")
                        if class_name is not None and name != class_name:
                            raise ValueError("multiple DUT classes in " + source.name)
                        class_name = name
                        method_name = match.group(2).decode("ascii")
                        if stores is not None and name != stores.class_name:
                            raise ValueError("numbered shard and store helper classes differ")
                        if size >= maximum and found:
                            current.close()
                            start()
                        if match.group(2).decode("ascii") in optnone:
                            current.write(OPTNONE)
                            size += len(OPTNONE)
                        in_method = True
                        found = True
                    elif line.strip():
                        raise ValueError("unexpected top-level content in " + source.name)
                elif line == b"}\n":
                    in_method = False
                elif line[:1] not in (b" ", b"\t", b"\n", b"\r"):
                    raise ValueError("unexpected unindented method content in " + source.name)
                current.write(stores.line(line, method_name) if stores is not None else line)
                if not in_method:
                    method_name = None
                # Partition at the same original method boundaries even when a
                # closure changes the physical size of the emitted method.
                size += len(line)
        if in_method:
            raise ValueError("unterminated method in " + source.name)
    finally:
        if current is not None:
            current.close()
    return outputs, class_name


def _localized_digest(path: Path, stores: _LocalStores) -> str:
    """Hash the exact method stream expected after localizing store calls."""
    value = hashlib.sha256()
    method = None
    with path.open("rb") as stream:
        for line in stream:
            if method is None:
                match = METHOD.fullmatch(line)
                if match:
                    if match.group(1).decode("ascii") != stores.class_name:
                        raise ValueError("numbered shard and store helper classes differ")
                    method = match.group(2).decode("ascii")
            value.update(stores.line(line, method))
            if line == b"}\n":
                method = None
    return value.hexdigest()


def _commit_outputs(class_name: str, body: str, count: int,
                    header: str | None = None) -> list[tuple[str, str]]:
    """Derive commit partitions, optionally specializing immutable tables."""
    helper = None
    if header is not None:
        # A macro can manufacture a table or helper name without spelling it
        # literally. Restrict specialization to the emitter's known header
        # directives and its single commit-source include.
        for directive in re.finditer(r"(?m)^[ \t]*#[^\n]*", header):
            if not STORE_HEADER_DIRECTIVE.fullmatch(directive[0].encode()):
                raise ValueError("unsupported directive in commit-block header")
        for directive in re.finditer(r"(?m)^[ \t]*#[^\n]*", body):
            if directive[0].strip() != '#include "dut.h"':
                raise ValueError("unsupported directive in commit-block source")
        body, helper = emit._outline_static_commit_blocks(
            header, class_name, body, enabled=True)
    parts = (emit._partition_commit_chunks(class_name, body, count) if count > 1
             else [("dut_commit_chunks.cpp", body)])
    if helper is not None:
        parts.append(("dut_commit_blocks.cpp", helper))
    return parts


def create(source: Path, out: Path, *, maximum: int, commit_parts: int,
           optnone_bytes: int = 0, local_store_helpers: bool = False,
           commit_blocks: bool = False) -> None:
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("source and output must be separate directories")
    if maximum <= len(INCLUDE) or not 1 <= commit_parts <= 64 or optnone_bytes < 0:
        raise ValueError("maximum must exceed the include, commit parts must be 1..64, and optnone must be nonnegative")
    inputs = _files(source)
    if any(p.name.startswith("dut_extra_") for p in inputs):
        raise ValueError("input has already been method-resharded")
    if (commit_parts > 1 or commit_blocks) and any(re.fullmatch(r"dut_commit_part\d+\.cpp", p.name) for p in inputs):
        raise ValueError("input commit helpers are already partitioned")
    if not isinstance(local_store_helpers, bool) or not isinstance(commit_blocks, bool):
        raise ValueError("local_store_helpers and commit_blocks must be booleans")
    if commit_blocks and (not (source / "dut_commit_chunks.cpp").is_file() or
                          (source / "dut_commit_blocks.cpp").exists()):
        raise ValueError("commit blocks need an unpartitioned generic commit dispatcher")
    stores = _LocalStores(source) if local_store_helpers else None
    if stores is not None:
        _no_other_store_references(inputs)
    before = {p.name: digest(p) for p in inputs}
    out.mkdir(parents=True, exist_ok=False)
    try:
        header = source / "dut.h"
        with header.open("rb") as src, (out / "dut.h").open("xb") as dst:
            first = src.read(len(GUARD))
            guard_added = first != GUARD
            if guard_added:
                dst.write(GUARD)
            src.seek(0)
            if stores is None:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            else:
                for line in stores.header_lines(src):
                    dst.write(line)
        methods = {}
        unoptimized = {}
        class_name = None
        for path in inputs[1:]:
            if NUMBERED.fullmatch(path.name):
                marked = _large_methods(path, optnone_bytes)
                unoptimized[path.name] = marked
                names, cls = _split_methods(path, out, maximum, set(marked), stores)
                methods[path.name] = names
                if cls is not None:
                    if class_name is not None and cls != class_name:
                        raise ValueError("numbered shards use different DUT classes")
                    class_name = cls
        if not class_name:
            raise ValueError("numbered DUT shards contain no methods")
        if stores is not None:
            stores.finish()
        commit_outputs = []
        for path in inputs[1:]:
            if NUMBERED.fullmatch(path.name):
                continue
            if stores is not None and path.name == "dut_mem_stores.cpp":
                continue
            if path.name == "dut_commit_chunks.cpp" and (commit_parts > 1 or commit_blocks):
                header = (out / "dut.h").read_text() if commit_blocks else None
                for name, text in _commit_outputs(class_name, path.read_text(), commit_parts, header):
                    if (out / name).exists():
                        raise ValueError("derived commit shard already exists: " + name)
                    (out / name).write_text(text)
                    commit_outputs.append(name)
            else:
                shutil.copyfile(path, out / path.name)
        if commit_parts > 1 and not commit_outputs:
            raise ValueError("input has no chunked commit dispatcher")
        if {p.name: digest(p) for p in inputs} != before:
            raise RuntimeError("DUT input changed during resharding")
        outputs = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
        record = {"version": 1, "tool_sha256": digest(Path(__file__)),
                  "emitter_sha256": digest(Path(emit.__file__)), "input_sha256": before,
                  "output_sha256": outputs, "method_parts": methods, "class": class_name,
                  "guard_added": guard_added, "maximum_bytes": maximum,
                  "optnone_bytes": optnone_bytes, "optnone_methods": unoptimized,
                  "commit_parts": commit_parts, "commit_outputs": commit_outputs}
        if stores is not None:
            record.update(local_store_helpers=True, local_store_count=len(stores.parts))
        if commit_blocks:
            record["commit_blocks"] = True
        (out / MANIFEST).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        verify(source, out)
    except BaseException:
        shutil.rmtree(out)
        raise


def verify(source: Path, out: Path) -> None:
    source, out = source.resolve(), out.resolve()
    record = json.loads((out / MANIFEST).read_text())
    if (record.get("version") != 1 or record.get("tool_sha256") != digest(Path(__file__)) or
            record.get("emitter_sha256") != digest(Path(emit.__file__))):
        raise ValueError("reshard transformer differs from its manifest")
    inputs = _files(source)
    if {p.name: digest(p) for p in inputs} != record["input_sha256"]:
        raise ValueError("reshard input differs from its manifest")
    local = record.get("local_store_helpers", False)
    if not isinstance(local, bool):
        raise ValueError("invalid local store helper setting")
    blocks = record.get("commit_blocks", False)
    if not isinstance(blocks, bool):
        raise ValueError("invalid commit block setting")
    if blocks and (not (source / "dut_commit_chunks.cpp").is_file() or
                   (source / "dut_commit_blocks.cpp").exists() or
                   any(re.fullmatch(r"dut_commit_part\d+\.cpp", p.name) for p in inputs)):
        raise ValueError("invalid input for commit blocks")
    stores = _LocalStores(source, record["class"]) if local else None
    if stores is not None:
        if record.get("local_store_count") != len(stores.parts):
            raise ValueError("local store helper count differs")
        _no_other_store_references(inputs)
    numbered = {p.name for p in inputs if NUMBERED.fullmatch(p.name)}
    if set(record["method_parts"]) != numbered or not 1 <= record["commit_parts"] <= 64:
        raise ValueError("reshard manifest does not cover the original methods")
    threshold = record.get("optnone_bytes", 0)
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("invalid optnone threshold")
    unoptimized = {p.name: _large_methods(p, threshold) for p in inputs if NUMBERED.fullmatch(p.name)}
    if record.get("optnone_methods", {name: [] for name in numbered}) != unoptimized:
        raise ValueError("optnone methods do not match the original source")
    expected = {"dut.h"}
    for name, parts in record["method_parts"].items():
        number = NUMBERED.fullmatch(name).group(1)
        correct = [name] + [f"dut_extra_{number}_{i:04d}.cpp" for i in range(1, len(parts))]
        if not parts or parts != correct or expected.intersection(parts):
            raise ValueError("reshard manifest contains invalid method paths")
        expected.update(parts)
    for path in inputs[1:]:
        if path.name not in numbered and path.name != "dut_commit_chunks.cpp":
            if stores is not None and path.name == "dut_mem_stores.cpp":
                continue
            if path.name in expected:
                raise ValueError("reshard helper collides with a method part")
            expected.add(path.name)
    commit_names = record["commit_outputs"] if record["commit_parts"] > 1 or blocks else (
        ["dut_commit_chunks.cpp"] if "dut_commit_chunks.cpp" in record["input_sha256"] else [])
    if len(set(commit_names)) != len(commit_names) or expected.intersection(commit_names):
        raise ValueError("reshard commit outputs collide")
    expected.update(commit_names)
    if set(record["output_sha256"]) != expected:
        raise ValueError("reshard manifest does not describe exactly the derived files")
    outputs = [p for p in out.iterdir() if p.name != MANIFEST]
    if any(p.is_symlink() or not p.is_file() for p in outputs):
        raise ValueError("reshard output contains a non-regular file")
    if {p.name: digest(p) for p in outputs} != record["output_sha256"]:
        raise ValueError("reshard output differs from its manifest")
    header = hashlib.sha256()
    if record["guard_added"]:
        header.update(GUARD)
    with (source / "dut.h").open("rb") as stream:
        if stores is None:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                header.update(block)
        else:
            for line in stores.header_lines(stream):
                header.update(line)
    if header.hexdigest() != record["output_sha256"]["dut.h"]:
        raise ValueError("derived header does not match the verified transformation")
    localized = {}
    if stores is not None:
        localized = {p.name: _localized_digest(p, stores) for p in inputs
                     if NUMBERED.fullmatch(p.name)}
        stores.finish()
    for name, parts in record["method_parts"].items():
        joined = hashlib.sha256()
        seen_optnone = []
        for i, part in enumerate(parts):
            with (out / part).open("rb") as stream:
                if stream.readline() != INCLUDE:
                    raise ValueError("derived shard is missing its include")
                if not i:
                    joined.update(INCLUDE)
                in_method = False
                while line := stream.readline():
                    marked = False
                    if not in_method and line == OPTNONE_LINES[0]:
                        for expected_line in OPTNONE_LINES[1:]:
                            if stream.readline() != expected_line:
                                raise ValueError("invalid optnone annotation")
                        line = stream.readline()
                        marked = True
                    if not in_method:
                        match = METHOD.fullmatch(line)
                        if match:
                            if match.group(1).decode("ascii") != record["class"]:
                                raise ValueError("derived shard contains another DUT class")
                            method = match.group(2).decode("ascii")
                            if marked:
                                seen_optnone.append(method)
                            if marked != (method in unoptimized[name]):
                                raise ValueError("derived method has incorrect optnone annotation")
                            in_method = True
                        elif marked:
                            raise ValueError("optnone annotation has no method")
                        elif line.strip():
                            raise ValueError("derived shard contains top-level content")
                    elif line == b"}\n":
                        in_method = False
                    elif line[:1] not in (b" ", b"\t", b"\n", b"\r"):
                        raise ValueError("derived shard has unindented method content")
                    joined.update(line)
                if in_method:
                    raise ValueError("derived shard ends within a method")
        expected_digest = localized[name] if stores is not None else record["input_sha256"][name]
        if joined.hexdigest() != expected_digest:
            raise ValueError("derived methods do not reconstruct " + name)
        if seen_optnone != unoptimized[name]:
            raise ValueError("derived optnone methods differ from original order")
    if record["commit_parts"] > 1 or blocks:
        header = (out / "dut.h").read_text() if blocks else None
        parts = _commit_outputs(record["class"],
            (source / "dut_commit_chunks.cpp").read_text(), record["commit_parts"], header)
        if ([name for name, _ in parts] != record["commit_outputs"] or
                any(hashlib.sha256(text.encode()).hexdigest() != record["output_sha256"].get(name)
                    for name, text in parts)):
            raise ValueError("derived commit helpers do not match the emitter")
    for path in inputs[1:]:
        if path.name not in record["method_parts"] and path.name != "dut_commit_chunks.cpp":
            if stores is not None and path.name == "dut_mem_stores.cpp":
                continue
            if record["output_sha256"].get(path.name) != record["input_sha256"][path.name]:
                raise ValueError("unchanged helper differs: " + path.name)
    if not blocks and record["commit_parts"] == 1 and "dut_commit_chunks.cpp" in record["input_sha256"]:
        if record["output_sha256"].get("dut_commit_chunks.cpp") != record["input_sha256"]["dut_commit_chunks.cpp"]:
            raise ValueError("unchanged commit helper differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "verify"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=16)
    parser.add_argument("--commit-parts", type=int, default=1)
    parser.add_argument("--optnone-mib", type=int, default=0,
                        help="compile methods above this size without Clang optimization; 0 disables")
    parser.add_argument("--local-store-helpers", action="store_true",
                        help="localize outlined store helpers while verifying their exact bodies and order")
    parser.add_argument("--commit-blocks", action="store_true",
                        help="specialize eligible immutable commit tables in an already generated model")
    args = parser.parse_args()
    if args.operation == "create":
        create(args.source, args.output, maximum=args.max_mib * 1024 * 1024,
               commit_parts=args.commit_parts, optnone_bytes=args.optnone_mib * 1024 * 1024,
               local_store_helpers=args.local_store_helpers, commit_blocks=args.commit_blocks)
    else:
        verify(args.source, args.output)
    print("reshard", args.operation, "verified", flush=True)


if __name__ == "__main__":
    main()
