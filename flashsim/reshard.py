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
OPTNONE = (b"#if defined(__clang__)\n"
           b"__attribute__((optnone, noinline))\n"
           b"#else\n"
           b'#error "this resharded model requires Clang"\n'
           b"#endif\n")
OPTNONE_LINES = OPTNONE.splitlines(keepends=True)
MANIFEST = "flashsim-reshard.json"


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
                   optnone: set[str]) -> tuple[list[str], str | None]:
    number = NUMBERED.fullmatch(source.name).group(1)
    outputs: list[str] = []
    current = None
    size = 0
    in_method = False
    found = False
    class_name = None

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
                current.write(line)
                size += len(line)
        if in_method:
            raise ValueError("unterminated method in " + source.name)
    finally:
        if current is not None:
            current.close()
    return outputs, class_name


def create(source: Path, out: Path, *, maximum: int, commit_parts: int,
           optnone_bytes: int = 0) -> None:
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("source and output must be separate directories")
    if maximum <= len(INCLUDE) or not 1 <= commit_parts <= 64 or optnone_bytes < 0:
        raise ValueError("maximum must exceed the include, commit parts must be 1..64, and optnone must be nonnegative")
    inputs = _files(source)
    if any(p.name.startswith("dut_extra_") for p in inputs):
        raise ValueError("input has already been method-resharded")
    if commit_parts > 1 and any(re.fullmatch(r"dut_commit_part\d+\.cpp", p.name) for p in inputs):
        raise ValueError("input commit helpers are already partitioned")
    before = {p.name: digest(p) for p in inputs}
    out.mkdir(parents=True, exist_ok=False)
    try:
        header = source / "dut.h"
        with header.open("rb") as src, (out / "dut.h").open("xb") as dst:
            first = src.read(len(GUARD))
            guard_added = first != GUARD
            if guard_added:
                dst.write(GUARD)
            dst.write(first)
            shutil.copyfileobj(src, dst, 1024 * 1024)
        methods = {}
        unoptimized = {}
        class_name = None
        for path in inputs[1:]:
            if NUMBERED.fullmatch(path.name):
                marked = _large_methods(path, optnone_bytes)
                unoptimized[path.name] = marked
                names, cls = _split_methods(path, out, maximum, set(marked))
                methods[path.name] = names
                if cls is not None:
                    if class_name is not None and cls != class_name:
                        raise ValueError("numbered shards use different DUT classes")
                    class_name = cls
        if not class_name:
            raise ValueError("numbered DUT shards contain no methods")
        commit_outputs = []
        for path in inputs[1:]:
            if NUMBERED.fullmatch(path.name):
                continue
            if path.name == "dut_commit_chunks.cpp" and commit_parts > 1:
                for name, text in emit._partition_commit_chunks(class_name, path.read_text(), commit_parts):
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
            if path.name in expected:
                raise ValueError("reshard helper collides with a method part")
            expected.add(path.name)
    commit_names = record["commit_outputs"] if record["commit_parts"] > 1 else (
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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            header.update(block)
    if header.hexdigest() != record["output_sha256"]["dut.h"]:
        raise ValueError("derived header is not the original with an optional guard")
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
        if joined.hexdigest() != record["input_sha256"][name]:
            raise ValueError("derived methods do not reconstruct " + name)
        if seen_optnone != unoptimized[name]:
            raise ValueError("derived optnone methods differ from original order")
    if record["commit_parts"] > 1:
        parts = emit._partition_commit_chunks(record["class"],
            (source / "dut_commit_chunks.cpp").read_text(), record["commit_parts"])
        if ([name for name, _ in parts] != record["commit_outputs"] or
                any(hashlib.sha256(text.encode()).hexdigest() != record["output_sha256"].get(name)
                    for name, text in parts)):
            raise ValueError("derived commit helpers do not match the emitter")
    for path in inputs[1:]:
        if path.name not in record["method_parts"] and path.name != "dut_commit_chunks.cpp":
            if record["output_sha256"].get(path.name) != record["input_sha256"][path.name]:
                raise ValueError("unchanged helper differs: " + path.name)
    if record["commit_parts"] == 1 and "dut_commit_chunks.cpp" in record["input_sha256"]:
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
    args = parser.parse_args()
    if args.operation == "create":
        create(args.source, args.output, maximum=args.max_mib * 1024 * 1024,
               commit_parts=args.commit_parts, optnone_bytes=args.optnone_mib * 1024 * 1024)
    else:
        verify(args.source, args.output)
    print("reshard", args.operation, "verified", flush=True)


if __name__ == "__main__":
    main()
