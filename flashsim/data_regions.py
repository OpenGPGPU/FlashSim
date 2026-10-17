"""Optional compiler-friendly grouping of bulk-initialized DUT storage."""
from __future__ import annotations

import math
import re


def group_primitive_storage(cpp: str, cls: str, target_fields: int = 0) -> str:
    """Keep member names/order while bounding each primitive data subobject.

    Only the initial uninitialized primitive declarations are moved. A bulk
    constructor already zeroes the entire object and restores nonzero values;
    the new bases contain neither initializers nor constructors. Cuts at
    16-byte boundaries leave no base tail padding for a derived member to
    reuse. Compile-time checks reject incompatible primitive/packed layouts.
    """
    if target_fields < 0:
        raise ValueError("data region field count must be >= 0")
    if not target_fields:
        return cpp
    declaration = f"struct {cls} {{\n"
    constructor = f"  {cls}() {{\n    memset(static_cast<void*>(this), 0, sizeof(*this));"
    if declaration not in cpp or constructor not in cpp:
        return cpp

    lines = cpp.splitlines(keepends=True)
    first = lines.index(declaration)
    field = re.compile(r"^  (uint(?:8|16|32|64)_t|unsigned __int128)\s+(.+);\n$")
    declarator = re.compile(r"^(\w+)((?:\[\d+\])*)$")
    sizes = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4,
             "uint64_t": 8, "unsigned __int128": 16}
    records = []
    offset = 0
    for index in range(first + 1, len(lines)):
        match = field.fullmatch(lines[index])
        if not match:
            break
        parts = [declarator.fullmatch(part.strip()) for part in match[2].split(",")]
        if not all(parts):
            break
        row = []
        size = sizes[match[1]]
        for part in parts:
            offset = (offset + size - 1) // size * size
            dimensions = [int(n) for n in re.findall(r"\[(\d+)\]", part[2])]
            row.append((part[1], offset, size))
            offset += size * math.prod(dimensions)
        records.append((index, offset, row))

    groups = []
    start = count = 0
    for index, (_, end, members) in enumerate(records):
        count += len(members)
        if count >= target_fields and end % 16 == 0:
            groups.append((start, index + 1))
            start, count = index + 1, 0
    if start < len(records):
        aligned = [i + 1 for i in range(start, len(records))
                   if records[i][1] % 16 == 0]
        if aligned:
            groups.append((start, aligned[-1]))
            start = aligned[-1]
    if len(groups) < 2:
        return cpp

    existing_names = set(re.findall(
        r"\b" + re.escape(cls) + r"_DataRegion_\d+\b", cpp))
    definitions, bases, checks = [], [], []
    for number, (begin, end) in enumerate(groups):
        name = f"{cls}_DataRegion_{number}"
        if name in existing_names:
            raise ValueError(f"data region name already exists: {name}")
        bases.append(name)
        definitions.append("struct " + name + " {\n" + "".join(
            lines[records[i][0]] for i in range(begin, end)) + "};\n")
        origin = records[begin - 1][1] if begin else 0
        members = [member for record in records[begin:end] for member in record[2]]
        alignment = max(member[2] for member in members)
        checks.append(f'static_assert(sizeof({name}) == {records[end - 1][1] - origin}'
                      f' && alignof({name}) == {alignment}, "data region layout mismatch");\n')
        checks.append(f'static_assert(__builtin_offsetof({cls}, {members[0][0]})'
                      f' == {origin}, "data region position mismatch");\n')

    # These declarations emit no executable code and keep unsupported ABIs
    # from silently changing a packed or unusually aligned field layout.
    guards = ["\n// BEGIN FLASHSIM DATA REGION LAYOUT CHECKS\n",
              '#pragma GCC diagnostic push\n',
              '#pragma GCC diagnostic ignored "-Winvalid-offsetof"\n']
    for kind, size in sizes.items():
        guards.append(f'static_assert(sizeof({kind}) == {size} && alignof({kind}) == {size},'
                      ' "data regions require natural primitive alignment");\n')
    guards.extend(checks)
    guards.extend([f'static_assert(__is_trivially_copyable({cls}),'
                   ' "data regions require trivially copyable DUT storage");\n',
                   '#pragma GCC diagnostic pop\n',
                   '// END FLASHSIM DATA REGION LAYOUT CHECKS\n'])
    end_line = records[start - 1][0] + 1
    return ("".join(lines[:first]) + "\n#define FLASHSIM_DATA_REGIONS 1\n"
            + "".join(definitions) + "struct " + cls + " : " + ", ".join(bases)
            + " {\n" + "".join(lines[end_line:]) + "".join(guards))
