from __future__ import annotations

from collections import defaultdict

from flashsim.ir import (
    ArrayGet,
    ArrayInject,
    ArrayZeros,
    BinOp,
    CMP_OPS,
    Concat,
    Const,
    Expr,
    Extract,
    Id,
    If,
    MemRead,
    Module,
    NbAssign,
    Signal,
    Stmt,
    Ternary,
    UnaryOp,
    expr_ids,
    stmt_writes,
)


def assign_map(mod: Module) -> dict[str, Expr]:
    return {a.lhs: a.rhs for a in mod.assigns}


def cone_leaves(root: str, assigns: dict[str, Expr], stop: set[str]) -> set[str]:
    found: set[str] = set()
    seen: set[str] = set()

    def walk(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        if name in stop or name not in assigns:
            found.add(name)
            return
        for dep in expr_ids(assigns[name]):
            walk(dep)

    walk(root)
    return found


def collect_regs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "reg"]


def collect_inputs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "input"]


def collect_outputs(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "output"]


def collect_mems(mod: Module) -> list[str]:
    return [n for n, s in mod.signals.items() if s.kind == "mem"]


def _is_array(sigs: dict[str, Signal], name: str) -> bool:
    return name in sigs and sigs[name].depth > 0


def seq_skip_roots(body: list[Stmt], gated: bool = False) -> set[str]:
    """Wires that need a skip cache in sequential code.

    Conditions are always cached. Assignment payloads are cached only when the
    write is not behind a hold-style `if (en)` (empty else). Gated payloads are
    emitted as locals in the taken branch so idle cycles skip them entirely.
    """
    names: set[str] = set()
    for stmt in body:
        if isinstance(stmt, If):
            names |= expr_ids(stmt.cond)
            if stmt.else_body:
                names |= seq_skip_roots(stmt.then_body, gated)
                names |= seq_skip_roots(stmt.else_body, gated)
            else:
                names |= seq_skip_roots(stmt.then_body, True)
        elif isinstance(stmt, NbAssign):
            if not gated:
                names |= expr_ids(stmt.rhs)
        else:
            raise TypeError(stmt)
    return names


def always_cond_reads(body: list[Stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in body:
        if isinstance(stmt, If):
            names |= expr_ids(stmt.cond)
            names |= always_cond_reads(stmt.then_body)
            names |= always_cond_reads(stmt.else_body)
    return names


def always_writes(body: list[Stmt]) -> list[str]:
    names: set[str] = set()
    for stmt in body:
        names |= stmt_writes(stmt)
    return sorted(names)


def cached_wires(mod: Module, assigns: dict[str, Expr]) -> set[str]:
    """Wires that need a skip cache: sequential roots, mem enables, and outputs."""
    names = seq_skip_roots(mod.always.body) | set(collect_outputs(mod))
    for wr in mod.mem_writes:
        names |= expr_ids(wr.enable)
    return {n for n in names if n in assigns}


def internal_cone(
    root: str,
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> list[str]:
    """Topological internals used only to compute `root`."""
    needed: list[str] = []
    seen: set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name in stop or name not in assigns:
            return
        if name != root and name in cached:
            return
        seen.add(name)
        for dep in expr_ids(assigns[name]):
            walk(dep)
        if name != root:
            needed.append(name)

    walk(root)
    return needed


def cached_deps(root: str, assigns: dict[str, Expr], cached: set[str], stop: set[str]) -> list[str]:
    deps: list[str] = []
    seen: set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name in stop or name not in assigns:
            return
        seen.add(name)
        if name != root and name in cached:
            deps.append(name)
            return
        for dep in expr_ids(assigns[name]):
            walk(dep)

    walk(root)
    return deps


def skip_partition_key(name: str) -> str:
    """Coarsen skip activity the way ESSENT coarsens CCSS partitions.

    Per-wire invalidation stores explode on GPU-sized nets. Wires that share a
    Chisel instance prefix share one generation counter, so a leaf change bumps
    O(partitions) instead of O(wires). Strip SSA temps (`t12`) first so a
    decoder helper does not get its own partition.
    """
    toks = name.split("_")
    while toks and len(toks[-1]) > 1 and toks[-1][0] == "t" and toks[-1][1:].isdigit():
        toks.pop()
    if not toks:
        return name
    if toks[0] == "computeUnits" and len(toks) >= 5:
        return "_".join(toks[:5])
    if toks[0] == "l2" and len(toks) >= 4:
        return "_".join(toks[:4])
    if len(toks) >= 3:
        return "_".join(toks[:3])
    return "_".join(toks)


def _partition_maps(
    cached: set[str], wire_deps: dict[str, set[str]]
) -> tuple[dict[str, int], dict[str, list[int]], int]:
    if len(cached) <= 128:
        uniq_sets = sorted(
            {frozenset(deps) for deps in wire_deps.values()},
            key=lambda s: tuple(sorted(s)),
        )
        part_id = {s: i for i, s in enumerate(uniq_sets)}
        wire_part = {w: part_id[frozenset(wire_deps[w])] for w in cached}
    else:
        keys = sorted({skip_partition_key(w) for w in cached})
        part_of = {k: i for i, k in enumerate(keys)}
        wire_part = {w: part_of[skip_partition_key(w)] for w in cached}
    leaf_parts: dict[str, set[int]] = defaultdict(set)
    for w, deps in wire_deps.items():
        p = wire_part[w]
        for leaf in deps:
            leaf_parts[leaf].add(p)
    return wire_part, {k: sorted(v) for k, v in leaf_parts.items()}, max(wire_part.values(), default=-1) + 1


_BUMP_INLINE = 6


def _invalidation_tables(leaf_parts: dict[str, list[int]]) -> dict[tuple[int, ...], int]:
    tables: dict[tuple[int, ...], int] = {}
    for parts in leaf_parts.values():
        key = tuple(parts)
        if len(key) > _BUMP_INLINE and key not in tables:
            tables[key] = len(tables)
    return tables


def _cond_stop_leaves(
    cond: Expr,
    assigns: dict[str, Expr],
    stop: set[str],
    wire_deps: dict[str, set[str]],
) -> set[str]:
    leaves: set[str] = set()
    for name in expr_ids(cond):
        if name in stop or name not in assigns:
            leaves.add(name)
        elif name in wire_deps:
            leaves |= wire_deps[name]
        else:
            leaves |= cone_leaves(name, assigns, stop)
    return leaves


def _hold_wake_tables(leaf_holds: dict[str, list[int]]) -> dict[tuple[int, ...], int]:
    tables: dict[tuple[int, ...], int] = {}
    for ks in leaf_holds.values():
        key = tuple(ks)
        if len(key) > _BUMP_INLINE and key not in tables:
            tables[key] = len(tables)
    return tables


def _bump_parts(
    leaf: str,
    leaf_parts: dict[str, list[int]],
    tables: dict[tuple[int, ...], int],
    lines: list[str],
    indent: int,
) -> None:
    sp = " " * indent
    parts = leaf_parts.get(leaf, ())
    if parts:
        if len(parts) <= _BUMP_INLINE:
            for p in parts:
                lines.append(f"{sp}_pg[{p}]++;")
        else:
            i = tables[tuple(parts)]
            lines.append(f"{sp}fs_bump(_pg, _inv{i}, {len(parts)}u);")
    ks = _LEAF_HOLDS.get(leaf, ())
    if not ks:
        return
    if len(ks) <= _BUMP_INLINE:
        for k in ks:
            word, bit = k >> 6, k & 63
            lines.append(f"{sp}_h_need[{word}] |= 1ull << {bit};")
        return
    i = _HOLD_WAKE_TABLES[tuple(ks)]
    lines.append(f"{sp}fs_wake(_h_need, _hw{i}, {len(ks)}u);")


_WIRE_PART: dict[str, int] = {}
_WRITE_INDEX: dict[str, int] = {}
_WRITE_WIDE: set[str] = set()
_LEAF_HOLDS: dict[str, list[int]] = {}
_HOLD_WAKE_TABLES: dict[tuple[int, ...], int] = {}
_ARRAY_INDEX: dict[str, int] = {}
_HOLD_TAKEN = ""
_LARGE = False
_LARGE_WRITES = 256


def _eval_invoke(name: str, indent: int) -> str:
    sp = " " * indent
    p = _WIRE_PART.get(name)
    if p is None:
        return f"{sp}eval_{name}();"
    return f"{sp}if ({name}__ok != _pg[{p}]) eval_{name}();"


def _mark_write(name: str, lines: list[str], indent: int) -> None:
    sp = " " * indent
    extra = f" {_HOLD_TAKEN}" if _HOLD_TAKEN else ""
    if _LARGE:
        idx = _WRITE_INDEX.get(name)
        if idx is None:
            return
        if name in _WRITE_WIDE:
            lines.append(
                f"{sp}if (memcmp({name}__n, {name}, sizeof({name}))) {{ _note({idx});{extra} }}"
            )
        else:
            lines.append(f"{sp}if ({name}__n != {name}) {{ _note({idx});{extra} }}")
        return
    lines.append(f"{sp}{name}__w = 1;")


def _stmt_names(stmt: Stmt) -> set[str]:
    acc: list[str] = []

    def walk(ss: list[Stmt]) -> None:
        for s in ss:
            if isinstance(s, NbAssign):
                acc.append(s.lhs)
            elif isinstance(s, If):
                acc.extend(expr_ids(s.cond))
                walk(s.then_body)
                walk(s.else_body)

    if isinstance(stmt, NbAssign):
        acc.append(stmt.lhs)
    elif isinstance(stmt, If):
        acc.extend(expr_ids(stmt.cond))
        walk(stmt.then_body)
        walk(stmt.else_body)
    return set(acc)


def _stmt_bucket_key(stmt: Stmt) -> str:
    names = _stmt_names(stmt)
    return skip_partition_key(sorted(names)[0]) if names else "_"


def c_type(width: int) -> str:
    if width <= 8:
        return "uint8_t"
    if width <= 16:
        return "uint16_t"
    if width <= 32:
        return "uint32_t"
    if width <= 64:
        return "uint64_t"
    if width <= 128:
        return "unsigned __int128"
    raise ValueError(f"width {width} > 128 needs limb storage")


def is_wide(width: int) -> bool:
    return width > 128


def limb_count(width: int) -> int:
    return (width + 63) // 64


def _storage_decl(name: str, width: int, depth: int = 0, init: bool = True) -> str:
    suffix = " = {}" if init else ""
    zero = " = 0" if init else ""
    if is_wide(width):
        n = limb_count(width)
        if depth:
            return f"uint64_t {name}[{depth}][{n}]{suffix};"
        return f"uint64_t {name}[{n}]{suffix};"
    ty = c_type(width)
    if depth:
        return f"{ty} {name}[{depth}]{suffix};"
    return f"{ty} {name}{zero};"


def mask_expr(width: int) -> str | None:
    if width >= 128:
        return None
    if width >= 64:
        if width == 64:
            return None
        return f"((((unsigned __int128)1) << {width}) - 1)"
    return hex((1 << width) - 1)


def _emit_const(value: int, width: int) -> str:
    value &= (1 << width) - 1 if width < 128 else (1 << 128) - 1
    if width <= 32:
        return f"{value}u"
    if width <= 64:
        return f"{value}ull"
    lo = value & ((1 << 64) - 1)
    hi = value >> 64
    if hi == 0:
        return f"((unsigned __int128){lo}ull)"
    return f"((((unsigned __int128){hi}ull) << 64) | {lo}ull)"


def _zero_lit(width: int) -> str:
    if width <= 32:
        return "0u"
    if width <= 64:
        return "0ull"
    return "(unsigned __int128)0"


def _shift_call(op: str, src: str, amt: str, width: int) -> str:
    if width <= 32:
        fn = "shl" if op == "<<" else "shr"
    elif width <= 64:
        fn = "shl64" if op == "<<" else "shr64"
    else:
        fn = "shl128" if op == "<<" else "shr128"
    return f"{fn}({src}, {amt})"


def _signed_cast(code: str, width: int) -> str:
    sign = 1 << (width - 1)
    if width <= 64:
        sl = f"{sign}ull"
        return f"((int64_t)(((uint64_t)({code})) ^ {sl}) - (int64_t){sl})"
    sl = _emit_const(sign, 128)
    return (
        f"((__int128)(((unsigned __int128)({code})) ^ {sl}) - (__int128){sl})"
    )


def emit_expr(expr: Expr, sigs: dict[str, Signal]) -> str:
    if isinstance(expr, Id):
        return expr.name
    if isinstance(expr, Const):
        return _emit_const(expr.value, expr.width)
    if isinstance(expr, UnaryOp):
        inner = emit_expr(expr.a, sigs)
        width = _expr_width(expr, sigs)
        if expr.op in {"~", "!"} and width == 1:
            return f"(!{inner})"
        code = f"({expr.op}{inner})"
        mask = mask_expr(width)
        return f"({code} & {mask})" if mask else code
    if isinstance(expr, BinOp):
        if expr.op == "<<":
            w = _expr_width(expr.a, sigs)
            return _shift_call("<<", emit_expr(expr.a, sigs), emit_expr(expr.b, sigs), w)
        if expr.op == ">>":
            w = _expr_width(expr.a, sigs)
            return _shift_call(">>", emit_expr(expr.a, sigs), emit_expr(expr.b, sigs), w)
        if expr.op in {"s<", "s<=", "s>", "s>="}:
            op = {"s<": "<", "s<=": "<=", "s>": ">", "s>=": ">="}[expr.op]
            w = _expr_width(expr.a, sigs)
            left = _signed_cast(emit_expr(expr.a, sigs), w)
            right = _signed_cast(emit_expr(expr.b, sigs), w)
            return f"({left} {op} {right})"
        if expr.op in {"s/", "s%"}:
            op = {"s/": "/", "s%": "%"}[expr.op]
            w = _expr_width(expr.a, sigs)
            left = _signed_cast(emit_expr(expr.a, sigs), w)
            right = _signed_cast(emit_expr(expr.b, sigs), w)
            return f"({left} {op} {right})"
        return f"({emit_expr(expr.a, sigs)} {expr.op} {emit_expr(expr.b, sigs)})"
    if isinstance(expr, Ternary):
        return f"({emit_expr(expr.cond, sigs)} ? {emit_expr(expr.a, sigs)} : {emit_expr(expr.b, sigs)})"
    if isinstance(expr, Extract):
        src = emit_expr(expr.a, sigs)
        if isinstance(expr.a, Id) and expr.a.name in sigs and is_wide(sigs[expr.a.name].width):
            return f"fs_getbits({expr.a.name}, {expr.low}u, {expr.width}u)"
        mask = mask_expr(expr.width)
        if expr.low:
            src_w = _expr_width(expr.a, sigs)
            shifted = _shift_call(">>", src, f"{expr.low}u", src_w)
        else:
            shifted = src
        return f"(({shifted}) & {mask})" if mask else f"({shifted})"
    if isinstance(expr, Concat):
        total = sum(_expr_width(part, sigs) for part in expr.parts)
        acc = _zero_lit(total)
        for part in expr.parts:
            width = _expr_width(part, sigs)
            piece = emit_expr(part, sigs)
            mask = mask_expr(width)
            if mask:
                piece = f"(({piece}) & {mask})"
            acc = f"(({acc} << {width}) | {piece})"
        return acc
    if isinstance(expr, ArrayGet):
        arr = emit_expr(expr.arr, sigs)
        idx = emit_expr(expr.index, sigs)
        depth = _array_depth(expr.arr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        return f"{arr}[{idx}]"
    if isinstance(expr, MemRead):
        depth = sigs[expr.mem].depth
        idx = emit_expr(expr.addr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        return f"{expr.mem}[{idx}]"
    raise TypeError(type(expr))


def _expr_width(expr: Expr, sigs: dict[str, Signal]) -> int:
    if isinstance(expr, Const):
        return expr.width
    if isinstance(expr, Id):
        return sigs[expr.name].width
    if isinstance(expr, Extract):
        return expr.width
    if isinstance(expr, Concat):
        return sum(_expr_width(part, sigs) for part in expr.parts)
    if isinstance(expr, UnaryOp):
        return _expr_width(expr.a, sigs)
    if isinstance(expr, BinOp):
        if expr.op in CMP_OPS:
            return 1
        if expr.op in {"<<", ">>"}:
            return _expr_width(expr.a, sigs)
        return max(_expr_width(expr.a, sigs), _expr_width(expr.b, sigs))
    if isinstance(expr, Ternary):
        return _expr_width(expr.a, sigs)
    if isinstance(expr, ArrayGet):
        if isinstance(expr.arr, Id):
            return sigs[expr.arr.name].width
        return _expr_width(expr.arr, sigs)
    if isinstance(expr, MemRead):
        return sigs[expr.mem].width
    raise TypeError(type(expr))


def _array_depth(expr: Expr, sigs: dict[str, Signal]) -> int:
    if isinstance(expr, Id):
        return sigs[expr.name].depth
    if isinstance(expr, ArrayInject) or isinstance(expr, ArrayZeros):
        if isinstance(expr, ArrayZeros):
            return expr.depth
        return _array_depth(expr.arr, sigs)
    return 0


def emit_cpp(mod: Module) -> str:
    assigns = assign_map(mod)
    regs = collect_regs(mod)
    mems = collect_mems(mod)
    inputs = [n for n in collect_inputs(mod) if n != mod.always.clock]
    outputs = collect_outputs(mod)
    sigs = mod.signals
    stop = set(regs) | set(collect_inputs(mod)) | set(mems)
    writes = always_writes(mod.always.body)
    cached = cached_wires(mod, assigns)
    wire_deps = {w: cone_leaves(w, assigns, stop) for w in cached}
    wire_part, leaf_parts, nparts = _partition_maps(cached, wire_deps)
    inv_tables = _invalidation_tables(leaf_parts)
    write_flags = [n for n in writes if not sigs[n].depth]
    large = len(writes) >= _LARGE_WRITES
    hold_buckets: list[tuple[str, list[Stmt]]] = []
    live_stmts: list[Stmt] = []
    leaf_holds: dict[str, list[int]] = {}
    if large:
        grouped: dict[str, list[Stmt]] = defaultdict(list)
        for stmt in mod.always.body:
            if isinstance(stmt, If) and not stmt.else_body:
                grouped[_stmt_bucket_key(stmt)].append(stmt)
            else:
                live_stmts.append(stmt)
        hold_buckets = sorted(grouped.items())
        holds_map: dict[str, set[int]] = defaultdict(set)
        for i, (_key, stmts) in enumerate(hold_buckets):
            for stmt in stmts:
                if isinstance(stmt, If):
                    for leaf in _cond_stop_leaves(stmt.cond, assigns, stop, wire_deps):
                        holds_map[leaf].add(i)
        leaf_holds = {k: sorted(v) for k, v in holds_map.items()}
    used_inputs = {
        name for name in inputs if name in leaf_parts or name in leaf_holds
    }
    array_writes = [n for n in writes if sigs[n].depth]
    global _WIRE_PART, _WRITE_INDEX, _WRITE_WIDE, _LEAF_HOLDS, _HOLD_WAKE_TABLES, _ARRAY_INDEX, _HOLD_TAKEN, _LARGE
    _WIRE_PART = wire_part
    _WRITE_INDEX = {n: i for i, n in enumerate(write_flags)} if large else {}
    _WRITE_WIDE = {n for n in write_flags if is_wide(sigs[n].width)} if large else set()
    _LEAF_HOLDS = leaf_holds
    _HOLD_WAKE_TABLES = _hold_wake_tables(leaf_holds)
    _ARRAY_INDEX = {n: i for i, n in enumerate(array_writes)} if large else {}
    _LARGE = large

    lines: list[str] = [
        "#include <cstdint>",
        "#include <cstring>",
        "",
        "static inline uint32_t shl(uint32_t x, uint32_t n) {",
        "  return n >= 32u ? 0u : (x << n);",
        "}",
        "static inline uint32_t shr(uint32_t x, uint32_t n) {",
        "  return n >= 32u ? 0u : (x >> n);",
        "}",
        "static inline uint64_t shl64(uint64_t x, uint32_t n) {",
        "  return n >= 64u ? 0ull : (x << n);",
        "}",
        "static inline uint64_t shr64(uint64_t x, uint32_t n) {",
        "  return n >= 64u ? 0ull : (x >> n);",
        "}",
        "static inline unsigned __int128 shl128(unsigned __int128 x, uint32_t n) {",
        "  return n >= 128u ? (unsigned __int128)0 : (x << n);",
        "}",
        "static inline unsigned __int128 shr128(unsigned __int128 x, uint32_t n) {",
        "  return n >= 128u ? (unsigned __int128)0 : (x >> n);",
        "}",
        "static inline uint64_t fs_getbits(const uint64_t *w, unsigned low, unsigned width) {",
        "  unsigned li = low / 64u;",
        "  unsigned off = low % 64u;",
        "  uint64_t v = w[li] >> off;",
        "  if (off && (off + width > 64u)) v |= w[li + 1] << (64u - off);",
        "  return width >= 64u ? v : (v & ((1ull << width) - 1ull));",
        "}",
        "static inline void fs_setbits(uint64_t *w, unsigned low, unsigned width, uint64_t val) {",
        "  unsigned li = low / 64u;",
        "  unsigned off = low % 64u;",
        "  uint64_t mask = width >= 64u ? ~0ull : ((1ull << width) - 1ull);",
        "  val &= mask;",
        "  if (off + width <= 64u) {",
        "    w[li] = (w[li] & ~(mask << off)) | (val << off);",
        "    return;",
        "  }",
        "  unsigned lo_bits = 64u - off;",
        "  uint64_t lo_mask = (1ull << lo_bits) - 1ull;",
        "  w[li] = (w[li] & ~(lo_mask << off)) | ((val & lo_mask) << off);",
        "  unsigned hi_bits = width - lo_bits;",
        "  uint64_t hi_mask = (1ull << hi_bits) - 1ull;",
        "  w[li + 1] = (w[li + 1] & ~hi_mask) | (val >> lo_bits);",
        "}",
        "static inline void fs_copy_bits(uint64_t *d, unsigned dlow, const uint64_t *s, unsigned slow, unsigned width) {",
        "  unsigned done = 0;",
        "  while (done < width) {",
        "    unsigned chunk = width - done;",
        "    if (chunk > 64u) chunk = 64u;",
        "    fs_setbits(d, dlow + done, chunk, fs_getbits(s, slow + done, chunk));",
        "    done += chunk;",
        "  }",
        "}",
        "static inline void fs_shl_limbs(uint64_t *d, const uint64_t *s, unsigned n, unsigned sh) {",
        "  memset(d, 0, n * sizeof(uint64_t));",
        "  if (sh >= n * 64u) return;",
        "  unsigned w = sh / 64u, b = sh % 64u;",
        "  if (b == 0) {",
        "    for (unsigned i = 0; i + w < n; i++) d[i + w] = s[i];",
        "    return;",
        "  }",
        "  for (unsigned i = 0; i + w < n; i++) {",
        "    d[i + w] |= s[i] << b;",
        "    if (i + w + 1u < n) d[i + w + 1u] |= s[i] >> (64u - b);",
        "  }",
        "}",
        "static inline void fs_shr_limbs(uint64_t *d, const uint64_t *s, unsigned n, unsigned sh) {",
        "  memset(d, 0, n * sizeof(uint64_t));",
        "  if (sh >= n * 64u) return;",
        "  unsigned w = sh / 64u, b = sh % 64u;",
        "  if (b == 0) {",
        "    for (unsigned i = 0; i + w < n; i++) d[i] = s[i + w];",
        "    return;",
        "  }",
        "  for (unsigned i = 0; i + w < n; i++) {",
        "    d[i] = s[i + w] >> b;",
        "    if (i + w + 1u < n) d[i] |= s[i + w + 1u] << (64u - b);",
        "  }",
        "}",
        "static inline void fs_and_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] & b[i];",
        "}",
        "static inline void fs_or_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] | b[i];",
        "}",
        "static inline void fs_xor_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) d[i] = a[i] ^ b[i];",
        "}",
        "static inline void fs_add_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  unsigned char c = 0;",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 s = (unsigned __int128)a[i] + b[i] + c;",
        "    d[i] = (uint64_t)s;",
        "    c = (unsigned char)(s >> 64);",
        "  }",
        "}",
        "static inline void fs_sub_limbs(uint64_t *d, const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  unsigned char br = 0;",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned __int128 x = a[i];",
        "    unsigned __int128 y = (unsigned __int128)b[i] + br;",
        "    br = (unsigned char)(x < y);",
        "    d[i] = (uint64_t)(x - y);",
        "  }",
        "}",
        "static inline int fs_ucmp_limbs(const uint64_t *a, const uint64_t *b, unsigned n) {",
        "  for (unsigned i = n; i-- > 0u; ) {",
        "    if (a[i] > b[i]) return 1;",
        "    if (a[i] < b[i]) return -1;",
        "  }",
        "  return 0;",
        "}",
        "static inline void fs_bump(uint32_t *pg, const uint16_t *ids, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) pg[ids[i]]++;",
        "}",
        "static inline void fs_wake(uint64_t *need, const uint16_t *ids, unsigned n) {",
        "  for (unsigned i = 0; i < n; i++) {",
        "    unsigned k = ids[i];",
        "    need[k >> 6] |= 1ull << (k & 63);",
        "  }",
        "}",
    ]
    for key, idx in sorted(inv_tables.items(), key=lambda kv: kv[1]):
        inner = ", ".join(str(p) for p in key)
        lines.append(f"static const uint16_t _inv{idx}[] = {{{inner}}};")
    for key, idx in sorted(_HOLD_WAKE_TABLES.items(), key=lambda kv: kv[1]):
        inner = ", ".join(str(p) for p in key)
        lines.append(f"static const uint16_t _hw{idx}[] = {{{inner}}};")
    lines.extend(
        [
        "",
        f"struct {mod.name}Dut {{",
        ]
    )
    for name in regs:
        sig = sigs[name]
        lines.append(f"  {_storage_decl(name, sig.width, sig.depth)}")
    for name in mems:
        sig = sigs[name]
        lines.append(f"  {_storage_decl(name, sig.width, sig.depth)}")
    for name in inputs:
        width = sigs[name].width
        lines.append(f"  {_storage_decl(name, width)}")
        if name in used_inputs:
            lines.append(f"  {_storage_decl(name + '__prev', width)}")
    for name in sorted(cached):
        sig = sigs.get(name)
        width = sig.width if sig else 32
        depth = sig.depth if sig else 0
        lines.append(f"  {_storage_decl(name, width, depth)}")
        lines.append(f"  uint32_t {name}__ok = 0;")
    if nparts:
        lines.append(f"  uint32_t _pg[{nparts}];")
        lines.append(f"  uint32_t _seq_seen[{nparts}] = {{}};")
    if large:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"  {_storage_decl(name + '__n', sig.width, sig.depth)}")
            else:
                lines.append(f"  {_storage_decl(name + '__n', sig.width)}")
        if write_flags:
            nwf = len(write_flags)
            lines.append(f"  uint8_t _w[{nwf}] = {{}};")
            lines.append(f"  uint16_t _wl[{nwf}] = {{}};")
            lines.append("  unsigned _nw = 0;")
            lines.append("  void _note(uint16_t i) { if (!_w[i]) { _w[i] = 1; _wl[_nw++] = i; } }")
        if _ARRAY_INDEX:
            lines.append(f"  uint8_t _ac[{len(_ARRAY_INDEX)}] = {{}};")
        if hold_buckets:
            nh = len(hold_buckets)
            nw = (nh + 63) // 64
            busy_init = ", ".join("~0ull" for _ in range(nw))
            lines.append(f"  uint64_t _h_need[{nw}] = {{}};")
            lines.append(f"  uint64_t _h_busy[{nw}] = {{{busy_init}}};")
    lines.append("  uint8_t __inited = 0;")
    lines.append("")
    lines.append("  void poke_inputs() {")
    lines.append("    if (!__inited) {")
    if nparts:
        lines.append(f"      for (unsigned i = 0; i < {nparts}u; i++) _pg[i] = 1;")
    for name in inputs:
        if name not in used_inputs:
            continue
        width = sigs[name].width
        if is_wide(width):
            lines.append(f"      memcpy({name}__prev, {name}, sizeof({name}));")
        else:
            lines.append(f"      {name}__prev = {name};")
    lines.append("      __inited = 1;")
    lines.append("      return;")
    lines.append("    }")
    for name in inputs:
        if name not in used_inputs:
            continue
        width = sigs[name].width
        if is_wide(width):
            lines.append(f"    if (memcmp({name}, {name}__prev, sizeof({name}))) {{")
        else:
            lines.append(f"    if ({name} != {name}__prev) {{")
        _bump_parts(name, leaf_parts, inv_tables, lines, 6)
        if is_wide(width):
            lines.append(f"      memcpy({name}__prev, {name}, sizeof({name}));")
        else:
            lines.append(f"      {name}__prev = {name};")
        lines.append("    }")
    lines.append("  }")
    lines.append("")
    for name in sorted(cached):
        _emit_eval_method(name, assigns, sigs, stop, cached, lines, wire_part[name])
    for name in outputs:
        if name not in cached:
            lines.append(f"  void eval_{name}() {{}}")
            lines.append("")
    if large:
        for i, (_key, stmts) in enumerate(hold_buckets):
            word, bit = i >> 6, i & 63
            mask = f"(1ull << {bit})"
            lines.append(f"  __attribute__((noinline)) void _nba_h{i}() {{")
            lines.append(f"    _h_need[{word}] &= ~{mask};")
            lines.append(f"    _h_busy[{word}] &= ~{mask};")
            _HOLD_TAKEN = f"_h_busy[{word}] |= {mask};"
            _emit_nba_tree(
                stmts,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                4,
                set(),
                [0],
            )
            _HOLD_TAKEN = ""
            lines.append("  }")
            lines.append("")
        if live_stmts:
            lines.append("  __attribute__((noinline)) void _nba_live() {")
            _emit_nba_tree(
                live_stmts, cached, assigns, stop, sigs, lines, 4, set(), [0]
            )
            lines.append("  }")
            lines.append("")
        if write_flags:
            lines.append("  void _commit() {")
            lines.append("    for (unsigned i = 0; i < _nw; i++) {")
            lines.append("      switch (_wl[i]) {")
            for name, idx in _WRITE_INDEX.items():
                sig = sigs[name]
                lines.append(f"      case {idx}u: {{")
                if is_wide(sig.width):
                    lines.append(
                        f"        if (memcmp({name}__n, {name}, sizeof({name}))) {{"
                    )
                    _bump_parts(name, leaf_parts, inv_tables, lines, 10)
                    lines.append(f"          memcpy({name}, {name}__n, sizeof({name}));")
                    lines.append("        }")
                else:
                    lines.append(f"        if ({name}__n != {name}) {{")
                    _bump_parts(name, leaf_parts, inv_tables, lines, 10)
                    lines.append(f"          {name} = {name}__n;")
                    lines.append("        }")
                lines.append("      } break;")
            lines.append("      }")
            lines.append("    }")
            lines.append("  }")
            lines.append("")
    lines.append("  void tick() {")
    lines.append("    poke_inputs();")
    seq_body = live_stmts if large else mod.always.body
    seq_cached = always_cond_reads(seq_body) & cached
    for wr in mod.mem_writes:
        seq_cached |= expr_ids(wr.enable) & cached
    _emit_eval_calls(seq_cached, cached, assigns, stop, lines, indent=4)
    if large:
        if write_flags:
            lines.append("    memset(_w, 0, sizeof(_w));")
            lines.append("    _nw = 0;")
        if _ARRAY_INDEX:
            lines.append("    memset(_ac, 0, sizeof(_ac));")
        if hold_buckets:
            nh = len(hold_buckets)
            nw = (nh + 63) // 64
            for w in range(nw):
                lo = w * 64
                hi = min(lo + 64, nh)
                last_bits = hi - lo
                mask = "~0ull" if last_bits == 64 else f"((1ull << {last_bits}) - 1ull)"
                lines.append(f"    {{ uint64_t __hm = (_h_need[{w}] | _h_busy[{w}]) & {mask};")
                lines.append("      if (__hm) {")
                for i in range(lo, hi):
                    b = i - lo
                    lines.append(f"        if (__hm & (1ull << {b})) _nba_h{i}();")
                lines.append("      } }")
        if live_stmts:
            lines.append("    _nba_live();")
    else:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"    {_storage_decl(name + '__n', sig.width, sig.depth, init=False)}")
                lines.append(f"    memcpy({name}__n, {name}, sizeof({name}__n));")
            elif is_wide(sig.width):
                lines.append(f"    {_storage_decl(name + '__n', sig.width, init=False)}")
                lines.append(f"    uint8_t {name}__w = 0;")
            else:
                lines.append(f"    {c_type(sig.width)} {name}__n;")
                lines.append(f"    uint8_t {name}__w = 0;")
        tick_scratch = [0]
        _emit_nba_tree(
            mod.always.body,
            cached,
            assigns,
            stop,
            sigs,
            lines,
            indent=4,
            declared=set(),
            scratch=tick_scratch,
        )
    tick_scratch = [0]
    for i, wr in enumerate(mod.mem_writes):
        depth = sigs[wr.mem].depth
        mask = depth - 1 if depth else 0
        w = sigs[wr.mem].width
        _emit_compute(wr.enable, cached, assigns, stop, sigs, lines, 4, set(), tick_scratch)
        lines.append(f"    uint8_t __we{i} = (uint8_t)({emit_expr(wr.enable, sigs)});")
        if is_wide(w):
            lines.append(f"    uint64_t __wd{i}[{limb_count(w)}];")
        else:
            lines.append(f"    {c_type(w)} __wd{i} = 0;")
        lines.append(f"    uint32_t __wa{i} = 0;")
        lines.append(f"    if (__we{i}) {{")
        decl: set[str] = set()
        _emit_compute(wr.data, cached, assigns, stop, sigs, lines, 6, decl, tick_scratch)
        _emit_compute(wr.addr, cached, assigns, stop, sigs, lines, 6, decl, tick_scratch)
        if is_wide(w):
            _emit_wide_assign(f"__wd{i}", wr.data, cached, sigs, lines, 6, tick_scratch)
        else:
            lines.append(
                f"      __wd{i} = {emit_expr(wr.data, sigs)};"
            )
        lines.append(
            f"      __wa{i} = ({emit_expr(wr.addr, sigs)}) & {mask}u;"
        )
        lines.append("    }")
    if large:
        if write_flags:
            lines.append("    _commit();")
        for name, idx in _ARRAY_INDEX.items():
            lines.append(
                f"    if (_ac[{idx}] && memcmp({name}__n, {name}, sizeof({name}))) {{"
            )
            _bump_parts(name, leaf_parts, inv_tables, lines, 6)
            lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
            lines.append("    }")
    else:
        for name in writes:
            sig = sigs[name]
            if sig.depth:
                lines.append(f"    if (memcmp({name}__n, {name}, sizeof({name}))) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
                lines.append("    }")
            elif is_wide(sig.width):
                lines.append(f"    if ({name}__w && memcmp({name}__n, {name}, sizeof({name}))) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append(f"      memcpy({name}, {name}__n, sizeof({name}));")
                lines.append("    }")
            else:
                lines.append(f"    if ({name}__w && {name}__n != {name}) {{")
                _bump_parts(name, leaf_parts, inv_tables, lines, 6)
                lines.append(f"      {name} = {name}__n;")
                lines.append("    }")
    for i, wr in enumerate(mod.mem_writes):
        w = sigs[wr.mem].width
        lines.append(f"    if (__we{i}) {{")
        if is_wide(w):
            lines.append(f"      if (memcmp({wr.mem}[__wa{i}], __wd{i}, sizeof(__wd{i}))) {{")
        else:
            lines.append(f"      if ({wr.mem}[__wa{i}] != __wd{i}) {{")
        _bump_parts(wr.mem, leaf_parts, inv_tables, lines, 8)
        lines.append("      }")
        if is_wide(w):
            lines.append(f"      memcpy({wr.mem}[__wa{i}], __wd{i}, sizeof(__wd{i}));")
        else:
            lines.append(f"      {wr.mem}[__wa{i}] = __wd{i};")
        lines.append("    }")
    lines.append("  }")
    lines.append("};")
    lines.append("")
    _WIRE_PART = {}
    _WRITE_INDEX = {}
    _WRITE_WIDE = set()
    _LEAF_HOLDS = {}
    _HOLD_WAKE_TABLES = {}
    _LARGE = False
    return "\n".join(lines) + "\n"


def _emit_eval_method(
    name: str,
    assigns: dict[str, Expr],
    sigs: dict[str, Signal],
    stop: set[str],
    cached: set[str],
    lines: list[str],
    part: int,
) -> None:
    expr = assigns[name]
    width = sigs[name].width if name in sigs else 32
    depth = sigs[name].depth if name in sigs else 0
    mask = None if depth else mask_expr(width)
    scratch = [0]
    lines.append(f"  void eval_{name}() {{")
    lines.append(f"    if ({name}__ok == _pg[{part}]) return;")
    for dep in cached_deps(name, assigns, cached, stop):
        lines.append(f"    eval_{dep}();")
    for tmp in internal_cone(name, assigns, cached, stop):
        tw = sigs[tmp].width if tmp in sigs else 32
        td = sigs[tmp].depth if tmp in sigs else 0
        if td:
            lines.append(f"    {_storage_decl(tmp, tw, td, init=False)}")
            _emit_array_assign(tmp, assigns[tmp], cached, sigs, lines, 4)
        elif is_wide(tw):
            lines.append(f"    {_storage_decl(tmp, tw, init=False)}")
            _emit_wide_assign(tmp, assigns[tmp], cached, sigs, lines, 4, scratch)
        elif _needs_limb_emit(assigns[tmp], sigs):
            tmask = mask_expr(tw)
            lines.append(f"    {c_type(tw)} {tmp};")
            _emit_assign(tmp, assigns[tmp], cached, sigs, lines, 4, tmask, scratch)
        else:
            tmask = mask_expr(tw)
            rhs = emit_expr(assigns[tmp], sigs)
            if tmask:
                lines.append(f"    {c_type(tw)} {tmp} = ({rhs}) & {tmask};")
            else:
                lines.append(f"    {c_type(tw)} {tmp} = {rhs};")
    if depth:
        _emit_array_assign(name, expr, cached, sigs, lines, 4)
    elif is_wide(width):
        _emit_wide_assign(name, expr, cached, sigs, lines, 4, scratch)
    else:
        _emit_assign(name, expr, cached, sigs, lines, 4, mask, scratch)
    lines.append(f"    {name}__ok = _pg[{part}];")
    lines.append("  }")
    lines.append("")


def _emit_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    mask: str | None,
    scratch: list[int] | None = None,
) -> None:
    sp = " " * indent
    if scratch is None:
        scratch = [0]
    if _expr_width(expr, sigs) > 128:
        _emit_wide_assign(dest, expr, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, (ArrayInject, ArrayZeros)) or (
        isinstance(expr, Id) and _is_array(sigs, expr.name)
    ):
        _emit_array_assign(dest, expr, cached, sigs, lines, indent)
        return
    if isinstance(expr, Ternary):
        _emit_demand(expr.cond, cached, lines, indent)
        lines.append(f"{sp}if ({emit_expr(expr.cond, sigs)}) {{")
        _emit_assign(dest, expr.a, cached, sigs, lines, indent + 2, mask, scratch)
        lines.append(f"{sp}}} else {{")
        _emit_assign(dest, expr.b, cached, sigs, lines, indent + 2, mask, scratch)
        lines.append(f"{sp}}}")
        return
    if _needs_limb_emit(expr, sigs):
        _emit_narrow_from_wide(
            dest, expr, _expr_width(expr, sigs), cached, sigs, lines, indent, scratch
        )
        if mask:
            lines.append(f"{sp}{dest} &= {mask};")
        return
    _emit_demand(expr, cached, lines, indent)
    rhs = emit_expr(expr, sigs)
    if mask:
        lines.append(f"{sp}{dest} = ({rhs}) & {mask};")
    else:
        lines.append(f"{sp}{dest} = {rhs};")


def _needs_limb_emit(expr: Expr, sigs: dict[str, Signal]) -> bool:
    """True when emit_expr would apply C operators to limb arrays."""
    if isinstance(expr, Const):
        return False
    if isinstance(expr, Id):
        return False
    if isinstance(expr, Extract):
        if isinstance(expr.a, Id) and expr.a.name in sigs and is_wide(sigs[expr.a.name].width):
            return False
        return is_wide(_expr_width(expr.a, sigs)) or _needs_limb_emit(expr.a, sigs)
    if isinstance(expr, Concat):
        total = sum(_expr_width(part, sigs) for part in expr.parts)
        if is_wide(total):
            return True
        return any(_needs_limb_emit(part, sigs) for part in expr.parts)
    if isinstance(expr, UnaryOp):
        return is_wide(_expr_width(expr.a, sigs)) or _needs_limb_emit(expr.a, sigs)
    if isinstance(expr, BinOp):
        if is_wide(_expr_width(expr.a, sigs)) or is_wide(_expr_width(expr.b, sigs)):
            return True
        return _needs_limb_emit(expr.a, sigs) or _needs_limb_emit(expr.b, sigs)
    if isinstance(expr, Ternary):
        return (
            _needs_limb_emit(expr.cond, sigs)
            or _needs_limb_emit(expr.a, sigs)
            or _needs_limb_emit(expr.b, sigs)
        )
    if isinstance(expr, ArrayGet):
        return _needs_limb_emit(expr.arr, sigs) or _needs_limb_emit(expr.index, sigs)
    if isinstance(expr, MemRead):
        return expr.mem in sigs and is_wide(sigs[expr.mem].width)
    return False


def _load_bits(src: str, low: int, width: int) -> str:
    if width <= 64:
        return f"fs_getbits({src}, {low}u, {width}u)"
    lo = f"fs_getbits({src}, {low}u, 64u)"
    hi = f"fs_getbits({src}, {low + 64}u, {width - 64}u)"
    return f"(((unsigned __int128)({hi}) << 64) | (unsigned __int128)({lo}))"


def _bind_narrow(
    expr: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    if isinstance(expr, Const) and not is_wide(expr.width):
        return _emit_const(expr.value, min(width, 128))
    if isinstance(expr, Id) and expr.name in sigs and not is_wide(sigs[expr.name].width):
        _emit_demand(expr, cached, lines, indent)
        return expr.name
    if not _needs_limb_emit(expr, sigs) and not is_wide(width):
        _emit_demand(expr, cached, lines, indent)
        return f"({emit_expr(expr, sigs)})"
    tmp = f"__ns{scratch[0]}"
    scratch[0] += 1
    lines.append(f"{' ' * indent}{c_type(width)} {tmp};")
    _emit_narrow_from_wide(tmp, expr, width, cached, sigs, lines, indent, scratch)
    return tmp


def _emit_narrow_from_wide(
    dest: str,
    expr: Expr,
    dest_w: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    sp = " " * indent
    ty = c_type(dest_w)
    if isinstance(expr, Ternary):
        _emit_demand(expr.cond, cached, lines, indent)
        lines.append(f"{sp}if ({emit_expr(expr.cond, sigs)}) {{")
        _emit_narrow_from_wide(dest, expr.a, dest_w, cached, sigs, lines, indent + 2, scratch)
        lines.append(f"{sp}}} else {{")
        _emit_narrow_from_wide(dest, expr.b, dest_w, cached, sigs, lines, indent + 2, scratch)
        lines.append(f"{sp}}}")
        return
    if isinstance(expr, Extract):
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, expr.low, expr.width)};")
            return
        src = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        shifted = _shift_call(">>", src, f"{expr.low}u", aw)
        mask = mask_expr(expr.width)
        lines.append(
            f"{sp}{dest} = ({ty})({shifted} & {mask});" if mask else f"{sp}{dest} = ({ty})({shifted});"
        )
        return
    if isinstance(expr, BinOp) and expr.op in {"<<", ">>"}:
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
            tmp = _wtmp(lines, indent, limb_count(aw), scratch)
            fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
            lines.append(f"{sp}{fn}({tmp}, {src}, {limb_count(aw)}u, {amt});")
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(tmp, 0, dest_w)};")
            return
        src = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}{dest} = ({ty}){_shift_call(expr.op, src, amt, aw)};")
        return
    if isinstance(expr, BinOp) and expr.op in CMP_OPS:
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        if is_wide(aw) or is_wide(bw):
            if expr.op.startswith("s"):
                raise TypeError(f"signed compare of width {max(aw, bw)}")
            left = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
            right = _materialize_limbs(expr.b, bw, cached, sigs, lines, indent, scratch)
            n = max(limb_count(aw), limb_count(bw))
            cop = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[expr.op]
            lines.append(
                f"{sp}{dest} = ({ty})(fs_ucmp_limbs({left}, {right}, {n}u) {cop} 0);"
            )
            return
        left = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _bind_narrow(expr.b, bw, cached, sigs, lines, indent, scratch)
        if expr.op.startswith("s"):
            op = {"s<": "<", "s<=": "<=", "s>": ">", "s>=": ">="}[expr.op]
            left = _signed_cast(left, aw)
            right = _signed_cast(right, bw)
            lines.append(f"{sp}{dest} = ({ty})({left} {op} {right});")
            return
        lines.append(f"{sp}{dest} = ({ty})({left} {expr.op} {right});")
        return
    if isinstance(expr, BinOp):
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        if is_wide(aw) or is_wide(bw):
            src = _materialize_limbs(expr, max(aw, bw), cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        left = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _bind_narrow(expr.b, bw, cached, sigs, lines, indent, scratch)
        if expr.op in {"s/", "s%"}:
            op = {"s/": "/", "s%": "%"}[expr.op]
            left = _signed_cast(left, aw)
            right = _signed_cast(right, bw)
            lines.append(f"{sp}{dest} = ({ty})({left} {op} {right});")
            return
        lines.append(f"{sp}{dest} = ({ty})({left} {expr.op} {right});")
        return
    if isinstance(expr, Concat):
        total = _expr_width(expr, sigs)
        if is_wide(total):
            src = _materialize_limbs(expr, total, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        terms: list[str] = []
        bit = 0
        for part in reversed(expr.parts):
            pw = _expr_width(part, sigs)
            val = _bind_narrow(part, pw, cached, sigs, lines, indent, scratch)
            mask = mask_expr(pw)
            piece = f"(({val}) & {mask})" if mask else f"({val})"
            terms.append(f"((({ty}){piece}) << {bit})")
            bit += pw
        lines.append(f"{sp}{dest} = {' | '.join(terms) if terms else '0'};")
        return
    if isinstance(expr, UnaryOp):
        aw = _expr_width(expr.a, sigs)
        if is_wide(aw):
            src = _materialize_limbs(expr, aw, cached, sigs, lines, indent, scratch)
            lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
            return
        inner = _bind_narrow(expr.a, aw, cached, sigs, lines, indent, scratch)
        if expr.op in {"~", "!"} and dest_w == 1:
            lines.append(f"{sp}{dest} = ({ty})(!{inner});")
            return
        mask = mask_expr(dest_w)
        code = f"({expr.op}{inner})"
        lines.append(
            f"{sp}{dest} = ({ty})({code} & {mask});" if mask else f"{sp}{dest} = ({ty}){code};"
        )
        return
    if isinstance(expr, Id) and is_wide(_expr_width(expr, sigs)):
        _emit_demand(expr, cached, lines, indent)
        lines.append(f"{sp}{dest} = ({ty}){_load_bits(expr.name, 0, dest_w)};")
        return
    if isinstance(expr, Const):
        lines.append(f"{sp}{dest} = {_emit_const(expr.value, dest_w)};")
        return
    if isinstance(expr, MemRead) and is_wide(_expr_width(expr, sigs)):
        src = _materialize_limbs(expr, _expr_width(expr, sigs), cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}{dest} = ({ty}){_load_bits(src, 0, dest_w)};")
        return
    _emit_demand(expr, cached, lines, indent)
    lines.append(f"{sp}{dest} = {emit_expr(expr, sigs)};")


def _wtmp(lines: list[str], indent: int, nlimbs: int, scratch: list[int]) -> str:
    i = scratch[0]
    scratch[0] += 1
    name = f"__wl{i}"
    lines.append(f"{' ' * indent}uint64_t {name}[{nlimbs}];")
    return name


def _try_fold_shift_amt(expr: Expr, sigs: dict[str, Signal]) -> str | None:
    if isinstance(expr, Const):
        return f"{expr.value}u"
    if isinstance(expr, Id):
        w = sigs[expr.name].width if expr.name in sigs else 32
        if w <= 32:
            return f"(unsigned)({expr.name})"
        return None
    if not isinstance(expr, Concat):
        return None
    bit = 0
    terms: list[str] = []
    for part in reversed(expr.parts):
        pw = _expr_width(part, sigs)
        if isinstance(part, Const):
            if part.value and bit < 32:
                terms.append(f"({part.value}u << {bit})")
        elif isinstance(part, Id) and pw <= 32:
            terms.append(f"((unsigned)({part.name}) << {bit})")
        else:
            return None
        bit += pw
        if bit >= 32:
            break
    return "(" + " | ".join(terms) + ")" if terms else "0u"


def _emit_shift_amt(
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    folded = _try_fold_shift_amt(expr, sigs)
    if folded is not None:
        _emit_demand(expr, cached, lines, indent)
        return folded
    w = _expr_width(expr, sigs)
    if not is_wide(w):
        _emit_demand(expr, cached, lines, indent)
        return f"(unsigned)({emit_expr(expr, sigs)})"
    tmp = _wtmp(lines, indent, limb_count(w), scratch)
    _emit_wide_assign(tmp, expr, cached, sigs, lines, indent, scratch)
    return f"(unsigned)fs_getbits({tmp}, 0u, 32u)"


def _emit_setbits_scalar(
    dest: str,
    offset: int,
    width: int,
    src_c: str,
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    """Write a ≤128-bit C integer into a limb array. 65–128 bits need two stores."""
    sp = " " * indent
    if width <= 64:
        lines.append(
            f"{sp}fs_setbits({dest}, {offset}u, {width}u, (uint64_t)({src_c}));"
        )
        return
    tmp = f"__i{scratch[0]}"
    scratch[0] += 1
    lines.append(f"{sp}unsigned __int128 {tmp} = (unsigned __int128)({src_c});")
    lines.append(f"{sp}fs_setbits({dest}, {offset}u, 64u, (uint64_t){tmp});")
    lines.append(
        f"{sp}fs_setbits({dest}, {offset + 64}u, {width - 64}u, "
        f"(uint64_t)({tmp} >> 64));"
    )


def _place_bits(
    dest: str,
    offset: int,
    part: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> None:
    sp = " " * indent
    if width <= 0:
        return
    if isinstance(part, Const):
        if part.value == 0:
            return
        val = part.value
        done = 0
        while done < width:
            chunk = min(64, width - done)
            chunk_val = (val >> done) & ((1 << chunk) - 1 if chunk < 64 else (1 << 64) - 1)
            lines.append(
                f"{sp}fs_setbits({dest}, {offset + done}u, {chunk}u, {chunk_val}ull);"
            )
            done += chunk
        return
    if isinstance(part, Id) and not is_wide(_expr_width(part, sigs)):
        _emit_demand(part, cached, lines, indent)
        _emit_setbits_scalar(
            dest, offset, width, emit_expr(part, sigs), lines, indent, scratch
        )
        return
    src = _materialize_limbs(part, width, cached, sigs, lines, indent, scratch)
    lines.append(f"{sp}fs_copy_bits({dest}, {offset}u, {src}, 0u, {width}u);")


def _materialize_limbs(
    expr: Expr,
    width: int,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int],
) -> str:
    if isinstance(expr, Id) and expr.name in sigs and is_wide(sigs[expr.name].width):
        _emit_demand(expr, cached, lines, indent)
        return expr.name
    tmp = _wtmp(lines, indent, limb_count(width), scratch)
    _emit_wide_assign(tmp, expr, cached, sigs, lines, indent, scratch)
    return tmp


def _emit_wide_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    scratch: list[int] | None = None,
) -> None:
    if scratch is None:
        scratch = [0]
    sp = " " * indent
    width = _expr_width(expr, sigs)
    n = limb_count(width)
    if isinstance(expr, Ternary):
        _emit_demand(expr.cond, cached, lines, indent)
        lines.append(f"{sp}if ({emit_expr(expr.cond, sigs)}) {{")
        _emit_wide_assign(dest, expr.a, cached, sigs, lines, indent + 2, scratch)
        lines.append(f"{sp}}} else {{")
        _emit_wide_assign(dest, expr.b, cached, sigs, lines, indent + 2, scratch)
        lines.append(f"{sp}}}")
        return
    lines.append(f"{sp}memset({dest}, 0, sizeof({dest}));")
    if isinstance(expr, Const):
        if expr.value:
            _place_bits(dest, 0, expr, width, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, Id):
        src_w = sigs[expr.name].width if expr.name in sigs else width
        if is_wide(src_w):
            copy_w = min(width, src_w)
            lines.append(f"{sp}fs_copy_bits({dest}, 0u, {expr.name}, 0u, {copy_w}u);")
        else:
            _emit_demand(expr, cached, lines, indent)
            _emit_setbits_scalar(
                dest, 0, min(width, src_w), emit_expr(expr, sigs), lines, indent, scratch
            )
        return
    if isinstance(expr, MemRead):
        _emit_demand(expr, cached, lines, indent)
        depth = sigs[expr.mem].depth
        idx = emit_expr(expr.addr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        mem_w = sigs[expr.mem].width
        if is_wide(mem_w):
            lines.append(f"{sp}memcpy({dest}, {expr.mem}[{idx}], sizeof({dest}));")
        else:
            _emit_setbits_scalar(
                dest, 0, mem_w, f"{expr.mem}[{idx}]", lines, indent, scratch
            )
        return
    if isinstance(expr, Concat):
        _emit_demand(expr, cached, lines, indent)
        offset = width
        for part in expr.parts:
            w = _expr_width(part, sigs)
            offset -= w
            _place_bits(dest, offset, part, w, cached, sigs, lines, indent, scratch)
        return
    if isinstance(expr, Extract):
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        lines.append(
            f"{sp}fs_copy_bits({dest}, 0u, {src}, {expr.low}u, {expr.width}u);"
        )
        return
    if isinstance(expr, BinOp) and expr.op in {"<<", ">>"}:
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        amt = _emit_shift_amt(expr.b, cached, sigs, lines, indent, scratch)
        src_n = limb_count(src_w)
        if src_n == n and dest != src:
            fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
            lines.append(f"{sp}{fn}({dest}, {src}, {src_n}u, {amt});")
            return
        tmp = _wtmp(lines, indent, src_n, scratch)
        fn = "fs_shl_limbs" if expr.op == "<<" else "fs_shr_limbs"
        lines.append(f"{sp}{fn}({tmp}, {src}, {src_n}u, {amt});")
        lines.append(f"{sp}fs_copy_bits({dest}, 0u, {tmp}, 0u, {width}u);")
        return
    if isinstance(expr, BinOp) and expr.op in {"&", "|", "^", "+", "-"}:
        aw = _expr_width(expr.a, sigs)
        bw = _expr_width(expr.b, sigs)
        left = _materialize_limbs(expr.a, aw, cached, sigs, lines, indent, scratch)
        right = _materialize_limbs(expr.b, bw, cached, sigs, lines, indent, scratch)
        fn = {
            "&": "fs_and_limbs",
            "|": "fs_or_limbs",
            "^": "fs_xor_limbs",
            "+": "fs_add_limbs",
            "-": "fs_sub_limbs",
        }[expr.op]
        use_n = max(limb_count(aw), limb_count(bw), n)
        if use_n == n:
            lines.append(f"{sp}{fn}({dest}, {left}, {right}, {n}u);")
        else:
            tmp = _wtmp(lines, indent, use_n, scratch)
            lines.append(f"{sp}{fn}({tmp}, {left}, {right}, {use_n}u);")
            lines.append(f"{sp}fs_copy_bits({dest}, 0u, {tmp}, 0u, {width}u);")
        return
    if isinstance(expr, UnaryOp) and expr.op == "~":
        src_w = _expr_width(expr.a, sigs)
        src = _materialize_limbs(expr.a, src_w, cached, sigs, lines, indent, scratch)
        lines.append(f"{sp}for (unsigned __i = 0; __i < {n}u; __i++) {dest}[__i] = ~{src}[__i];")
        return
    if isinstance(expr, ArrayGet):
        _emit_demand(expr, cached, lines, indent)
        arr = emit_expr(expr.arr, sigs)
        idx = emit_expr(expr.index, sigs)
        depth = _array_depth(expr.arr, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        elem_w = _expr_width(expr, sigs)
        if is_wide(elem_w):
            lines.append(f"{sp}memcpy({dest}, {arr}[{idx}], sizeof({dest}));")
        else:
            _emit_setbits_scalar(
                dest, 0, elem_w, f"{arr}[{idx}]", lines, indent, scratch
            )
        return
    raise TypeError(f"wide assign {type(expr)} {getattr(expr, 'op', '')}")


def _emit_array_assign(
    dest: str,
    expr: Expr,
    cached: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
) -> None:
    sp = " " * indent
    if isinstance(expr, Ternary):
        _emit_demand(expr.cond, cached, lines, indent)
        lines.append(f"{sp}if ({emit_expr(expr.cond, sigs)}) {{")
        _emit_array_assign(dest, expr.a, cached, sigs, lines, indent + 2)
        lines.append(f"{sp}}} else {{")
        _emit_array_assign(dest, expr.b, cached, sigs, lines, indent + 2)
        lines.append(f"{sp}}}")
        return
    if isinstance(expr, ArrayZeros):
        lines.append(f"{sp}memset({dest}, 0, sizeof({dest}));")
        return
    if isinstance(expr, Id):
        lines.append(f"{sp}memcpy({dest}, {expr.name}, sizeof({dest}));")
        return
    if isinstance(expr, ArrayInject):
        _emit_array_assign(dest, expr.arr, cached, sigs, lines, indent)
        _emit_demand(expr.index, cached, lines, indent)
        _emit_demand(expr.value, cached, lines, indent)
        depth = _array_depth(expr, sigs)
        idx = emit_expr(expr.index, sigs)
        if depth:
            idx = f"(({idx}) & {depth - 1}u)"
        val_w = _expr_width(expr.value, sigs)
        val = emit_expr(expr.value, sigs)
        if is_wide(val_w):
            lines.append(f"{sp}memcpy({dest}[{idx}], {val}, sizeof({dest}[{idx}]));")
        else:
            lines.append(f"{sp}{dest}[{idx}] = {val};")
        return
    raise TypeError(f"array assign {type(expr)}")


def _emit_eval_calls(
    names: set[str],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    lines: list[str],
    indent: int,
) -> None:
    seen: set[str] = set()
    ordered: list[str] = []

    def walk(name: str) -> None:
        if name not in cached or name in seen:
            return
        seen.add(name)
        for dep in cached_deps(name, assigns, cached, stop):
            walk(dep)
        ordered.append(name)

    for name in sorted(names):
        walk(name)
    if not _WIRE_PART:
        for name in ordered:
            lines.append(_eval_invoke(name, indent))
        return
    groups: dict[int, list[str]] = {}
    for name in ordered:
        groups.setdefault(_WIRE_PART[name], []).append(name)
    sp = " " * indent
    for p, group in sorted(groups.items()):
        lines.append(f"{sp}if (_seq_seen[{p}] != _pg[{p}]) {{")
        for name in group:
            lines.append(f"{sp}  eval_{name}();")
        lines.append(f"{sp}  _seq_seen[{p}] = _pg[{p}];")
        lines.append(f"{sp}}}")


def _emit_demand(
    expr: Expr,
    cached: set[str],
    lines: list[str],
    indent: int,
) -> None:
    seen: set[str] = set()
    for dep in expr_ids(expr):
        if dep in cached and dep not in seen:
            seen.add(dep)
            lines.append(_eval_invoke(dep, indent))


def _uncached_needed(
    expr: Expr,
    assigns: dict[str, Expr],
    cached: set[str],
    stop: set[str],
) -> list[str]:
    needed: list[str] = []
    seen: set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name in stop or name in cached or name not in assigns:
            return
        seen.add(name)
        for dep in expr_ids(assigns[name]):
            walk(dep)
        needed.append(name)

    for name in expr_ids(expr):
        walk(name)
    return needed


def _emit_compute(
    expr: Expr,
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int] | None = None,
) -> None:
    """Eval cached deps and bind uncached cone wires as locals."""
    if scratch is None:
        scratch = [0]
    sp = " " * indent
    needed = _uncached_needed(expr, assigns, cached, stop)
    demand: set[str] = set()
    for dep in expr_ids(expr):
        if dep in cached:
            demand.add(dep)
    for tmp in needed:
        for dep in expr_ids(assigns[tmp]):
            if dep in cached:
                demand.add(dep)
    for dep in sorted(demand):
        lines.append(_eval_invoke(dep, indent))
    for tmp in needed:
        if tmp in declared:
            continue
        declared.add(tmp)
        tw = sigs[tmp].width if tmp in sigs else 32
        td = sigs[tmp].depth if tmp in sigs else 0
        if td:
            lines.append(f"{sp}{_storage_decl(tmp, tw, td, init=False)}")
            _emit_array_assign(tmp, assigns[tmp], cached, sigs, lines, indent)
        elif is_wide(tw):
            lines.append(f"{sp}{_storage_decl(tmp, tw, init=False)}")
            _emit_wide_assign(tmp, assigns[tmp], cached, sigs, lines, indent, scratch)
        elif _needs_limb_emit(assigns[tmp], sigs):
            tmask = mask_expr(tw)
            lines.append(f"{sp}{c_type(tw)} {tmp};")
            _emit_assign(tmp, assigns[tmp], cached, sigs, lines, indent, tmask, scratch)
        else:
            tmask = mask_expr(tw)
            rhs = emit_expr(assigns[tmp], sigs)
            if tmask:
                lines.append(f"{sp}{c_type(tw)} {tmp} = ({rhs}) & {tmask};")
            else:
                lines.append(f"{sp}{c_type(tw)} {tmp} = {rhs};")


def _emit_nba_tree(
    stmts: list[Stmt],
    cached: set[str],
    assigns: dict[str, Expr],
    stop: set[str],
    sigs: dict[str, Signal],
    lines: list[str],
    indent: int,
    declared: set[str],
    scratch: list[int] | None = None,
) -> None:
    if scratch is None:
        scratch = [0]
    sp = " " * indent
    for stmt in stmts:
        if isinstance(stmt, NbAssign):
            tmp = f"{stmt.lhs}__n"
            _emit_compute(stmt.rhs, cached, assigns, stop, sigs, lines, indent, declared, scratch)
            if _is_array(sigs, stmt.lhs):
                aidx = _ARRAY_INDEX.get(stmt.lhs)
                if aidx is not None:
                    lines.append(f"{sp}_ac[{aidx}] = 1;")
                    if _HOLD_TAKEN:
                        lines.append(f"{sp}{_HOLD_TAKEN}")
                _emit_array_assign(tmp, stmt.rhs, cached, sigs, lines, indent)
            elif is_wide(sigs[stmt.lhs].width):
                _emit_wide_assign(tmp, stmt.rhs, cached, sigs, lines, indent, scratch)
                _mark_write(stmt.lhs, lines, indent)
            elif _needs_limb_emit(stmt.rhs, sigs):
                mask = mask_expr(sigs[stmt.lhs].width)
                _emit_assign(tmp, stmt.rhs, cached, sigs, lines, indent, mask, scratch)
                _mark_write(stmt.lhs, lines, indent)
            else:
                rhs_c = emit_expr(stmt.rhs, sigs)
                mask = mask_expr(sigs[stmt.lhs].width)
                if mask:
                    lines.append(f"{sp}{tmp} = ({rhs_c}) & {mask};")
                else:
                    lines.append(f"{sp}{tmp} = {rhs_c};")
                _mark_write(stmt.lhs, lines, indent)
        elif isinstance(stmt, If):
            _emit_compute(stmt.cond, cached, assigns, stop, sigs, lines, indent, declared, scratch)
            lines.append(f"{sp}if ({emit_expr(stmt.cond, sigs)}) {{")
            _emit_nba_tree(
                stmt.then_body,
                cached,
                assigns,
                stop,
                sigs,
                lines,
                indent + 2,
                set(declared),
                scratch,
            )
            if stmt.else_body:
                lines.append(f"{sp}}} else {{")
                _emit_nba_tree(
                    stmt.else_body,
                    cached,
                    assigns,
                    stop,
                    sigs,
                    lines,
                    indent + 2,
                    set(declared),
                    scratch,
                )
                lines.append(f"{sp}}}")
            else:
                lines.append(f"{sp}}}")
        else:
            raise TypeError(stmt)
