from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchIO:
    decls: str
    stimulus: str
    poke: str
    dump: str
    sink: str
    clock: str = "clk"
    prelude: str = ""
    tail: str = ""


_STD = BenchIO(
    decls="    uint8_t rst;\n    uint8_t en;\n    uint32_t din;",
    stimulus="",
    poke="",
        dump='    std::printf("%08x\\n", {p}dout);\n',
    sink="{p}dout",
)


def _std(stim: str) -> BenchIO:
    return BenchIO(
        decls=_STD.decls,
        stimulus=stim,
        poke="    {p}rst = rst;\n    {p}en = en;\n    {p}din = din;",
        dump=_STD.dump,
        sink=_STD.sink,
    )


# DrawContext fields as flattened by Chisel/firtool. Types match FlashSim c_type.
_FIFO_CTX = (
    ("shaderPc", "uint32_t", "(uint32_t)(0x1000u + (uint32_t)i)"),
    ("kernargBase", "uint32_t", "(uint32_t)(i << 4)"),
    ("kernargBankStride", "uint32_t", "(uint32_t)(i * 3ull)"),
    ("texBase", "uint32_t", "(uint32_t)(0x80000000u + (uint32_t)i)"),
    ("texWidth", "uint16_t", "(uint16_t)((i * 7ull) & 0x3fffull)"),
    ("texHeight", "uint16_t", "(uint16_t)((i * 11ull) & 0x3fffull)"),
    ("texWrapClamp", "uint8_t", "(uint8_t)((i >> 1) & 1ull)"),
    ("texMaxLevel", "uint8_t", "(uint8_t)(i & 15ull)"),
    ("texLodBias", "uint8_t", "(uint8_t)(i & 31ull)"),
    ("texMinLevel", "uint8_t", "(uint8_t)((i >> 2) & 15ull)"),
    ("colorBase", "uint32_t", "(uint32_t)(0x20000000u + (uint32_t)i)"),
    ("depthBase", "uint32_t", "(uint32_t)(0x30000000u + (uint32_t)i)"),
    ("stride", "uint32_t", "(uint32_t)((i & 255ull) * 4ull)"),
    ("depthTestEnable", "uint8_t", "(uint8_t)((i >> 3) & 1ull)"),
    ("depthFunc", "uint8_t", "(uint8_t)(i & 7ull)"),
    ("depthWriteEnable", "uint8_t", "(uint8_t)((i >> 4) & 1ull)"),
    ("blendEnable", "uint8_t", "(uint8_t)((i >> 5) & 1ull)"),
)


def _draw_fifo_io() -> BenchIO:
    enq = [(f"io_enq_bits_{name}", ty, stim) for name, ty, stim in _FIFO_CTX]
    controls = [
        ("reset", "uint8_t", "(uint8_t)(i < 4)"),
        ("io_enq_valid", "uint8_t", "(uint8_t)((i >= 4) && ((i % 3ull) == 0))"),
        ("io_retire", "uint8_t", "(uint8_t)((i >= 12) && ((i % 5ull) == 0))"),
    ]
    inputs = controls + enq
    decls = "\n".join(f"    {ty} {name};" for name, ty, _ in inputs)
    stimulus = "\n".join(f"    {name} = {stim};" for name, _, stim in inputs)
    poke = "\n".join(f"    {{p}}{name} = {name};" for name, _, _ in inputs)
    dump_fields = [
        "io_enq_ready",
        "io_headValid",
        "io_tailValid",
        *[f"io_head_{name}" for name, _, _ in _FIFO_CTX],
        *[f"io_tail_{name}" for name, _, _ in _FIFO_CTX],
    ]
    fmt = " ".join("%x" for _ in dump_fields)
    args = ", ".join("{p}" + n for n in dump_fields)
    dump = f'    std::printf("{fmt}\\n", {args});\n'
    sink = (
        "(uint64_t){p}io_enq_ready + {p}io_headValid + {p}io_tailValid + "
        "{p}io_head_shaderPc + {p}io_tail_shaderPc"
    )
    return BenchIO(decls, stimulus, poke, dump, sink, clock="clock")


def _raster_io() -> BenchIO:
    inputs = [
        ("reset", "uint8_t", "(uint8_t)(i < 4)"),
        ("io_draw_valid", "uint8_t", "(uint8_t)((i >= 4) && ((i % 512ull) < 4))"),
        ("io_draw_bits_v0_x", "uint32_t", "8u"),
        ("io_draw_bits_v0_y", "uint32_t", "8u"),
        ("io_draw_bits_v1_x", "uint32_t", "40u"),
        ("io_draw_bits_v1_y", "uint32_t", "8u"),
        ("io_draw_bits_v2_x", "uint32_t", "24u"),
        ("io_draw_bits_v2_y", "uint32_t", "36u"),
        ("io_cullMode", "uint8_t", "0u"),
        ("io_pixel_ready", "uint8_t", "1u"),
        ("io_quad_ready", "uint8_t", "1u"),
    ]
    decls = "\n".join(f"    {ty} {name};" for name, ty, _ in inputs)
    stimulus = "\n".join(f"    {name} = {stim};" for name, _, stim in inputs)
    poke = "\n".join(f"    {{p}}{name} = {name};" for name, _, _ in inputs)
    dump_u32 = ["io_draw_ready", "io_pixel_valid", "io_pixel_bits_x", "io_pixel_bits_y", "io_pixel_bits_covered", "io_quad_valid"]
    dump_u64 = ["io_pixel_bits_e0", "io_pixel_bits_e1", "io_pixel_bits_e2", "io_pixel_bits_area"]
    for lane in range(4):
        dump_u32 += [
            f"io_quad_bits_lanes_{lane}_x",
            f"io_quad_bits_lanes_{lane}_y",
            f"io_quad_bits_lanes_{lane}_covered",
        ]
        dump_u64 += [
            f"io_quad_bits_lanes_{lane}_e0",
            f"io_quad_bits_lanes_{lane}_e1",
            f"io_quad_bits_lanes_{lane}_e2",
            f"io_quad_bits_lanes_{lane}_area",
        ]
    fmt = " ".join(["%x"] * len(dump_u32) + ["%llx"] * len(dump_u64))
    args = ", ".join(
        ["{p}" + n for n in dump_u32]
        + [f"(unsigned long long){{p}}{n}" for n in dump_u64]
    )
    dump = f'    std::printf("{fmt}\\n", {args});\n'
    sink = (
        "(uint64_t){p}io_draw_ready + {p}io_pixel_valid + {p}io_quad_valid + "
        "{p}io_pixel_bits_x + {p}io_pixel_bits_y + (uint64_t){p}io_pixel_bits_e0"
    )
    return BenchIO(decls, stimulus, poke, dump, sink, clock="clock")


def _warp_io() -> BenchIO:
    inputs = [
        ("reset", "uint8_t", "(uint8_t)(i < 4)"),
        ("io_launch_valid", "uint8_t", "(uint8_t)((i >= 8) && (i < 12))"),
        ("io_launch_bits_warpId", "uint8_t", "(uint8_t)((i - 8ull) & 3ull)"),
        ("io_launch_bits_startPc", "uint32_t", "(uint32_t)(0x80001000u + (((uint32_t)i - 8u) << 8))"),
        ("io_launch_bits_activeMask", "uint8_t", "0xfu"),
        ("io_issue_ready", "uint8_t", "(uint8_t)((i % 3ull) != 0)"),
        ("io_resume_valid", "uint8_t", "(uint8_t)((i >= 16) && ((i % 5ull) == 0) && (i < 200))"),
        ("io_resume_bits_warpId", "uint8_t", "(uint8_t)((i / 5ull) & 3ull)"),
        ("io_resume_bits_nextPc", "uint32_t", "(uint32_t)(0x80001000u + (uint32_t)(i * 4ull))"),
        ("io_resume_bits_activeMask", "uint8_t", "0xfu"),
        ("io_finish_valid", "uint8_t", "(uint8_t)((i >= 240) && ((i % 17ull) == 0))"),
        ("io_finish_bits", "uint8_t", "(uint8_t)((i / 17ull) & 3ull)"),
    ]
    decls = "\n".join(f"    {ty} {name};" for name, ty, _ in inputs)
    stimulus = "\n".join(f"    {name} = {stim};" for name, _, stim in inputs)
    poke = "\n".join(f"    {{p}}{name} = {name};" for name, _, _ in inputs)
    dump_fields = [
        "io_launch_ready",
        "io_issue_valid",
        "io_issue_bits_warpId",
        "io_issue_bits_pc",
        "io_issue_bits_activeMask",
        "io_active",
        "io_blocked",
    ]
    fmt = "%x %x %x %08x %x %x %x"
    args = ", ".join("{p}" + n for n in dump_fields)
    dump = f'    std::printf("{fmt}\\n", {args});\n'
    sink = (
        "(uint64_t){p}io_launch_ready + {p}io_issue_valid + {p}io_issue_bits_warpId + "
        "{p}io_issue_bits_pc + {p}io_active + {p}io_blocked"
    )
    return BenchIO(decls, stimulus, poke, dump, sink, clock="clock")


def _cw(width: int) -> str:
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
    return f"uint64_t"


def _cdecl(name: str, width: int) -> str:
    if width <= 128:
        return f"{_cw(width)} {name}"
    return f"uint64_t {name}[{(width + 63) // 64}]"


def _poke_one(name: str, width: int) -> str:
    if width > 128:
        return f"    memcpy(&{{p}}{name}, {name}, sizeof({name}));"
    return f"    {{p}}{name} = {name};"


def _router_io() -> BenchIO:
    # Kernel opcode=0 commands with delayed completions. Other engines stay idle.
    fields: list[tuple[str, int, str]] = [
        ("reset", 1, "(uint8_t)(i < 4)"),
        ("io_command_valid", 1, "(uint8_t)((i >= 8) && ((i % 8ull) == 0))"),
        ("io_command_bits_commandId", 4, "(uint8_t)((i / 8ull) & 15ull)"),
        ("io_command_bits_opcode", 3, "0u"),
        ("io_command_bits_launch_kernelPc", 32, "(uint32_t)(0x80000000u + (uint32_t)i)"),
        ("io_command_bits_launch_kernargAddress", 32, "(uint32_t)(0x1000u + (uint32_t)i)"),
        ("io_command_bits_launch_gridSize_0", 32, "1u"),
        ("io_command_bits_launch_gridSize_1", 32, "1u"),
        ("io_command_bits_launch_gridSize_2", 32, "1u"),
        ("io_command_bits_launch_localSize_0", 16, "4u"),
        ("io_command_bits_launch_localSize_1", 16, "1u"),
        ("io_command_bits_launch_localSize_2", 16, "1u"),
        ("io_command_bits_waitForDma", 1, "0u"),
        ("io_command_bits_dmaSource", 2, "0u"),
        ("io_command_bits_dmaDescriptorId", 4, "0u"),
        ("io_command_bits_sourceAddress", 32, "0u"),
        ("io_command_bits_destinationAddress", 32, "0u"),
        ("io_command_bits_bytes", 32, "0u"),
        ("io_command_bits_pattern", 32, "0u"),
        ("io_command_bits_widthBytes", 32, "0u"),
        ("io_command_bits_height", 32, "0u"),
        ("io_command_bits_sourceStride", 32, "0u"),
        ("io_command_bits_destinationStride", 32, "0u"),
        ("io_command_bits_waitForEvent", 1, "0u"),
        ("io_command_bits_waitEventId", 4, "0u"),
        ("io_command_bits_waitEventGeneration", 8, "0u"),
        ("io_command_bits_signalEvent", 1, "0u"),
        ("io_command_bits_signalEventId", 4, "0u"),
        ("io_command_bits_signalEventGeneration", 8, "0u"),
        ("io_completion_ready", 1, "1u"),
        ("io_kernel_ready", 1, "1u"),
        ("io_copy_ready", 1, "1u"),
        ("io_fill_ready", 1, "1u"),
        ("io_stridedCopy_ready", 1, "1u"),
        ("io_kernelCompletion_valid", 1, "(uint8_t)((i >= 40) && ((i % 8ull) == 4))"),
        ("io_kernelCompletion_bits_commandId", 4, "(uint8_t)(((i - 32ull) / 8ull) & 15ull)"),
        ("io_kernelCompletion_bits_status", 3, "0u"),
        ("io_kernelCompletion_bits_success", 1, "1u"),
        ("io_copyCompletion_valid", 1, "0u"),
        ("io_copyCompletion_bits_descriptorId", 4, "0u"),
        ("io_copyCompletion_bits_status", 3, "0u"),
        ("io_copyCompletion_bits_success", 1, "0u"),
        ("io_copyCompletion_bits_bytesCopied", 32, "0u"),
        ("io_fillCompletion_valid", 1, "0u"),
        ("io_fillCompletion_bits_descriptorId", 4, "0u"),
        ("io_fillCompletion_bits_status", 3, "0u"),
        ("io_fillCompletion_bits_success", 1, "0u"),
        ("io_fillCompletion_bits_bytesFilled", 32, "0u"),
        ("io_stridedCopyCompletion_valid", 1, "0u"),
        ("io_stridedCopyCompletion_bits_descriptorId", 4, "0u"),
        ("io_stridedCopyCompletion_bits_status", 3, "0u"),
        ("io_stridedCopyCompletion_bits_success", 1, "0u"),
        ("io_stridedCopyCompletion_bits_bytesCopied", 64, "0ull"),
    ]
    decls = "\n".join(f"    {_cw(w)} {n};" for n, w, _ in fields)
    stimulus = "\n".join(f"    {n} = {stim};" for n, _, stim in fields)
    poke = "\n".join(f"    {{p}}{n} = {n};" for n, _, _ in fields)
    dump_fields = [
        "io_command_ready",
        "io_kernel_valid",
        "io_kernel_bits_commandId",
        "io_kernel_bits_launch_kernelPc",
        "io_copy_valid",
        "io_fill_valid",
        "io_stridedCopy_valid",
        "io_completion_valid",
        "io_completion_bits_commandId",
        "io_busy",
        "io_duplicateCommandId",
        "io_kernelCompletion_ready",
    ]
    fmt = " ".join("%x" if "Pc" not in n else "%08x" for n in dump_fields)
    args = ", ".join("{p}" + n for n in dump_fields)
    dump = f'    std::printf("{fmt}\\n", {args});\n'
    sink = (
        "(uint64_t){p}io_command_ready + {p}io_kernel_valid + {p}io_kernel_bits_commandId + "
        "{p}io_kernel_bits_launch_kernelPc + {p}io_completion_valid + {p}io_busy"
    )
    return BenchIO(decls, stimulus, poke, dump, sink, clock="clock")


def _fields_io(
    fields: list[tuple[str, int, str]],
    dump_fields: list[str],
    sink: str,
    dump_hex32: frozenset[str] = frozenset(),
) -> BenchIO:
    decls = "\n".join(f"    {_cdecl(n, w)};" for n, w, _ in fields)
    stim_lines: list[str] = []
    for n, w, stim in fields:
        if w > 128:
            stim_lines.append(f"    memset({n}, 0, sizeof({n}));")
            if stim:
                stim_lines.append(f"    {stim}")
        else:
            stim_lines.append(f"    {n} = {stim};")
    stimulus = "\n".join(stim_lines)
    poke = "\n".join(_poke_one(n, w) for n, w, _ in fields)
    fmt = " ".join("%08x" if n in dump_hex32 else "%x" for n in dump_fields)
    args = ", ".join("{p}" + n for n in dump_fields)
    dump = f'    std::printf("{fmt}\\n", {args});\n'
    return BenchIO(decls, stimulus, poke, dump, sink, clock="clock")


def _shared_io() -> BenchIO:
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_in_valid", 1, "(uint8_t)((i == 8) || ((i >= 32) && ((i % 16ull) == 0)))"),
            ("io_in_bits_warpId", 2, "0u"),
            ("io_in_bits_addresses_0", 32, "0x10000000u"),
            ("io_in_bits_addresses_1", 32, "0x10000004u"),
            ("io_in_bits_addresses_2", 32, "0x10000008u"),
            ("io_in_bits_addresses_3", 32, "0x1000000cu"),
            ("io_in_bits_writeData_0", 32, "0xa0000001u"),
            ("io_in_bits_writeData_1", 32, "0xa0000002u"),
            ("io_in_bits_writeData_2", 32, "0xa0000003u"),
            ("io_in_bits_writeData_3", 32, "0xa0000004u"),
            ("io_in_bits_laneMask", 4, "0xfu"),
            ("io_in_bits_elementSize", 2, "2u"),
            ("io_in_bits_isStore", 1, "(uint8_t)(i == 8)"),
            ("io_out_ready", 1, "1u"),
            ("io_atomicIn_valid", 1, "(uint8_t)((i >= 80) && ((i % 32ull) == 0))"),
            ("io_atomicIn_bits_warpId", 2, "1u"),
            ("io_atomicIn_bits_address", 32, "0x10000000u"),
            ("io_atomicIn_bits_operand", 32, "1u"),
            ("io_atomicIn_bits_operation", 4, "1u"),
            ("io_atomicOut_ready", 1, "1u"),
        ],
        [
            "io_in_ready",
            "io_out_valid",
            "io_out_bits_readData_0",
            "io_out_bits_readData_1",
            "io_out_bits_readData_2",
            "io_out_bits_readData_3",
            "io_out_bits_faultMask",
            "io_out_bits_pageFault",
            "io_atomicIn_ready",
            "io_atomicOut_valid",
            "io_atomicOut_bits_oldValue",
            "io_idle",
        ],
        "(uint64_t){p}io_in_ready + {p}io_out_valid + {p}io_out_bits_readData_0 + "
        "{p}io_atomicOut_valid + {p}io_atomicOut_bits_oldValue + {p}io_idle",
        dump_hex32=frozenset(
            {
                "io_out_bits_readData_0",
                "io_out_bits_readData_1",
                "io_out_bits_readData_2",
                "io_out_bits_readData_3",
                "io_atomicOut_bits_oldValue",
            }
        ),
    )


def _frontend_io() -> BenchIO:
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_launch_valid", 1, "(uint8_t)((i >= 8) && (i < 10))"),
            ("io_launch_bits_warpId", 2, "(uint8_t)((i >= 8) && (i < 10) ? ((i - 8ull) & 3ull) : 0ull)"),
            ("io_launch_bits_startPc", 32, "((i >= 8) && (i < 10) ? (uint32_t)(0x80000000u + (((uint32_t)i - 8u) << 8)) : 0x80000000u)"),
            ("io_launch_bits_activeMask", 4, "0xfu"),
            ("io_fetchRequest_ready", 1, "1u"),
            ("io_fetchResponse_valid", 1, "(uint8_t)((i >= 16) && ((i % 3ull) == 0) && (i < 300))"),
            ("io_fetchResponse_bits_warpId", 2, "(uint8_t)((i >= 16) && ((i % 3ull) == 0) && (i < 300) ? ((i / 3ull) & 1ull) : 0ull)"),
            ("io_fetchResponse_bits_instruction", 32, "((i >= 16) && ((i % 3ull) == 0) && (i < 300) ? (uint32_t)(0x00100093u + (uint32_t)i) : 0x00100093u)"),
            ("io_fetchResponse_bits_accessFault", 1, "0u"),
            ("io_scalarOut_ready", 1, "1u"),
            ("io_fpuOut_ready", 1, "1u"),
            ("io_vectorOut_ready", 1, "1u"),
            ("io_branch_valid", 1, "0u"),
            ("io_branch_bits_warpId", 2, "0u"),
            ("io_branch_bits_pc", 32, "0u"),
            ("io_branch_bits_targetPc", 32, "0u"),
            ("io_branch_bits_fallthroughPc", 32, "0u"),
            ("io_branch_bits_reconvergePc", 32, "0u"),
            ("io_branch_bits_takenMask", 4, "0u"),
            ("io_branch_bits_activeMask", 4, "0u"),
            ("io_scalarRedirect_valid", 1, "0u"),
            ("io_scalarRedirect_bits_warpId", 2, "0u"),
            ("io_scalarRedirect_bits_pc", 32, "0u"),
            ("io_scalarRedirect_bits_activeMask", 4, "0u"),
            ("io_restore_valid", 1, "0u"),
            ("io_restore_bits", 2, "0u"),
            ("io_finish_valid", 1, "(uint8_t)((i >= 320) && ((i % 40ull) == 0))"),
            ("io_finish_bits", 2, "(uint8_t)((i >= 320) && ((i % 40ull) == 0) ? ((i / 40ull) & 1ull) : 0ull)"),
        ],
        [
            "io_launch_ready",
            "io_fetchRequest_valid",
            "io_fetchRequest_bits_warpId",
            "io_fetchRequest_bits_pc",
            "io_fetchResponse_ready",
            "io_scalarOut_valid",
            "io_scalarOut_bits_instruction",
            "io_scalarOut_bits_pc",
            "io_scalarOut_bits_decoded_aluOp",
            "io_fpuOut_valid",
            "io_vectorOut_valid",
            "io_active",
            "io_blocked",
        ],
        "(uint64_t){p}io_launch_ready + {p}io_fetchRequest_valid + {p}io_fetchRequest_bits_pc + "
        "{p}io_scalarOut_valid + {p}io_scalarOut_bits_instruction + {p}io_active + {p}io_blocked",
        dump_hex32=frozenset(
            {
                "io_fetchRequest_bits_pc",
                "io_scalarOut_bits_instruction",
                "io_scalarOut_bits_pc",
            }
        ),
    )


def _icache_io() -> BenchIO:
    fetch = "((i >= 8) && ((i % 8ull) == 0) && (i < 400))"
    refill = "((i >= 12) && ((i % 8ull) == 4) && (i < 404))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_fetch_valid", 1, f"(uint8_t){fetch}"),
            ("io_fetch_bits_warpId", 2, f"(uint8_t)({fetch} ? ((i / 8ull) & 1ull) : 0ull)"),
            (
                "io_fetch_bits_pc",
                32,
                f"({fetch} ? (uint32_t)(0x80000000u + (uint32_t)(((i / 8ull) & 3ull) << 3)) : 0x80000000u)",
            ),
            ("io_response_ready", 1, "1u"),
            ("io_lowerRequest_ready", 1, "1u"),
            ("io_lowerResponse_valid", 1, f"(uint8_t){refill}"),
            (
                "io_lowerResponse_bits_readData",
                64,
                f"({refill} ? (0x00100093ull + (i / 8ull)) : 0x00100093ull)",
            ),
            ("io_lowerResponse_bits_fault", 1, "0u"),
            ("io_lowerResponse_bits_requestId", 1, "0u"),
            ("io_invalidate", 1, "(uint8_t)((i >= 480) && ((i % 200ull) == 0))"),
        ],
        [
            "io_fetch_ready",
            "io_response_valid",
            "io_response_bits_warpId",
            "io_response_bits_instruction",
            "io_response_bits_accessFault",
            "io_lowerRequest_valid",
            "io_lowerRequest_bits_lineAddress",
            "io_lowerRequest_bits_requestId",
            "io_lowerResponse_ready",
        ],
        "(uint64_t){p}io_fetch_ready + {p}io_response_valid + {p}io_response_bits_instruction + "
        "{p}io_lowerRequest_valid + {p}io_lowerRequest_bits_lineAddress",
        dump_hex32=frozenset(
            {
                "io_response_bits_instruction",
                "io_lowerRequest_bits_lineAddress",
            }
        ),
    )


def _frontend_icache_io() -> BenchIO:
    launch = "((i >= 8) && (i < 10))"
    refill = "((i >= 20) && ((i % 8ull) == 4) && (i < 404))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_launch_valid", 1, f"(uint8_t){launch}"),
            ("io_launch_bits_warpId", 2, f"(uint8_t)({launch} ? ((i - 8ull) & 3ull) : 0ull)"),
            (
                "io_launch_bits_startPc",
                32,
                f"({launch} ? (uint32_t)(0x80000000u + (((uint32_t)i - 8u) << 8)) : 0x80000000u)",
            ),
            ("io_launch_bits_activeMask", 4, "0xfu"),
            ("io_scalarOut_ready", 1, "1u"),
            ("io_fpuOut_ready", 1, "1u"),
            ("io_vectorOut_ready", 1, "1u"),
            ("io_branch_valid", 1, "0u"),
            ("io_branch_bits_warpId", 2, "0u"),
            ("io_branch_bits_pc", 32, "0u"),
            ("io_branch_bits_targetPc", 32, "0u"),
            ("io_branch_bits_fallthroughPc", 32, "0u"),
            ("io_branch_bits_reconvergePc", 32, "0u"),
            ("io_branch_bits_activeMask", 4, "0u"),
            ("io_branch_bits_takenMask", 4, "0u"),
            ("io_scalarRedirect_valid", 1, "0u"),
            ("io_scalarRedirect_bits_warpId", 2, "0u"),
            ("io_scalarRedirect_bits_pc", 32, "0u"),
            ("io_scalarRedirect_bits_activeMask", 4, "0u"),
            ("io_restore_valid", 1, "0u"),
            ("io_restore_bits", 2, "0u"),
            ("io_finish_valid", 1, "(uint8_t)((i >= 320) && ((i % 40ull) == 0))"),
            ("io_finish_bits", 2, "(uint8_t)((i >= 320) && ((i % 40ull) == 0) ? ((i / 40ull) & 1ull) : 0ull)"),
            ("io_lowerRequest_ready", 1, "1u"),
            ("io_lowerResponse_valid", 1, f"(uint8_t){refill}"),
            (
                "io_lowerResponse_bits_readData",
                64,
                f"({refill} ? (0x00100093ull + (i / 8ull)) : 0x00100093ull)",
            ),
            ("io_lowerResponse_bits_fault", 1, "0u"),
            ("io_lowerResponse_bits_requestId", 1, "0u"),
            ("io_invalidate", 1, "(uint8_t)((i >= 480) && ((i % 200ull) == 0))"),
        ],
        [
            "io_launch_ready",
            "io_scalarOut_valid",
            "io_scalarOut_bits_instruction",
            "io_scalarOut_bits_pc",
            "io_scalarOut_bits_warpId",
            "io_fpuOut_valid",
            "io_vectorOut_valid",
            "io_lowerRequest_valid",
            "io_lowerRequest_bits_lineAddress",
            "io_lowerRequest_bits_requestId",
            "io_lowerResponse_ready",
            "io_active",
            "io_blocked",
        ],
        "(uint64_t){p}io_launch_ready + {p}io_scalarOut_valid + {p}io_scalarOut_bits_instruction + "
        "{p}io_lowerRequest_valid + {p}io_lowerRequest_bits_lineAddress + {p}io_active + {p}io_blocked",
        dump_hex32=frozenset(
            {
                "io_scalarOut_bits_instruction",
                "io_scalarOut_bits_pc",
                "io_lowerRequest_bits_lineAddress",
            }
        ),
    )


def _vector_io() -> BenchIO:
    cfg = "((i >= 16) && (i < 18))"
    alu = "((i >= 32) && ((i % 16ull) == 0) && (i < 280))"
    issue = f"({cfg} || {alu})"
    init1 = "((i >= 8) && (i < 9))"
    init2 = "((i >= 9) && (i < 10))"
    init = f"({init1} || {init2})"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_in_valid", 1, f"(uint8_t){issue}"),
            (
                "io_in_bits_instruction",
                32,
                f"({cfg} ? 0x010070d7u : 0x022081d7u)",
            ),
            ("io_in_bits_pc", 32, f"({issue} ? (uint32_t)(0x80000000u + (uint32_t)i) : 0x80000000u)"),
            ("io_in_bits_warpId", 2, "0u"),
            ("io_in_bits_activeMask", 4, "0xfu"),
            ("io_in_bits_decoded_recognized", 1, "1u"),
            ("io_in_bits_decoded_valid", 1, "1u"),
            ("io_in_bits_decoded_unit", 4, f"({cfg} ? 6u : 1u)"),
            ("io_in_bits_decoded_funct6", 6, "0u"),
            ("io_in_bits_decoded_operandType", 3, f"({cfg} ? 7u : 0u)"),
            ("io_in_bits_decoded_vm", 1, f"({cfg} ? 0u : 1u)"),
            ("io_in_bits_decoded_nf", 3, "0u"),
            ("io_in_bits_decoded_mop", 2, "0u"),
            ("io_in_bits_decoded_elementWidth", 3, f"({cfg} ? 7u : 0u)"),
            ("io_in_bits_decoded_readsVs1", 1, f"({cfg} ? 0u : 1u)"),
            ("io_in_bits_decoded_readsVs2", 1, f"({cfg} ? 0u : 1u)"),
            ("io_in_bits_decoded_readsScalar", 1, "0u"),
            ("io_in_bits_decoded_readsFloat", 1, "0u"),
            ("io_in_bits_decoded_writesVd", 1, f"({cfg} ? 0u : 1u)"),
            ("io_in_bits_decoded_memoryRead", 1, "0u"),
            ("io_in_bits_decoded_memoryWrite", 1, "0u"),
            ("io_in_bits_decoded_configure", 1, f"(uint8_t){cfg}"),
            ("io_scalarRs1Data", 32, "0u"),
            ("io_scalarRs2Data", 32, "0u"),
            ("io_scalarFpData", 32, "0u"),
            ("io_scalarFpBusy_0", 32, "0u"),
            ("io_scalarFpBusy_1", 32, "0u"),
            ("io_scalarFpBusy_2", 32, "0u"),
            ("io_scalarFpBusy_3", 32, "0u"),
            ("io_scalarFlagsWrite_valid", 1, "0u"),
            ("io_scalarFlagsWrite_bits_warpId", 2, "0u"),
            ("io_scalarFlagsWrite_bits_flags", 5, "0u"),
            ("io_scalarReserve_ready", 1, "1u"),
            ("io_initialize_valid", 1, f"(uint8_t){init}"),
            ("io_initialize_bits_warpId", 2, "0u"),
            ("io_initialize_bits_vd", 5, f"({init1} ? 1u : 2u)"),
            ("io_initialize_bits_data_0", 32, f"({init1} ? 0x10u : 1u)"),
            ("io_initialize_bits_data_1", 32, f"({init1} ? 0x20u : 2u)"),
            ("io_initialize_bits_data_2", 32, f"({init1} ? 0x30u : 3u)"),
            ("io_initialize_bits_data_3", 32, f"({init1} ? 0x40u : 4u)"),
            ("io_scalarWriteback_ready", 1, "1u"),
            ("io_redirect_ready", 1, "1u"),
            ("io_memoryRequest_ready", 1, "1u"),
            ("io_memoryResponse_valid", 1, "0u"),
            ("io_memoryResponse_bits_readData_0", 32, "0u"),
            ("io_memoryResponse_bits_readData_1", 32, "0u"),
            ("io_memoryResponse_bits_readData_2", 32, "0u"),
            ("io_memoryResponse_bits_readData_3", 32, "0u"),
            ("io_memoryResponse_bits_faultMask", 4, "0u"),
            ("io_memoryResponse_bits_pageFault", 1, "0u"),
            ("io_memoryFault_ready", 1, "1u"),
            ("io_unimplemented_ready", 1, "1u"),
            ("io_texSample_ready", 1, "1u"),
            ("io_texCommit_valid", 1, "0u"),
            ("io_texCommit_bits_writeback_warpId", 2, "0u"),
            ("io_texCommit_bits_writeback_vd", 5, "0u"),
            ("io_texCommit_bits_writeback_data_0", 32, "0u"),
            ("io_texCommit_bits_writeback_data_1", 32, "0u"),
            ("io_texCommit_bits_writeback_data_2", 32, "0u"),
            ("io_texCommit_bits_writeback_data_3", 32, "0u"),
            ("io_texCommit_bits_saturated", 1, "0u"),
            ("io_texCommit_bits_writesVd", 1, "0u"),
            ("io_texCommit_bits_flags", 5, "0u"),
            ("io_texCommit_bits_writesFlags", 1, "0u"),
            ("io_texCommit_bits_pc", 32, "0u"),
            ("io_texCommit_bits_warpActiveMask", 4, "0u"),
        ],
        [
            "io_in_ready",
            "io_initialize_ready",
            "io_scalarWriteback_valid",
            "io_redirect_valid",
            "io_redirect_bits_pc",
            "io_committedVectorWriteback_valid",
            "io_committedVectorWriteback_bits_vd",
            "io_committedVectorWriteback_bits_data_0",
            "io_rawHazard",
            "io_wawHazard",
        ],
        "(uint64_t){p}io_in_ready + {p}io_redirect_valid + {p}io_redirect_bits_pc + "
        "{p}io_committedVectorWriteback_valid + {p}io_committedVectorWriteback_bits_data_0 + "
        "{p}io_rawHazard",
        dump_hex32=frozenset(
            {
                "io_redirect_bits_pc",
                "io_committedVectorWriteback_bits_data_0",
            }
        ),
    )


def _fma_lane_io() -> BenchIO:
    # Two sparse ADD pulses so skip cannot reuse a previous 1.0+2.0 result.
    # Operands are zero while idle; that matches FMA capture during issue bubbles.
    pulse1 = "((i >= 16) && (i < 17))"
    pulse2 = "((i >= 80) && (i < 81))"
    pulse = f"({pulse1} || {pulse2})"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_in_valid", 1, f"(uint8_t){pulse}"),
            ("io_in_bits_operandA", 32, "0u"),
            ("io_in_bits_operandB", 32, f"({pulse1} ? 0x3f800000u : ({pulse2} ? 0x40800000u : 0u))"),
            ("io_in_bits_operandC", 32, f"({pulse1} ? 0x40000000u : ({pulse2} ? 0x40a00000u : 0u))"),
            ("io_in_bits_roundingMode", 3, "0u"),
            ("io_in_bits_operation", 5, "2u"),
            ("io_in_bits_operationModifier", 1, "0u"),
            ("io_in_bits_tag", 16, f"({pulse1} ? 1u : ({pulse2} ? 2u : 0u))"),
            ("io_out_ready", 1, "1u"),
            ("io_flush", 1, "0u"),
        ],
        [
            "io_in_ready",
            "io_out_valid",
            "io_out_bits_result",
            "io_out_bits_tag",
        ],
        "(uint64_t){p}io_in_ready + {p}io_out_valid + {p}io_out_bits_result + {p}io_out_bits_tag",
        dump_hex32=frozenset({"io_out_bits_result"}),
    )


def _fpu_io() -> BenchIO:
    # fadd.s f3, f1, f2 after initializing f1=1.0 and f2=2.0. Memory stays idle;
    # the 512-bit cache-line response is left at zero and not poked.
    issue = "((i >= 16) && ((i % 16ull) == 0) && (i < 280))"
    init1 = "((i >= 8) && (i < 9))"
    init2 = "((i >= 9) && (i < 10))"
    init = f"({init1} || {init2})"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_in_valid", 1, f"(uint8_t){issue}"),
            ("io_in_bits_instruction", 32, "0x002081d3u"),
            ("io_in_bits_pc", 32, f"({issue} ? (uint32_t)(0x80000000u + (uint32_t)i) : 0x80000000u)"),
            ("io_in_bits_warpId", 2, "0u"),
            ("io_in_bits_activeMask", 4, "0xfu"),
            ("io_in_bits_decoded_recognized", 1, "1u"),
            ("io_in_bits_decoded_valid", 1, "1u"),
            ("io_in_bits_decoded_unit", 3, "1u"),
            ("io_in_bits_decoded_format", 2, "0u"),
            ("io_in_bits_decoded_funct5", 5, "0u"),
            ("io_in_bits_decoded_rm", 3, "0u"),
            ("io_in_bits_decoded_readsRs1", 1, "1u"),
            ("io_in_bits_decoded_readsRs2", 1, "1u"),
            ("io_in_bits_decoded_readsRs3", 1, "0u"),
            ("io_in_bits_decoded_writesFp", 1, "1u"),
            ("io_in_bits_decoded_writesInteger", 1, "0u"),
            ("io_in_bits_decoded_memoryRead", 1, "0u"),
            ("io_in_bits_decoded_memoryWrite", 1, "0u"),
            ("io_in_bits_decoded_setsFlags", 1, "1u"),
            ("io_redirect_ready", 1, "1u"),
            ("io_initialize_valid", 1, f"(uint8_t){init}"),
            ("io_initialize_bits_warpId", 2, "0u"),
            ("io_initialize_bits_rd", 5, f"({init1} ? 1u : 2u)"),
            ("io_initialize_bits_data", 32, f"({init1} ? 0x3f800000u : 0x40000000u)"),
            ("io_unimplemented_ready", 1, "1u"),
            ("io_scalarReserve_ready", 1, "1u"),
            ("io_scalarWriteback_ready", 1, "1u"),
            ("io_scalarRs1Data", 32, "0u"),
            ("io_frm_0", 3, "0u"),
            ("io_frm_1", 3, "0u"),
            ("io_frm_2", 3, "0u"),
            ("io_frm_3", 3, "0u"),
            ("io_fvfRead_warpId", 2, "0u"),
            ("io_fvfRead_rs1", 5, "0u"),
            ("io_fvfRead_rs2", 5, "0u"),
            ("io_fvfRead_rs3", 5, "0u"),
            ("io_memoryRequest_ready", 1, "1u"),
            ("io_memoryResponse_valid", 1, "0u"),
            ("io_memoryResponse_bits_fault", 1, "0u"),
            ("io_memoryResponse_bits_pageFault", 1, "0u"),
            ("io_memoryFault_ready", 1, "1u"),
            ("io_flush", 1, "0u"),
        ],
        [
            "io_in_ready",
            "io_initialize_ready",
            "io_redirect_valid",
            "io_redirect_bits_pc",
            "io_committedWriteback_valid",
            "io_committedWriteback_bits_rd",
            "io_committedWriteback_bits_data",
            "io_rawHazard",
            "io_wawHazard",
        ],
        "(uint64_t){p}io_in_ready + {p}io_redirect_valid + {p}io_redirect_bits_pc + "
        "{p}io_committedWriteback_valid + {p}io_committedWriteback_bits_data + {p}io_rawHazard",
        dump_hex32=frozenset(
            {
                "io_redirect_bits_pc",
                "io_committedWriteback_bits_data",
            }
        ),
    )


def _frontend_scalar_io() -> BenchIO:
    # Closed fetch/decode/ALU/redirect loop. IMEM refill returns addi x1, x1, 1
    # in both 32-bit words of the 8-byte line. Warps stay live (no finish).
    launch = "((i >= 8) && (i < 10))"
    init = "((i >= 6) && (i < 7))"
    refill = "((i >= 20) && ((i % 8ull) == 4) && (i < 404))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_initialize_valid", 1, f"(uint8_t){init}"),
            ("io_initialize_bits_warpId", 2, "0u"),
            ("io_initialize_bits_rd", 5, "1u"),
            ("io_initialize_bits_data", 32, "0x10u"),
            ("io_launch_valid", 1, f"(uint8_t){launch}"),
            ("io_launch_bits_warpId", 2, f"(uint8_t)({launch} ? ((i - 8ull) & 3ull) : 0ull)"),
            (
                "io_launch_bits_startPc",
                32,
                f"({launch} ? (uint32_t)(0x80000000u + (((uint32_t)i - 8u) << 8)) : 0x80000000u)",
            ),
            ("io_launch_bits_activeMask", 4, "0xfu"),
            ("io_finish_valid", 1, "0u"),
            ("io_finish_bits", 2, "0u"),
            ("io_lowerRequest_ready", 1, "1u"),
            ("io_lowerResponse_valid", 1, f"(uint8_t){refill}"),
            ("io_lowerResponse_bits_readData", 64, "0x0010809300108093ull"),
            ("io_lowerResponse_bits_fault", 1, "0u"),
            ("io_lowerResponse_bits_requestId", 1, "0u"),
            ("io_invalidate", 1, "(uint8_t)((i >= 480) && ((i % 200ull) == 0))"),
        ],
        [
            "io_launch_ready",
            "io_initialize_ready",
            "io_committedWriteback_valid",
            "io_committedWriteback_bits_rd",
            "io_committedWriteback_bits_data",
            "io_lowerRequest_valid",
            "io_lowerRequest_bits_lineAddress",
            "io_lowerResponse_ready",
            "io_active",
            "io_blocked",
            "io_rawHazard",
        ],
        "(uint64_t){p}io_launch_ready + {p}io_committedWriteback_valid + "
        "{p}io_committedWriteback_bits_data + {p}io_lowerRequest_valid + "
        "{p}io_lowerRequest_bits_lineAddress + {p}io_active + {p}io_blocked",
        dump_hex32=frozenset(
            {
                "io_committedWriteback_bits_data",
                "io_lowerRequest_bits_lineAddress",
            }
        ),
    )


def _gpu_io() -> BenchIO:
    # One workgroup / one warp. Mixed IMEM: addi x1, x1, 1 then fadd.s f3, f1, f2.
    # Finish the resident warp after the burst so 1e6-cycle perf is mostly idle.
    kernel = "((i >= 8) && (i < 40))"
    init_f1 = "((i >= 6) && (i < 7))"
    init_f2 = "((i >= 7) && (i < 8))"
    init_f = f"({init_f1} || {init_f2})"
    refill = "((i >= 20) && ((i % 8ull) == 4) && (i < 404))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_fpuInitialize_valid", 1, f"(uint8_t){init_f}"),
            ("io_fpuInitialize_bits_warpId", 2, "0u"),
            ("io_fpuInitialize_bits_rd", 5, f"({init_f1} ? 1u : 2u)"),
            ("io_fpuInitialize_bits_data", 32, f"({init_f1} ? 0x3f800000u : 0x40000000u)"),
            ("io_kernel_valid", 1, f"(uint8_t){kernel}"),
            ("io_kernel_bits_kernelPc", 32, "0x80000000u"),
            ("io_kernel_bits_kernargAddress", 32, "0u"),
            ("io_kernel_bits_gridSize_0", 32, "1u"),
            ("io_kernel_bits_gridSize_1", 32, "1u"),
            ("io_kernel_bits_gridSize_2", 32, "1u"),
            ("io_kernel_bits_localSize_0", 16, "4u"),
            ("io_kernel_bits_localSize_1", 16, "1u"),
            ("io_kernel_bits_localSize_2", 16, "1u"),
            ("io_completion_ready", 1, "1u"),
            ("io_finish_valid", 1, "(uint8_t)((i >= 320) && (i < 321))"),
            ("io_finish_bits", 2, "0u"),
            ("io_lowerRequest_ready", 1, "1u"),
            ("io_lowerResponse_valid", 1, f"(uint8_t){refill}"),
            ("io_lowerResponse_bits_readData", 64, "0x002081d300108093ull"),
            ("io_lowerResponse_bits_fault", 1, "0u"),
            ("io_lowerResponse_bits_requestId", 1, f"({refill} ? (uint8_t)((i / 8ull) & 1ull) : 0ull)"),
            ("io_invalidate", 1, "(uint8_t)((i >= 200) && (i < 280) && ((i % 40ull) == 0))"),
        ],
        [
            "io_kernel_ready",
            "io_completion_valid",
            "io_completion_bits_success",
            "io_fpuInitialize_ready",
            "io_committedWriteback_valid",
            "io_committedWriteback_bits_rd",
            "io_committedWriteback_bits_data",
            "io_committedVectorWriteback_valid",
            "io_committedVectorWriteback_bits_vd",
            "io_committedVectorWriteback_bits_data_0",
            "io_committedFpuWriteback_valid",
            "io_committedFpuWriteback_bits_rd",
            "io_committedFpuWriteback_bits_data",
            "io_lowerRequest_valid",
            "io_lowerRequest_bits_lineAddress",
            "io_active",
            "io_blocked",
            "io_barrierWaiting",
        ],
        "(uint64_t){p}io_kernel_ready + {p}io_completion_valid + "
        "{p}io_committedWriteback_valid + {p}io_committedWriteback_bits_data + "
        "{p}io_committedVectorWriteback_valid + {p}io_committedVectorWriteback_bits_data_0 + "
        "{p}io_committedFpuWriteback_valid + {p}io_committedFpuWriteback_bits_data + "
        "{p}io_lowerRequest_valid + {p}io_lowerRequest_bits_lineAddress + "
        "{p}io_active + {p}io_blocked",
        dump_hex32=frozenset(
            {
                "io_committedWriteback_bits_data",
                "io_committedVectorWriteback_bits_data_0",
                "io_committedFpuWriteback_bits_data",
                "io_lowerRequest_bits_lineAddress",
            }
        ),
    )


def _gpu_host_axi_io() -> BenchIO:
    # QEMU/ARTI control top. Tick `clock`; inner reset is !s_axi_aresetn.
    # One single-beat AXI read of the ID register at 0x00 after reset.
    # Untouched mem-port inputs stay 0 (idle backends).
    ar = "((i >= 8) && (i == 16ull))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 8)"),
            ("io_s_axi_aresetn", 1, "(uint8_t)(i >= 8)"),
            ("io_s_axi_araddr", 32, "0u"),
            ("io_s_axi_arlen", 8, "0u"),
            ("io_s_axi_arsize", 3, "2u"),
            ("io_s_axi_arburst", 2, "0u"),
            ("io_s_axi_arvalid", 1, f"(uint8_t){ar}"),
            ("io_s_axi_rready", 1, "1u"),
            ("io_s_axi_bready", 1, "1u"),
        ],
        [
            "io_s_axi_arready",
            "io_s_axi_rvalid",
            "io_s_axi_rdata",
            "io_s_axi_rresp",
            "io_s_axi_rlast",
            "io_s_axi_awready",
            "io_m_irq",
        ],
        "(uint64_t){p}io_s_axi_arready + {p}io_s_axi_rvalid + {p}io_s_axi_rdata + "
        "{p}io_s_axi_rresp + {p}io_s_axi_rlast + {p}io_s_axi_awready + {p}io_m_irq",
        dump_hex32=frozenset({"io_s_axi_rdata"}),
    )


def _gpu_system_io() -> BenchIO:
    # One CU behind L2. IMEM line is addi, fadd.s, cease so the kernel finishes
    # and the 1e6-cycle run is mostly idle. DRAM responses are delayed one
    # cycle and echo the interconnect transaction id.
    kernel = "((i >= 8) && (i < 40))"
    base = _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_command_valid", 1, f"(uint8_t){kernel}"),
            ("io_command_bits_commandId", 4, "1u"),
            ("io_command_bits_launch_kernelPc", 32, "0x80000000u"),
            ("io_command_bits_launch_kernargAddress", 32, "0u"),
            ("io_command_bits_launch_gridSize_0", 32, "1u"),
            ("io_command_bits_launch_gridSize_1", 32, "1u"),
            ("io_command_bits_launch_gridSize_2", 32, "1u"),
            ("io_command_bits_launch_localSize_0", 16, "4u"),
            ("io_command_bits_launch_localSize_1", 16, "1u"),
            ("io_command_bits_launch_localSize_2", 16, "1u"),
            ("io_command_bits_waitForDma", 1, "0u"),
            ("io_command_bits_dmaSource", 2, "0u"),
            ("io_command_bits_dmaDescriptorId", 4, "0u"),
            ("io_commandCompletion_ready", 1, "1u"),
            ("io_copyDescriptor_valid", 1, "0u"),
            ("io_copyCompletion_ready", 1, "1u"),
            ("io_fillDescriptor_valid", 1, "0u"),
            ("io_fillCompletion_ready", 1, "1u"),
            ("io_stridedCopyDescriptor_valid", 1, "0u"),
            ("io_stridedCopyCompletion_ready", 1, "1u"),
            ("io_gpuCommand_valid", 1, "0u"),
            ("io_gpuCompletion_ready", 1, "1u"),
            ("io_graphicsHostRequest_valid", 1, "0u"),
            ("io_graphicsHostResponse_ready", 1, "1u"),
            ("io_graphicsShaderRequest_valid", 1, "0u"),
            ("io_graphicsShaderResponse_ready", 1, "1u"),
            ("io_graphicsShaderL1Invalidate_ready", 1, "1u"),
            ("io_graphicsShaderL1InvalidateDone_valid", 1, "0u"),
            ("io_graphicsShaderAtomicRequest_valid", 1, "0u"),
            ("io_graphicsShaderAtomicResponse_ready", 1, "1u"),
            ("io_memoryRequest_ready", 1, "1u"),
            ("io_memoryResponse_valid", 1, "pend_v"),
            ("io_memoryResponse_bits_fault", 1, "0u"),
            ("io_memoryResponse_bits_transactionId", 5, "pend_id"),
            ("io_invalidateInstructionCache", 1, "0u"),
            ("io_instructionSatp", 32, "0u"),
            ("io_instructionTlbFlush_valid", 1, "0u"),
            ("io_vectorSatp", 32, "0u"),
            ("io_vectorTlbFlush_valid", 1, "0u"),
            ("io_fpu_0_ready", 1, "1u"),
            ("io_vector_0_ready", 1, "1u"),
            ("io_scalarMemory_0_ready", 1, "1u"),
            ("io_unsupportedSystem_0_ready", 1, "1u"),
            ("io_trap_0_ready", 1, "1u"),
            ("io_simtBranch_0_valid", 1, "0u"),
            ("io_clearPerformanceCounters", 1, "0u"),
        ],
        [
            "io_command_ready",
            "io_commandCompletion_valid",
            "io_commandCompletion_bits_success",
            "io_committedWriteback_0_valid",
            "io_committedWriteback_0_bits_rd",
            "io_committedWriteback_0_bits_data",
            "io_committedVectorWriteback_0_valid",
            "io_committedFpuWriteback_0_valid",
            "io_committedFpuWriteback_0_bits_data",
            "io_memoryRequest_valid",
            "io_memoryRequest_bits_address",
            "io_memoryRequest_bits_transactionId",
            "io_activeWarps_0",
            "io_busyComputeUnits",
        ],
        "(uint64_t){p}io_command_ready + {p}io_commandCompletion_valid + "
        "{p}io_committedWriteback_0_valid + {p}io_committedWriteback_0_bits_data + "
        "{p}io_committedFpuWriteback_0_valid + {p}io_committedFpuWriteback_0_bits_data + "
        "{p}io_memoryRequest_valid + {p}io_memoryRequest_bits_address + "
        "{p}io_activeWarps_0 + {p}io_busyComputeUnits",
        dump_hex32=frozenset(
            {
                "io_committedWriteback_0_bits_data",
                "io_committedFpuWriteback_0_bits_data",
                "io_memoryRequest_bits_address",
            }
        ),
    )
    decls = (
        base.decls
        + "\n    uint64_t io_memoryResponse_bits_readData[8];"
    )
    stimulus = (
        base.stimulus
        + """
    memset(io_memoryResponse_bits_readData, 0, sizeof(io_memoryResponse_bits_readData));
    if (pend_v) {
      io_memoryResponse_bits_readData[0] = 0x002081d300108093ull;
      io_memoryResponse_bits_readData[1] = 0x30500073ull;
    }"""
    )
    poke = (
        base.poke
        + "\n    memcpy(&{p}io_memoryResponse_bits_readData, io_memoryResponse_bits_readData, sizeof(io_memoryResponse_bits_readData));"
    )
    return BenchIO(
        decls=decls,
        stimulus=stimulus,
        poke=poke,
        dump=base.dump,
        sink=base.sink,
        clock="clock",
        prelude=(
            "  static uint8_t pend_v = 0;\n"
            "  static uint8_t pend_id = 0;\n"
        ),
        tail=(
            "    pend_v = {p}io_memoryRequest_valid;\n"
            "    pend_id = (uint8_t){p}io_memoryRequest_bits_transactionId;\n"
        ),
    )


def _frontend_scalar_fpu_io() -> BenchIO:
    # Mixed IMEM line: addi x1, x1, 1 then fadd.s f3, f1, f2. Finish warps after
    # the burst so the 1e6-cycle perf run is mostly idle.
    launch = "((i >= 8) && (i < 10))"
    init_s = "((i >= 6) && (i < 7))"
    init_f1 = "((i >= 6) && (i < 7))"
    init_f2 = "((i >= 7) && (i < 8))"
    init_f = f"({init_f1} || {init_f2})"
    refill = "((i >= 20) && ((i % 8ull) == 4) && (i < 404))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_initialize_valid", 1, f"(uint8_t){init_s}"),
            ("io_initialize_bits_warpId", 2, "0u"),
            ("io_initialize_bits_rd", 5, "1u"),
            ("io_initialize_bits_data", 32, "0x10u"),
            ("io_fpuInitialize_valid", 1, f"(uint8_t){init_f}"),
            ("io_fpuInitialize_bits_warpId", 2, "0u"),
            ("io_fpuInitialize_bits_rd", 5, f"({init_f1} ? 1u : 2u)"),
            ("io_fpuInitialize_bits_data", 32, f"({init_f1} ? 0x3f800000u : 0x40000000u)"),
            ("io_launch_valid", 1, f"(uint8_t){launch}"),
            ("io_launch_bits_warpId", 2, f"(uint8_t)({launch} ? ((i - 8ull) & 3ull) : 0ull)"),
            (
                "io_launch_bits_startPc",
                32,
                f"({launch} ? (uint32_t)(0x80000000u + (((uint32_t)i - 8u) << 8)) : 0x80000000u)",
            ),
            ("io_launch_bits_activeMask", 4, "0xfu"),
            ("io_finish_valid", 1, "(uint8_t)((i >= 320) && (i < 400) && ((i % 40ull) == 0))"),
            ("io_finish_bits", 2, "(uint8_t)((i >= 320) && (i < 400) && ((i % 40ull) == 0) ? ((i / 40ull) & 1ull) : 0ull)"),
            ("io_lowerRequest_ready", 1, "1u"),
            ("io_lowerResponse_valid", 1, f"(uint8_t){refill}"),
            ("io_lowerResponse_bits_readData", 64, "0x002081d300108093ull"),
            ("io_lowerResponse_bits_fault", 1, "0u"),
            ("io_lowerResponse_bits_requestId", 1, f"({refill} ? (uint8_t)((i / 8ull) & 1ull) : 0ull)"),
            ("io_invalidate", 1, "(uint8_t)((i >= 200) && (i < 280) && ((i % 40ull) == 0))"),
        ],
        [
            "io_launch_ready",
            "io_initialize_ready",
            "io_fpuInitialize_ready",
            "io_committedWriteback_valid",
            "io_committedWriteback_bits_rd",
            "io_committedWriteback_bits_data",
            "io_committedFpuWriteback_valid",
            "io_committedFpuWriteback_bits_rd",
            "io_committedFpuWriteback_bits_data",
            "io_lowerRequest_valid",
            "io_lowerRequest_bits_lineAddress",
            "io_active",
            "io_blocked",
        ],
        "(uint64_t){p}io_launch_ready + {p}io_committedWriteback_valid + "
        "{p}io_committedWriteback_bits_data + {p}io_committedFpuWriteback_valid + "
        "{p}io_committedFpuWriteback_bits_data + {p}io_lowerRequest_valid + "
        "{p}io_lowerRequest_bits_lineAddress + {p}io_active + {p}io_blocked",
        dump_hex32=frozenset(
            {
                "io_committedWriteback_bits_data",
                "io_committedFpuWriteback_bits_data",
                "io_lowerRequest_bits_lineAddress",
            }
        ),
    )


def _scalar_io() -> BenchIO:
    issue = "((i >= 16) && ((i % 8ull) == 0) && (i < 280))"
    init = "((i >= 8) && (i < 10))"
    return _fields_io(
        [
            ("reset", 1, "(uint8_t)(i < 4)"),
            ("io_in_valid", 1, f"(uint8_t){issue}"),
            ("io_in_bits_instruction", 32, "0x00108093u"),
            ("io_in_bits_pc", 32, f"({issue} ? (uint32_t)(0x80000000u + (uint32_t)i) : 0x80000000u)"),
            ("io_in_bits_warpId", 2, "0u"),
            ("io_in_bits_activeMask", 4, "0xfu"),
            ("io_in_bits_instructionAccessFault", 1, "0u"),
            ("io_in_bits_executionType", 3, "1u"),
            ("io_in_bits_illegalInstruction", 1, "0u"),
            ("io_in_bits_decoded_legal", 1, "1u"),
            ("io_in_bits_decoded_rs1", 5, "1u"),
            ("io_in_bits_decoded_rs2", 5, "0u"),
            ("io_in_bits_decoded_rd", 5, "1u"),
            ("io_in_bits_decoded_immediate", 32, "1u"),
            ("io_in_bits_decoded_aluOp", 4, "0u"),
            ("io_in_bits_decoded_branchOp", 3, "0u"),
            ("io_in_bits_decoded_useImmediate", 1, "1u"),
            ("io_in_bits_decoded_usePc", 1, "0u"),
            ("io_in_bits_decoded_useRs1", 1, "1u"),
            ("io_in_bits_decoded_useRs2", 1, "0u"),
            ("io_in_bits_decoded_writeRd", 1, "1u"),
            ("io_in_bits_decoded_memoryRead", 1, "0u"),
            ("io_in_bits_decoded_memoryWrite", 1, "0u"),
            ("io_in_bits_decoded_jump", 1, "0u"),
            ("io_in_bits_decoded_multiply", 1, "0u"),
            ("io_in_bits_decoded_divide", 1, "0u"),
            ("io_in_bits_decoded_atomic", 1, "0u"),
            ("io_in_bits_decoded_atomicOp", 4, "0u"),
            ("io_in_bits_decoded_csr", 1, "0u"),
            ("io_in_bits_decoded_system", 1, "0u"),
            ("io_in_bits_decoded_fence", 1, "0u"),
            ("io_in_bits_decoded_barrier", 1, "0u"),
            ("io_in_bits_decoded_texSample", 1, "0u"),
            ("io_in_bits_decoded_vectorBranch", 1, "0u"),
            ("io_in_bits_decoded_join", 1, "0u"),
            ("io_in_bits_decoded_cease", 1, "0u"),
            ("io_initialize_valid", 1, f"(uint8_t){init}"),
            ("io_initialize_bits_warpId", 2, "0u"),
            ("io_initialize_bits_rd", 5, "1u"),
            ("io_initialize_bits_data", 32, "0x10u"),
            ("io_redirect_ready", 1, "1u"),
            ("io_externalWriteback_valid", 1, "0u"),
            ("io_externalWriteback_bits_warpId", 2, "0u"),
            ("io_externalWriteback_bits_rd", 5, "0u"),
            ("io_externalWriteback_bits_data", 32, "0u"),
            ("io_externalReserve_valid", 1, "0u"),
            ("io_externalReserve_bits_warpId", 2, "0u"),
            ("io_externalReserve_bits_rs1", 5, "0u"),
            ("io_externalReserve_bits_rs2", 5, "0u"),
            ("io_externalReserve_bits_rd", 5, "0u"),
            ("io_externalReserve_bits_useRs1", 1, "0u"),
            ("io_externalReserve_bits_useRs2", 1, "0u"),
            ("io_externalReserve_bits_writeRd", 1, "0u"),
            ("io_cacheRequest_ready", 1, "1u"),
            ("io_cacheResponse_valid", 1, "0u"),
            ("io_cacheResponse_bits_fault", 1, "0u"),
            ("io_cacheResponse_bits_pageFault", 1, "0u"),
            ("io_memoryFault_ready", 1, "1u"),
            ("io_sharedAtomicRequest_ready", 1, "1u"),
            ("io_sharedAtomicResponse_valid", 1, "0u"),
            ("io_sharedAtomicResponse_bits_warpId", 2, "0u"),
            ("io_sharedAtomicResponse_bits_oldValue", 32, "0u"),
            ("io_sharedAtomicResponse_bits_fault", 1, "0u"),
            ("io_globalAtomicRequest_ready", 1, "1u"),
            ("io_globalAtomicResponse_valid", 1, "0u"),
            ("io_globalAtomicResponse_bits_warpId", 2, "0u"),
            ("io_globalAtomicResponse_bits_oldValue", 32, "0u"),
            ("io_globalAtomicResponse_bits_fault", 1, "0u"),
            ("io_memory_ready", 1, "1u"),
            ("io_system_ready", 1, "1u"),
            ("io_texCommit_valid", 1, "0u"),
            ("io_texCommit_bits_warpId", 2, "0u"),
            ("io_texCommit_bits_nextPc", 32, "0u"),
            ("io_texCommit_bits_activeMask", 4, "0u"),
            ("io_texCommit_bits_writeRd", 1, "0u"),
            ("io_texCommit_bits_rd", 5, "0u"),
            ("io_texCommit_bits_data", 32, "0u"),
            ("io_trap_ready", 1, "1u"),
        ],
        [
            "io_in_ready",
            "io_initialize_ready",
            "io_redirect_valid",
            "io_redirect_bits_pc",
            "io_committedWriteback_valid",
            "io_committedWriteback_bits_rd",
            "io_committedWriteback_bits_data",
            "io_rawHazard",
            "io_wawHazard",
        ],
        "(uint64_t){p}io_in_ready + {p}io_redirect_valid + {p}io_redirect_bits_pc + "
        "{p}io_committedWriteback_valid + {p}io_committedWriteback_bits_data + {p}io_rawHazard",
        dump_hex32=frozenset(
            {
                "io_redirect_bits_pc",
                "io_committedWriteback_bits_data",
            }
        ),
    )


IOS = {
    "gated_pipe": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = (uint8_t)((i >= 2) && ((i % 64ull) == 0));
    din = (uint32_t)i;"""
    ),
    "sticky_input": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = 1;
    din = (uint32_t)(i / 64ull);"""
    ),
    "busy_alu": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = 1;
    din = (uint32_t)i;"""
    ),
    "counter": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = (uint8_t)((i % 3ull) == 0);
    din = (uint32_t)i;"""
    ),
    "cmp_acc": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = (uint8_t)((i % 5ull) == 0);
    din = (uint32_t)((i % 17ull) == 0 ? 0 : (i * 3ull));"""
    ),
    "hier_pipe": _std(
        """\
    rst = (uint8_t)(i < 2);
    en = (uint8_t)((i % 3ull) != 0);
    din = (uint32_t)i;"""
    ),
    "sync_fifo": BenchIO(
        decls="    uint8_t rst;\n    uint8_t wr_en;\n    uint8_t rd_en;\n    uint32_t din;",
        stimulus="""\
    rst = (uint8_t)(i < 2);
    wr_en = (uint8_t)((i >= 2) && ((i % 5ull) == 0));
    rd_en = (uint8_t)((i >= 20) && ((i % 7ull) == 0));
    din = (uint32_t)i;""",
        poke="    {p}rst = rst;\n    {p}wr_en = wr_en;\n    {p}rd_en = rd_en;\n    {p}din = din;",
        dump='    std::printf("%08x %x %x\\n", {p}dout, {p}full, {p}empty);\n',
        sink="(uint64_t){p}dout + {p}full + {p}empty",
    ),
    "mini_rf": BenchIO(
        decls="    uint8_t rst;\n    uint8_t we;\n    uint8_t waddr;\n    uint8_t raddr;\n    uint32_t wdata;",
        stimulus="""\
    rst = (uint8_t)(i < 2);
    we = (uint8_t)((i >= 2) && ((i % 8ull) == 0));
    waddr = (uint8_t)((i / 8ull) & 15ull);
    raddr = (uint8_t)((i / 3ull) & 15ull);
    wdata = (uint32_t)(0x9e3779b9u * (uint32_t)i);""",
        poke="    {p}rst = rst;\n    {p}we = we;\n    {p}waddr = waddr;\n    {p}raddr = raddr;\n    {p}wdata = wdata;",
        dump='    std::printf("%08x\\n", {p}rdata);\n',
        sink="{p}rdata",
    ),
    "DrawContextFifo": _draw_fifo_io(),
    "TriangleRasterizer": _raster_io(),
    "WarpScheduler": _warp_io(),
    "GpuCommandRouter": _router_io(),
    "BankedSharedMemory": _shared_io(),
    "GpuFrontend": _frontend_io(),
    "InstructionCache": _icache_io(),
    "FrontendICache": _frontend_icache_io(),
    "ScalarBackend": _scalar_io(),
    "VectorBackend": _vector_io(),
    "Fp32FmaLane": _fma_lane_io(),
    "FpuBackend": _fpu_io(),
    "FrontendScalar": _frontend_scalar_io(),
    "FrontendScalarFpu": _frontend_scalar_fpu_io(),
    "Gpu": _gpu_io(),
    "GpuSystem": _gpu_system_io(),
    "GpuHostAxi": _gpu_host_axi_io(),
}


def _fmt(io: BenchIO, prefix: str) -> tuple[str, str, str, str, str]:
    poke = io.poke.format(p=prefix)
    dump = io.dump.format(p=prefix)
    sink = io.sink.format(p=prefix)
    tail = io.tail.format(p=prefix)
    return poke, dump, sink, io.decls, tail


def _observe_evals(io: BenchIO) -> str:
    names: list[str] = []
    for name in re.findall(r"\{p\}(\w+)", io.dump + " " + io.sink + " " + io.tail):
        if name not in names:
            names.append(name)
    return "".join(f"    dut.eval_{n}();\n" for n in names)


def flashsim_main(module: str, mode: str) -> str:
    io = IOS[module]
    poke, dump, sink, decls, tail = _fmt(io, "dut.")
    do_dump = mode == "check"
    observe = _observe_evals(io)
    return f"""#include "dut.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {{
  {module}Dut dut;
  const uint64_t N = CYCLES;
  uint64_t sink = 0;
  auto t0 = std::chrono::steady_clock::now();
{io.prelude}  for (uint64_t i = 0; i < N; i++) {{
{decls}
{io.stimulus}
{poke}
    dut.tick();
{observe}    sink += {sink};
{dump if do_dump else ""}{tail}  }}
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  if (!{int(do_dump)}) {{
    std::printf("sink=%llu seconds=%.6f mhz=%.3f\\n",
                (unsigned long long)sink, secs, (N / secs) / 1e6);
  }}
  return 0;
}}
"""


def verilator_main(module: str, mode: str) -> str:
    io = IOS[module]
    poke, dump, sink, decls, tail = _fmt(io, "top.")
    do_dump = mode == "check"
    return f"""#include "V{module}.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {{
  Verilated::commandArgs(argc, argv);
  V{module} top;
  const uint64_t N = CYCLES;
  uint64_t sink = 0;
  auto t0 = std::chrono::steady_clock::now();
{io.prelude}  for (uint64_t i = 0; i < N; i++) {{
{decls}
{io.stimulus}
{poke}
    top.{io.clock} = 0;
    top.eval();
    top.{io.clock} = 1;
    top.eval();
    sink += {sink};
{dump if do_dump else ""}{tail}  }}
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  if (!{int(do_dump)}) {{
    std::printf("sink=%llu seconds=%.6f mhz=%.3f\\n",
                (unsigned long long)sink, secs, (N / secs) / 1e6);
  }}
  return 0;
}}
"""
