from __future__ import annotations

from pathlib import Path


def mix_chain(prefix: str, src: str, rounds: int) -> tuple[str, list[str]]:
    """Return (final_wire, assign_lines) for an expensive 32-bit mix."""
    lines: list[str] = []
    prev = src
    for i in range(rounds):
        a = f"{prefix}_a{i}"
        b = f"{prefix}_b{i}"
        c = f"{prefix}_c{i}"
        d = f"{prefix}_d{i}"
        lines.append(f"  wire [31:0] {a};")
        lines.append(f"  wire [31:0] {b};")
        lines.append(f"  wire [31:0] {c};")
        lines.append(f"  wire [31:0] {d};")
        lines.append(f"  assign {a} = {prev} ^ ({prev} << 13);")
        lines.append(f"  assign {b} = {a} ^ ({a} >> 17);")
        lines.append(f"  assign {c} = {b} + ({b} << 5);")
        lines.append(f"  assign {d} = {c} ^ 32'h9e3779b9;")
        prev = d
    final = f"{prefix}_out"
    lines.append(f"  wire [31:0] {final};")
    lines.append(f"  assign {final} = {prev};")
    return final, lines


def gated_pipe(stages: int = 16, rounds: int = 12) -> str:
    """Pipeline whose mix logic is combinational and only sampled when valid."""
    body: list[str] = []
    mix_out: list[str] = []
    src = "din"
    for i in range(stages):
        if i == 0:
            src = "din"
        else:
            src = f"s{i-1}"
        final, lines = mix_chain(f"m{i}", src, rounds)
        mix_out.append(final)
        body.extend(lines)
        body.append(f"  reg [31:0] s{i};")
        body.append(f"  reg v{i};")
    body.append("  always @(posedge clk) begin")
    body.append("    if (rst) begin")
    for i in range(stages):
        body.append(f"      s{i} <= 32'd0;")
        body.append(f"      v{i} <= 1'b0;")
    body.append("    end else begin")
    body.append("      v0 <= en;")
    body.append(f"      if (en) s0 <= {mix_out[0]};")
    for i in range(1, stages):
        body.append(f"      v{i} <= v{i-1};")
        body.append(f"      if (v{i-1}) s{i} <= {mix_out[i]};")
    body.append("    end")
    body.append("  end")
    body.append(f"  assign dout = s{stages-1};")
    return _wrap("gated_pipe", body)


def sticky_input(copies: int = 16, rounds: int = 12) -> str:
    """Many always-enabled mixers that only change when din changes."""
    body: list[str] = []
    mix_out: list[str] = []
    for i in range(copies):
        src = "din" if i == 0 else f"(din + 32'd{i})"
        # Keep src as a wire so the parser does not need parens-in-assign-src as a new net.
        src_w = f"src{i}"
        body.append(f"  wire [31:0] {src_w};")
        if i == 0:
            body.append(f"  assign {src_w} = din;")
        else:
            body.append(f"  assign {src_w} = din + 32'd{i};")
        final, lines = mix_chain(f"m{i}", src_w, rounds)
        mix_out.append(final)
        body.extend(lines)
        body.append(f"  reg [31:0] s{i};")
    body.append("  always @(posedge clk) begin")
    body.append("    if (rst) begin")
    for i in range(copies):
        body.append(f"      s{i} <= 32'd0;")
    body.append("    end else begin")
    for i in range(copies):
        body.append(f"      s{i} <= {mix_out[i]};")
    body.append("    end")
    body.append("  end")
    xor_terms = " ^ ".join(f"s{i}" for i in range(copies))
    body.append(f"  assign dout = {xor_terms};")
    return _wrap("sticky_input", body)


def busy_alu(copies: int = 16, rounds: int = 12) -> str:
    """Same as sticky_input; stimulus changes din every cycle."""
    src = sticky_input(copies, rounds)
    return src.replace("module sticky_input", "module busy_alu", 1)


def counter() -> str:
    return """module counter(
  input clk,
  input rst,
  input en,
  input [31:0] din,
  output [31:0] dout
);
  reg [31:0] acc;
  always @(posedge clk) begin
    if (rst) acc <= 32'd0;
    else if (en) acc <= acc + din;
    else acc <= acc + 32'd1;
  end
  assign dout = acc;
endmodule
"""


def sync_fifo() -> str:
    return """module sync_fifo(
  input clk,
  input rst,
  input wr_en,
  input rd_en,
  input [31:0] din,
  output [31:0] dout,
  output full,
  output empty
);
  localparam DEPTH = 8;
  localparam AW = 3;
  reg [31:0] mem [0:DEPTH-1];
  reg [AW:0] wptr, rptr;
  wire [AW-1:0] waddr = wptr[AW-1:0];
  wire [AW-1:0] raddr = rptr[AW-1:0];
  assign full  = (wptr[AW] != rptr[AW]) && (waddr == raddr);
  assign empty = (wptr == rptr);
  always @(posedge clk) begin
    if (rst) begin
      wptr <= 0;
      rptr <= 0;
    end else begin
      if (wr_en && !full) begin
        mem[waddr] <= din;
        wptr <= wptr + 1;
      end
      if (rd_en && !empty)
        rptr <= rptr + 1;
    end
  end
  assign dout = mem[raddr];
endmodule
"""


def cmp_acc() -> str:
    return """module cmp_acc(
  input clk,
  input rst,
  input en,
  input [31:0] din,
  output [31:0] dout
);
  reg [31:0] acc;
  always @(posedge clk) begin
    if (rst) acc <= 0;
    else if (en && (din > acc)) acc <= din;
    else if (din == 32'd0) acc <= acc + 1;
  end
  assign dout = acc;
endmodule
"""


def hier_pipe() -> str:
    return """module add1(input [7:0] a, output [7:0] y);
  assign y = a + 8'd1;
endmodule
module hier_pipe(
  input clk,
  input rst,
  input en,
  input [31:0] din,
  output [31:0] dout
);
  wire [7:0] s;
  add1 u0(.a(din[7:0]), .y(s));
  reg [31:0] acc;
  always @(posedge clk) begin
    if (rst) acc <= 0;
    else if (en) acc <= acc + {24'd0, s};
  end
  assign dout = acc;
endmodule
"""


def mini_rf() -> str:
    return """module mini_rf(
  input clk,
  input rst,
  input we,
  input [3:0] waddr,
  input [3:0] raddr,
  input [31:0] wdata,
  output [31:0] rdata
);
  reg [31:0] rf [0:15];
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i < 16; i = i + 1) rf[i] <= 0;
    end else if (we)
      rf[waddr] <= wdata;
  end
  assign rdata = rf[raddr];
endmodule
"""


def _wrap(name: str, body: list[str]) -> str:
    header = f"""module {name}(
  input clk,
  input rst,
  input en,
  input [31:0] din,
  output [31:0] dout
);
"""
    return header + "\n".join(body) + "\nendmodule\n"


BENCHES = {
    "gated_pipe": gated_pipe,
    "sticky_input": sticky_input,
    "busy_alu": busy_alu,
    "counter": counter,
    "sync_fifo": sync_fifo,
    "cmp_acc": cmp_acc,
    "hier_pipe": hier_pipe,
    "mini_rf": mini_rf,
}

# Benches that need CIRCT (memory, hierarchy, or the native parser cannot accept them).
CIRCT_ONLY = {
    "sync_fifo",
    "hier_pipe",
    "mini_rf",
    "DrawContextFifo",
    "TriangleRasterizer",
    "WarpScheduler",
    "GpuCommandRouter",
    "BankedSharedMemory",
    "GpuFrontend",
    "InstructionCache",
    "FrontendICache",
    "ScalarBackend",
    "VectorBackend",
    "FpuBackend",
    "FrontendScalar",
    "FrontendScalarFpu",
    "Gpu",
    "GpuSystem",
    "GpuHostAxi",
}


def write_benches(out_dir: Path, stages: int = 16, rounds: int = 12) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["gated_pipe"] = out_dir / "gated_pipe.v"
    paths["gated_pipe"].write_text(gated_pipe(stages, rounds))
    paths["sticky_input"] = out_dir / "sticky_input.v"
    paths["sticky_input"].write_text(sticky_input(stages, rounds))
    paths["busy_alu"] = out_dir / "busy_alu.v"
    paths["busy_alu"].write_text(busy_alu(stages, rounds))
    paths["counter"] = out_dir / "counter.v"
    paths["counter"].write_text(counter())
    paths["sync_fifo"] = out_dir / "sync_fifo.v"
    paths["sync_fifo"].write_text(sync_fifo())
    paths["cmp_acc"] = out_dir / "cmp_acc.v"
    paths["cmp_acc"].write_text(cmp_acc())
    paths["hier_pipe"] = out_dir / "hier_pipe.v"
    paths["hier_pipe"].write_text(hier_pipe())
    paths["mini_rf"] = out_dir / "mini_rf.v"
    paths["mini_rf"].write_text(mini_rf())
    return paths
