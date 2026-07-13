// kernel: 27 taps, 27 nonzero, weights folded in
module kernel(input logic signed [7:0] x0, input logic signed [7:0] x1, input logic signed [7:0] x2, input logic signed [7:0] x3, input logic signed [7:0] x4, input logic signed [7:0] x5, input logic signed [7:0] x6, input logic signed [7:0] x7, input logic signed [7:0] x8, input logic signed [7:0] x9, input logic signed [7:0] x10, input logic signed [7:0] x11, input logic signed [7:0] x12, input logic signed [7:0] x13, input logic signed [7:0] x14, input logic signed [7:0] x15, input logic signed [7:0] x16, input logic signed [7:0] x17, input logic signed [7:0] x18, input logic signed [7:0] x19, input logic signed [7:0] x20, input logic signed [7:0] x21, input logic signed [7:0] x22, input logic signed [7:0] x23, input logic signed [7:0] x24, input logic signed [7:0] x25, input logic signed [7:0] x26, output logic signed [7:0] y);
  logic signed [21:0] acc;
  logic signed [53:0]     scaled;

  assign acc    = 22'sd89*x0 + (-22'sd29)*x1 + 22'sd67*x2 + 22'sd100*x3 + (-22'sd20)*x4 + (-22'sd117)*x5 + (-22'sd61)*x6 + 22'sd120*x7 + 22'sd3*x8 + (-22'sd3)*x9 + (-22'sd24)*x10 + 22'sd108*x11 + 22'sd73*x12 + 22'sd85*x13 + (-22'sd50)*x14 + 22'sd120*x15 + (-22'sd5)*x16 + (-22'sd36)*x17 + 22'sd22*x18 + 22'sd101*x19 + 22'sd105*x20 + (-22'sd72)*x21 + 22'sd2*x22 + (-22'sd92)*x23 + (-22'sd55)*x24 + (-22'sd92)*x25 + 22'sd66*x26 + (-22'sd437);
  assign scaled = (acc * 54'sd45113) >>> 16;

  always_comb begin
    if (scaled < 54'sd0)      y = 8'sd0;
    else if (scaled > 54'sd255) y = 8'sd255;
    else                                y = scaled[7:0];
  end
endmodule
