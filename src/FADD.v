module ShiftRightJam(
  input  [25:0] io_in,
  input  [7:0]  io_shamt,
  output [25:0] io_out,
  output        io_sticky
);
  wire  exceed_max_shift = io_shamt > 8'h1a; // @[ShiftRightJam.scala 17:35]
  wire [4:0] shamt = io_shamt[4:0]; // @[ShiftRightJam.scala 18:23]
  wire [31:0] _sticky_mask_T = 32'h1 << shamt; // @[ShiftRightJam.scala 20:11]
  wire [31:0] _sticky_mask_T_2 = _sticky_mask_T - 32'h1; // @[ShiftRightJam.scala 20:30]
  wire [25:0] _sticky_mask_T_5 = exceed_max_shift ? 26'h3ffffff : 26'h0; // @[Bitwise.scala 74:12]
  wire [25:0] sticky_mask = _sticky_mask_T_2[25:0] | _sticky_mask_T_5; // @[ShiftRightJam.scala 20:49]
  wire [25:0] _io_out_T = io_in >> io_shamt; // @[ShiftRightJam.scala 21:46]
  wire [25:0] _io_sticky_T = io_in & sticky_mask; // @[ShiftRightJam.scala 22:23]
  assign io_out = exceed_max_shift ? 26'h0 : _io_out_T; // @[ShiftRightJam.scala 21:16]
  assign io_sticky = |_io_sticky_T; // @[ShiftRightJam.scala 22:41]
endmodule
module RoundingUnit(
  input  [22:0] io_in,
  input         io_roundIn,
  input         io_stickyIn,
  input         io_signIn,
  input  [2:0]  io_rm,
  output [22:0] io_out,
  output        io_inexact,
  output        io_cout
);
  wire  g = io_in[0]; // @[RoundingUnit.scala 19:25]
  wire  inexact = io_roundIn | io_stickyIn; // @[RoundingUnit.scala 20:19]
  wire  _r_up_T_4 = io_roundIn & io_stickyIn | io_roundIn & ~io_stickyIn & g; // @[RoundingUnit.scala 25:24]
  wire  _r_up_T_6 = inexact & ~io_signIn; // @[RoundingUnit.scala 27:23]
  wire  _r_up_T_7 = inexact & io_signIn; // @[RoundingUnit.scala 28:23]
  wire  _r_up_T_11 = 3'h1 == io_rm ? 1'h0 : 3'h0 == io_rm & _r_up_T_4; // @[Mux.scala 81:58]
  wire  _r_up_T_13 = 3'h3 == io_rm ? _r_up_T_6 : _r_up_T_11; // @[Mux.scala 81:58]
  wire  _r_up_T_15 = 3'h2 == io_rm ? _r_up_T_7 : _r_up_T_13; // @[Mux.scala 81:58]
  wire  r_up = 3'h4 == io_rm ? io_roundIn : _r_up_T_15; // @[Mux.scala 81:58]
  wire [22:0] out_r_up = io_in + 23'h1; // @[RoundingUnit.scala 32:24]
  assign io_out = r_up ? out_r_up : io_in; // @[RoundingUnit.scala 33:16]
  assign io_inexact = io_roundIn | io_stickyIn; // @[RoundingUnit.scala 20:19]
  assign io_cout = r_up & (&io_in); // @[RoundingUnit.scala 36:19]
endmodule
module FarPath(
  input         io_in_a_sign,
  input  [7:0]  io_in_a_exp,
  input  [23:0] io_in_a_sig,
  input  [23:0] io_in_b_sig,
  input  [7:0]  io_in_expDiff,
  input         io_in_effSub,
  input         io_in_smallAdd,
  input  [2:0]  io_in_rm,
  output        io_out_result_sign,
  output [7:0]  io_out_result_exp,
  output [25:0] io_out_result_sig,
  output        io_out_tininess
);
  wire [25:0] shiftRightJam_io_in; // @[ShiftRightJam.scala 27:31]
  wire [7:0] shiftRightJam_io_shamt; // @[ShiftRightJam.scala 27:31]
  wire [25:0] shiftRightJam_io_out; // @[ShiftRightJam.scala 27:31]
  wire  shiftRightJam_io_sticky; // @[ShiftRightJam.scala 27:31]
  wire [22:0] tininess_rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] tininess_rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] tininess_rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire [27:0] adder_in_sig_b = {1'h0,shiftRightJam_io_out,shiftRightJam_io_sticky}; // @[Cat.scala 31:58]
  wire [27:0] adder_in_sig_a = {1'h0,io_in_a_sig,3'h0}; // @[Cat.scala 31:58]
  wire [27:0] _adder_result_T = ~adder_in_sig_b; // @[FADD.scala 34:19]
  wire [27:0] _adder_result_T_1 = io_in_effSub ? _adder_result_T : adder_in_sig_b; // @[FADD.scala 34:10]
  wire [27:0] _adder_result_T_3 = adder_in_sig_a + _adder_result_T_1; // @[FADD.scala 33:20]
  wire [27:0] _GEN_0 = {{27'd0}, io_in_effSub}; // @[FADD.scala 34:61]
  wire [27:0] adder_result = _adder_result_T_3 + _GEN_0; // @[FADD.scala 34:61]
  wire [7:0] exp_a_plus_1 = io_in_a_exp + 8'h1; // @[FADD.scala 36:28]
  wire [7:0] exp_a_minus_1 = io_in_a_exp - 8'h1; // @[FADD.scala 37:29]
  wire  cout = adder_result[27]; // @[FADD.scala 39:31]
  wire  keep = adder_result[27:26] == 2'h1; // @[FADD.scala 40:35]
  wire  cancellation = adder_result[27:26] == 2'h0; // @[FADD.scala 41:43]
  wire  _far_path_sig_T = keep | io_in_smallAdd; // @[FADD.scala 44:20]
  wire  _far_path_sig_T_2 = cancellation & ~io_in_smallAdd; // @[FADD.scala 44:46]
  wire [24:0] _far_path_sig_T_8 = cout ? adder_result[27:3] : 25'h0; // @[Mux.scala 27:73]
  wire [24:0] _far_path_sig_T_9 = _far_path_sig_T ? adder_result[26:2] : 25'h0; // @[Mux.scala 27:73]
  wire [24:0] _far_path_sig_T_10 = _far_path_sig_T_2 ? adder_result[25:1] : 25'h0; // @[Mux.scala 27:73]
  wire [24:0] _far_path_sig_T_11 = _far_path_sig_T_8 | _far_path_sig_T_9; // @[Mux.scala 27:73]
  wire [24:0] far_path_sig = _far_path_sig_T_11 | _far_path_sig_T_10; // @[Mux.scala 27:73]
  wire  _far_path_sticky_T_4 = |adder_result[2:0]; // @[FADD.scala 55:43]
  wire  _far_path_sticky_T_6 = |adder_result[1:0]; // @[FADD.scala 56:43]
  wire  _far_path_sticky_T_8 = |adder_result[0]; // @[FADD.scala 57:43]
  wire  far_path_sticky = cout & _far_path_sticky_T_4 | _far_path_sig_T & _far_path_sticky_T_6 | _far_path_sig_T_2 &
    _far_path_sticky_T_8; // @[Mux.scala 27:73]
  wire [7:0] _far_path_exp_T = cout ? exp_a_plus_1 : 8'h0; // @[Mux.scala 27:73]
  wire [7:0] _far_path_exp_T_1 = keep ? io_in_a_exp : 8'h0; // @[Mux.scala 27:73]
  wire [7:0] _far_path_exp_T_2 = cancellation ? exp_a_minus_1 : 8'h0; // @[Mux.scala 27:73]
  wire [7:0] _far_path_exp_T_3 = _far_path_exp_T | _far_path_exp_T_1; // @[Mux.scala 27:73]
  wire  _tininess_T_7 = adder_result[26:25] == 2'h1 & ~tininess_rounder_io_cout; // @[FADD.scala 75:48]
  wire  _tininess_T_8 = adder_result[26:25] == 2'h0 | _tininess_T_7; // @[FADD.scala 74:46]
  ShiftRightJam shiftRightJam ( // @[ShiftRightJam.scala 27:31]
    .io_in(shiftRightJam_io_in),
    .io_shamt(shiftRightJam_io_shamt),
    .io_out(shiftRightJam_io_out),
    .io_sticky(shiftRightJam_io_sticky)
  );
  RoundingUnit tininess_rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(tininess_rounder_io_in),
    .io_roundIn(tininess_rounder_io_roundIn),
    .io_stickyIn(tininess_rounder_io_stickyIn),
    .io_signIn(tininess_rounder_io_signIn),
    .io_rm(tininess_rounder_io_rm),
    .io_out(tininess_rounder_io_out),
    .io_inexact(tininess_rounder_io_inexact),
    .io_cout(tininess_rounder_io_cout)
  );
  assign io_out_result_sign = io_in_a_sign; // @[FADD.scala 80:20 81:15]
  assign io_out_result_exp = _far_path_exp_T_3 | _far_path_exp_T_2; // @[Mux.scala 27:73]
  assign io_out_result_sig = {far_path_sig,far_path_sticky}; // @[Cat.scala 31:58]
  assign io_out_tininess = io_in_smallAdd & _tininess_T_8; // @[FADD.scala 73:27]
  assign shiftRightJam_io_in = {io_in_b_sig,2'h0}; // @[Cat.scala 31:58]
  assign shiftRightJam_io_shamt = io_in_expDiff; // @[ShiftRightJam.scala 29:28]
  assign tininess_rounder_io_in = adder_result[24:2]; // @[RoundingUnit.scala 45:33]
  assign tininess_rounder_io_roundIn = adder_result[1]; // @[RoundingUnit.scala 46:50]
  assign tininess_rounder_io_stickyIn = |adder_result[0]; // @[RoundingUnit.scala 47:54]
  assign tininess_rounder_io_signIn = io_in_a_sign; // @[RoundingUnit.scala 49:23]
  assign tininess_rounder_io_rm = io_in_rm; // @[RoundingUnit.scala 48:19]
endmodule
module LZA(
  input  [24:0] io_a,
  input  [24:0] io_b,
  output [24:0] io_f
);
  wire  k_0 = ~io_a[0] & ~io_b[0]; // @[LZA.scala 19:21]
  wire  p_1 = io_a[1] ^ io_b[1]; // @[LZA.scala 18:18]
  wire  k_1 = ~io_a[1] & ~io_b[1]; // @[LZA.scala 19:21]
  wire  f_1 = p_1 ^ ~k_0; // @[LZA.scala 23:20]
  wire  p_2 = io_a[2] ^ io_b[2]; // @[LZA.scala 18:18]
  wire  k_2 = ~io_a[2] & ~io_b[2]; // @[LZA.scala 19:21]
  wire  f_2 = p_2 ^ ~k_1; // @[LZA.scala 23:20]
  wire  p_3 = io_a[3] ^ io_b[3]; // @[LZA.scala 18:18]
  wire  k_3 = ~io_a[3] & ~io_b[3]; // @[LZA.scala 19:21]
  wire  f_3 = p_3 ^ ~k_2; // @[LZA.scala 23:20]
  wire  p_4 = io_a[4] ^ io_b[4]; // @[LZA.scala 18:18]
  wire  k_4 = ~io_a[4] & ~io_b[4]; // @[LZA.scala 19:21]
  wire  f_4 = p_4 ^ ~k_3; // @[LZA.scala 23:20]
  wire  p_5 = io_a[5] ^ io_b[5]; // @[LZA.scala 18:18]
  wire  k_5 = ~io_a[5] & ~io_b[5]; // @[LZA.scala 19:21]
  wire  f_5 = p_5 ^ ~k_4; // @[LZA.scala 23:20]
  wire  p_6 = io_a[6] ^ io_b[6]; // @[LZA.scala 18:18]
  wire  k_6 = ~io_a[6] & ~io_b[6]; // @[LZA.scala 19:21]
  wire  f_6 = p_6 ^ ~k_5; // @[LZA.scala 23:20]
  wire  p_7 = io_a[7] ^ io_b[7]; // @[LZA.scala 18:18]
  wire  k_7 = ~io_a[7] & ~io_b[7]; // @[LZA.scala 19:21]
  wire  f_7 = p_7 ^ ~k_6; // @[LZA.scala 23:20]
  wire  p_8 = io_a[8] ^ io_b[8]; // @[LZA.scala 18:18]
  wire  k_8 = ~io_a[8] & ~io_b[8]; // @[LZA.scala 19:21]
  wire  f_8 = p_8 ^ ~k_7; // @[LZA.scala 23:20]
  wire  p_9 = io_a[9] ^ io_b[9]; // @[LZA.scala 18:18]
  wire  k_9 = ~io_a[9] & ~io_b[9]; // @[LZA.scala 19:21]
  wire  f_9 = p_9 ^ ~k_8; // @[LZA.scala 23:20]
  wire  p_10 = io_a[10] ^ io_b[10]; // @[LZA.scala 18:18]
  wire  k_10 = ~io_a[10] & ~io_b[10]; // @[LZA.scala 19:21]
  wire  f_10 = p_10 ^ ~k_9; // @[LZA.scala 23:20]
  wire  p_11 = io_a[11] ^ io_b[11]; // @[LZA.scala 18:18]
  wire  k_11 = ~io_a[11] & ~io_b[11]; // @[LZA.scala 19:21]
  wire  f_11 = p_11 ^ ~k_10; // @[LZA.scala 23:20]
  wire  p_12 = io_a[12] ^ io_b[12]; // @[LZA.scala 18:18]
  wire  k_12 = ~io_a[12] & ~io_b[12]; // @[LZA.scala 19:21]
  wire  f_12 = p_12 ^ ~k_11; // @[LZA.scala 23:20]
  wire  p_13 = io_a[13] ^ io_b[13]; // @[LZA.scala 18:18]
  wire  k_13 = ~io_a[13] & ~io_b[13]; // @[LZA.scala 19:21]
  wire  f_13 = p_13 ^ ~k_12; // @[LZA.scala 23:20]
  wire  p_14 = io_a[14] ^ io_b[14]; // @[LZA.scala 18:18]
  wire  k_14 = ~io_a[14] & ~io_b[14]; // @[LZA.scala 19:21]
  wire  f_14 = p_14 ^ ~k_13; // @[LZA.scala 23:20]
  wire  p_15 = io_a[15] ^ io_b[15]; // @[LZA.scala 18:18]
  wire  k_15 = ~io_a[15] & ~io_b[15]; // @[LZA.scala 19:21]
  wire  f_15 = p_15 ^ ~k_14; // @[LZA.scala 23:20]
  wire  p_16 = io_a[16] ^ io_b[16]; // @[LZA.scala 18:18]
  wire  k_16 = ~io_a[16] & ~io_b[16]; // @[LZA.scala 19:21]
  wire  f_16 = p_16 ^ ~k_15; // @[LZA.scala 23:20]
  wire  p_17 = io_a[17] ^ io_b[17]; // @[LZA.scala 18:18]
  wire  k_17 = ~io_a[17] & ~io_b[17]; // @[LZA.scala 19:21]
  wire  f_17 = p_17 ^ ~k_16; // @[LZA.scala 23:20]
  wire  p_18 = io_a[18] ^ io_b[18]; // @[LZA.scala 18:18]
  wire  k_18 = ~io_a[18] & ~io_b[18]; // @[LZA.scala 19:21]
  wire  f_18 = p_18 ^ ~k_17; // @[LZA.scala 23:20]
  wire  p_19 = io_a[19] ^ io_b[19]; // @[LZA.scala 18:18]
  wire  k_19 = ~io_a[19] & ~io_b[19]; // @[LZA.scala 19:21]
  wire  f_19 = p_19 ^ ~k_18; // @[LZA.scala 23:20]
  wire  p_20 = io_a[20] ^ io_b[20]; // @[LZA.scala 18:18]
  wire  k_20 = ~io_a[20] & ~io_b[20]; // @[LZA.scala 19:21]
  wire  f_20 = p_20 ^ ~k_19; // @[LZA.scala 23:20]
  wire  p_21 = io_a[21] ^ io_b[21]; // @[LZA.scala 18:18]
  wire  k_21 = ~io_a[21] & ~io_b[21]; // @[LZA.scala 19:21]
  wire  f_21 = p_21 ^ ~k_20; // @[LZA.scala 23:20]
  wire  p_22 = io_a[22] ^ io_b[22]; // @[LZA.scala 18:18]
  wire  k_22 = ~io_a[22] & ~io_b[22]; // @[LZA.scala 19:21]
  wire  f_22 = p_22 ^ ~k_21; // @[LZA.scala 23:20]
  wire  p_23 = io_a[23] ^ io_b[23]; // @[LZA.scala 18:18]
  wire  k_23 = ~io_a[23] & ~io_b[23]; // @[LZA.scala 19:21]
  wire  f_23 = p_23 ^ ~k_22; // @[LZA.scala 23:20]
  wire  p_24 = io_a[24] ^ io_b[24]; // @[LZA.scala 18:18]
  wire  f_24 = p_24 ^ ~k_23; // @[LZA.scala 23:20]
  wire [5:0] io_f_lo_lo = {f_5,f_4,f_3,f_2,f_1,1'h0}; // @[Cat.scala 31:58]
  wire [11:0] io_f_lo = {f_11,f_10,f_9,f_8,f_7,f_6,io_f_lo_lo}; // @[Cat.scala 31:58]
  wire [5:0] io_f_hi_lo = {f_17,f_16,f_15,f_14,f_13,f_12}; // @[Cat.scala 31:58]
  wire [12:0] io_f_hi = {f_24,f_23,f_22,f_21,f_20,f_19,f_18,io_f_hi_lo}; // @[Cat.scala 31:58]
  assign io_f = {io_f_hi,io_f_lo}; // @[Cat.scala 31:58]
endmodule
module CLZ(
  input  [24:0] io_in,
  output [4:0]  io_out
);
  wire [4:0] _io_out_T_25 = io_in[1] ? 5'h17 : 5'h18; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_26 = io_in[2] ? 5'h16 : _io_out_T_25; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_27 = io_in[3] ? 5'h15 : _io_out_T_26; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_28 = io_in[4] ? 5'h14 : _io_out_T_27; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_29 = io_in[5] ? 5'h13 : _io_out_T_28; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_30 = io_in[6] ? 5'h12 : _io_out_T_29; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_31 = io_in[7] ? 5'h11 : _io_out_T_30; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_32 = io_in[8] ? 5'h10 : _io_out_T_31; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_33 = io_in[9] ? 5'hf : _io_out_T_32; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_34 = io_in[10] ? 5'he : _io_out_T_33; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_35 = io_in[11] ? 5'hd : _io_out_T_34; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_36 = io_in[12] ? 5'hc : _io_out_T_35; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_37 = io_in[13] ? 5'hb : _io_out_T_36; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_38 = io_in[14] ? 5'ha : _io_out_T_37; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_39 = io_in[15] ? 5'h9 : _io_out_T_38; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_40 = io_in[16] ? 5'h8 : _io_out_T_39; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_41 = io_in[17] ? 5'h7 : _io_out_T_40; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_42 = io_in[18] ? 5'h6 : _io_out_T_41; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_43 = io_in[19] ? 5'h5 : _io_out_T_42; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_44 = io_in[20] ? 5'h4 : _io_out_T_43; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_45 = io_in[21] ? 5'h3 : _io_out_T_44; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_46 = io_in[22] ? 5'h2 : _io_out_T_45; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_47 = io_in[23] ? 5'h1 : _io_out_T_46; // @[Mux.scala 47:70]
  assign io_out = io_in[24] ? 5'h0 : _io_out_T_47; // @[Mux.scala 47:70]
endmodule
module NearPath(
  input         io_in_a_sign,
  input  [7:0]  io_in_a_exp,
  input  [23:0] io_in_a_sig,
  input         io_in_b_sign,
  input  [23:0] io_in_b_sig,
  input         io_in_need_shift_b,
  input  [2:0]  io_in_rm,
  output        io_out_result_sign,
  output [7:0]  io_out_result_exp,
  output [25:0] io_out_result_sig,
  output        io_out_sig_is_zero,
  output        io_out_a_lt_b,
  output        io_out_tininess
);
  wire [24:0] lza_ab_io_a; // @[FADD.scala 112:22]
  wire [24:0] lza_ab_io_b; // @[FADD.scala 112:22]
  wire [24:0] lza_ab_io_f; // @[FADD.scala 112:22]
  wire [24:0] lzc_clz_io_in; // @[CLZ.scala 22:21]
  wire [4:0] lzc_clz_io_out; // @[CLZ.scala 22:21]
  wire [22:0] tininess_rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] tininess_rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] tininess_rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire [24:0] _b_sig_T = {io_in_b_sig,1'h0}; // @[Cat.scala 31:58]
  wire [24:0] b_sig = _b_sig_T >> io_in_need_shift_b; // @[FADD.scala 106:37]
  wire [24:0] b_neg = ~b_sig; // @[FADD.scala 107:16]
  wire [25:0] _a_minus_b_T = {1'h0,io_in_a_sig,1'h0}; // @[Cat.scala 31:58]
  wire [25:0] _a_minus_b_T_1 = {1'h1,b_neg}; // @[Cat.scala 31:58]
  wire [25:0] _a_minus_b_T_3 = _a_minus_b_T + _a_minus_b_T_1; // @[FADD.scala 109:40]
  wire [25:0] a_minus_b = _a_minus_b_T_3 + 26'h1; // @[FADD.scala 109:63]
  wire  a_lt_b = a_minus_b[25]; // @[FADD.scala 110:30]
  wire [24:0] sig_raw = a_minus_b[24:0]; // @[FADD.scala 111:31]
  wire  lza_str_zero = ~(|lza_ab_io_f); // @[FADD.scala 116:22]
  wire  need_shift_lim = io_in_a_exp < 8'h19; // @[FADD.scala 119:30]
  wire [25:0] _shift_lim_mask_raw_T_2 = 26'h2000000 >> io_in_a_exp[4:0]; // @[FADD.scala 122:41]
  wire [24:0] shift_lim_mask_raw = _shift_lim_mask_raw_T_2[24:0]; // @[FADD.scala 123:16]
  wire [24:0] shift_lim_mask = need_shift_lim ? shift_lim_mask_raw : 25'h0; // @[FADD.scala 124:27]
  wire [24:0] _shift_lim_bit_T = shift_lim_mask_raw & sig_raw; // @[FADD.scala 125:43]
  wire  shift_lim_bit = |_shift_lim_bit_T; // @[FADD.scala 125:57]
  wire [24:0] lzc_str = shift_lim_mask | lza_ab_io_f; // @[FADD.scala 127:32]
  wire  _int_bit_mask_T_5 = lzc_str[23] & ~(|lzc_str[24]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_10 = lzc_str[22] & ~(|lzc_str[24:23]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_15 = lzc_str[21] & ~(|lzc_str[24:22]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_20 = lzc_str[20] & ~(|lzc_str[24:21]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_25 = lzc_str[19] & ~(|lzc_str[24:20]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_30 = lzc_str[18] & ~(|lzc_str[24:19]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_35 = lzc_str[17] & ~(|lzc_str[24:18]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_40 = lzc_str[16] & ~(|lzc_str[24:17]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_45 = lzc_str[15] & ~(|lzc_str[24:16]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_50 = lzc_str[14] & ~(|lzc_str[24:15]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_55 = lzc_str[13] & ~(|lzc_str[24:14]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_60 = lzc_str[12] & ~(|lzc_str[24:13]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_65 = lzc_str[11] & ~(|lzc_str[24:12]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_70 = lzc_str[10] & ~(|lzc_str[24:11]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_75 = lzc_str[9] & ~(|lzc_str[24:10]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_80 = lzc_str[8] & ~(|lzc_str[24:9]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_85 = lzc_str[7] & ~(|lzc_str[24:8]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_90 = lzc_str[6] & ~(|lzc_str[24:7]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_95 = lzc_str[5] & ~(|lzc_str[24:6]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_100 = lzc_str[4] & ~(|lzc_str[24:5]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_105 = lzc_str[3] & ~(|lzc_str[24:4]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_110 = lzc_str[2] & ~(|lzc_str[24:3]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_115 = lzc_str[1] & ~(|lzc_str[24:2]); // @[FADD.scala 132:40]
  wire  _int_bit_mask_T_120 = lzc_str[0] & ~(|lzc_str[24:1]); // @[FADD.scala 132:40]
  wire [5:0] int_bit_mask_lo_lo = {_int_bit_mask_T_95,_int_bit_mask_T_100,_int_bit_mask_T_105,_int_bit_mask_T_110,
    _int_bit_mask_T_115,_int_bit_mask_T_120}; // @[Cat.scala 31:58]
  wire [11:0] int_bit_mask_lo = {_int_bit_mask_T_65,_int_bit_mask_T_70,_int_bit_mask_T_75,_int_bit_mask_T_80,
    _int_bit_mask_T_85,_int_bit_mask_T_90,int_bit_mask_lo_lo}; // @[Cat.scala 31:58]
  wire [5:0] int_bit_mask_hi_lo = {_int_bit_mask_T_35,_int_bit_mask_T_40,_int_bit_mask_T_45,_int_bit_mask_T_50,
    _int_bit_mask_T_55,_int_bit_mask_T_60}; // @[Cat.scala 31:58]
  wire [24:0] int_bit_mask = {lzc_str[24],_int_bit_mask_T_5,_int_bit_mask_T_10,_int_bit_mask_T_15,_int_bit_mask_T_20,
    _int_bit_mask_T_25,_int_bit_mask_T_30,int_bit_mask_hi_lo,int_bit_mask_lo}; // @[Cat.scala 31:58]
  wire [24:0] _GEN_0 = {{24'd0}, lza_str_zero}; // @[FADD.scala 136:20]
  wire [24:0] _int_bit_predicted_T = int_bit_mask | _GEN_0; // @[FADD.scala 136:20]
  wire [24:0] _int_bit_predicted_T_1 = _int_bit_predicted_T & sig_raw; // @[FADD.scala 136:36]
  wire  int_bit_predicted = |_int_bit_predicted_T_1; // @[FADD.scala 136:50]
  wire [24:0] _int_bit_rshift_1_T = {{1'd0}, int_bit_mask[24:1]}; // @[FADD.scala 138:20]
  wire [24:0] _int_bit_rshift_1_T_1 = _int_bit_rshift_1_T & sig_raw; // @[FADD.scala 138:37]
  wire  int_bit_rshift_1 = |_int_bit_rshift_1_T_1; // @[FADD.scala 138:51]
  wire  _exceed_lim_mask_T_1 = |lza_ab_io_f[24]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_3 = |lza_ab_io_f[24:23]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_5 = |lza_ab_io_f[24:22]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_7 = |lza_ab_io_f[24:21]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_9 = |lza_ab_io_f[24:20]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_11 = |lza_ab_io_f[24:19]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_13 = |lza_ab_io_f[24:18]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_15 = |lza_ab_io_f[24:17]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_17 = |lza_ab_io_f[24:16]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_19 = |lza_ab_io_f[24:15]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_21 = |lza_ab_io_f[24:14]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_23 = |lza_ab_io_f[24:13]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_25 = |lza_ab_io_f[24:12]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_27 = |lza_ab_io_f[24:11]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_29 = |lza_ab_io_f[24:10]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_31 = |lza_ab_io_f[24:9]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_33 = |lza_ab_io_f[24:8]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_35 = |lza_ab_io_f[24:7]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_37 = |lza_ab_io_f[24:6]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_39 = |lza_ab_io_f[24:5]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_41 = |lza_ab_io_f[24:4]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_43 = |lza_ab_io_f[24:3]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_45 = |lza_ab_io_f[24:2]; // @[FADD.scala 142:64]
  wire  _exceed_lim_mask_T_47 = |lza_ab_io_f[24:1]; // @[FADD.scala 142:64]
  wire [5:0] exceed_lim_mask_lo_lo = {_exceed_lim_mask_T_37,_exceed_lim_mask_T_39,_exceed_lim_mask_T_41,
    _exceed_lim_mask_T_43,_exceed_lim_mask_T_45,_exceed_lim_mask_T_47}; // @[Cat.scala 31:58]
  wire [11:0] exceed_lim_mask_lo = {_exceed_lim_mask_T_25,_exceed_lim_mask_T_27,_exceed_lim_mask_T_29,
    _exceed_lim_mask_T_31,_exceed_lim_mask_T_33,_exceed_lim_mask_T_35,exceed_lim_mask_lo_lo}; // @[Cat.scala 31:58]
  wire [5:0] exceed_lim_mask_hi_lo = {_exceed_lim_mask_T_13,_exceed_lim_mask_T_15,_exceed_lim_mask_T_17,
    _exceed_lim_mask_T_19,_exceed_lim_mask_T_21,_exceed_lim_mask_T_23}; // @[Cat.scala 31:58]
  wire [24:0] exceed_lim_mask = {1'h0,_exceed_lim_mask_T_1,_exceed_lim_mask_T_3,_exceed_lim_mask_T_5,
    _exceed_lim_mask_T_7,_exceed_lim_mask_T_9,_exceed_lim_mask_T_11,exceed_lim_mask_hi_lo,exceed_lim_mask_lo}; // @[Cat.scala 31:58]
  wire [24:0] _exceed_lim_T = exceed_lim_mask & shift_lim_mask_raw; // @[FADD.scala 145:41]
  wire  exceed_lim = need_shift_lim & ~(|_exceed_lim_T); // @[FADD.scala 145:20]
  wire  int_bit = exceed_lim ? shift_lim_bit : int_bit_rshift_1 | int_bit_predicted; // @[FADD.scala 148:8]
  wire  lza_error = ~int_bit_predicted & ~exceed_lim; // @[FADD.scala 150:38]
  wire [7:0] _GEN_2 = {{3'd0}, lzc_clz_io_out}; // @[FADD.scala 151:22]
  wire [7:0] exp_s1 = io_in_a_exp - _GEN_2; // @[FADD.scala 151:22]
  wire [7:0] _GEN_3 = {{7'd0}, lza_error}; // @[FADD.scala 152:23]
  wire [7:0] exp_s2 = exp_s1 - _GEN_3; // @[FADD.scala 152:23]
  wire [55:0] _GEN_4 = {{31'd0}, sig_raw}; // @[FADD.scala 153:25]
  wire [55:0] _sig_s1_T = _GEN_4 << lzc_clz_io_out; // @[FADD.scala 153:25]
  wire [24:0] sig_s1 = _sig_s1_T[24:0]; // @[FADD.scala 153:32]
  wire [24:0] _sig_s2_T_1 = {sig_s1[23:0],1'h0}; // @[Cat.scala 31:58]
  wire [24:0] near_path_sig = lza_error ? _sig_s2_T_1 : sig_s1; // @[FADD.scala 154:19]
  wire [24:0] in_pad = {near_path_sig[22:0],2'h0}; // @[Cat.scala 31:58]
  wire  _tininess_T_5 = near_path_sig[24:23] == 2'h1 & ~tininess_rounder_io_cout; // @[FADD.scala 166:39]
  LZA lza_ab ( // @[FADD.scala 112:22]
    .io_a(lza_ab_io_a),
    .io_b(lza_ab_io_b),
    .io_f(lza_ab_io_f)
  );
  CLZ lzc_clz ( // @[CLZ.scala 22:21]
    .io_in(lzc_clz_io_in),
    .io_out(lzc_clz_io_out)
  );
  RoundingUnit tininess_rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(tininess_rounder_io_in),
    .io_roundIn(tininess_rounder_io_roundIn),
    .io_stickyIn(tininess_rounder_io_stickyIn),
    .io_signIn(tininess_rounder_io_signIn),
    .io_rm(tininess_rounder_io_rm),
    .io_out(tininess_rounder_io_out),
    .io_inexact(tininess_rounder_io_inexact),
    .io_cout(tininess_rounder_io_cout)
  );
  assign io_out_result_sign = a_lt_b ? io_in_b_sign : io_in_a_sign; // @[FADD.scala 157:27]
  assign io_out_result_exp = int_bit ? exp_s2 : 8'h0; // @[FADD.scala 156:26]
  assign io_out_result_sig = {near_path_sig,1'h0}; // @[Cat.scala 31:58]
  assign io_out_sig_is_zero = lza_str_zero & ~sig_raw[0]; // @[FADD.scala 175:38]
  assign io_out_a_lt_b = a_minus_b[25]; // @[FADD.scala 110:30]
  assign io_out_tininess = near_path_sig[24:23] == 2'h0 | _tininess_T_5; // @[FADD.scala 165:52]
  assign lza_ab_io_a = {io_in_a_sig,1'h0}; // @[Cat.scala 31:58]
  assign lza_ab_io_b = ~b_sig; // @[FADD.scala 107:16]
  assign lzc_clz_io_in = shift_lim_mask | lza_ab_io_f; // @[FADD.scala 127:32]
  assign tininess_rounder_io_in = in_pad[24:2]; // @[RoundingUnit.scala 45:33]
  assign tininess_rounder_io_roundIn = in_pad[1]; // @[RoundingUnit.scala 46:50]
  assign tininess_rounder_io_stickyIn = |in_pad[0]; // @[RoundingUnit.scala 47:54]
  assign tininess_rounder_io_signIn = a_lt_b ? io_in_b_sign : io_in_a_sign; // @[FADD.scala 157:27]
  assign tininess_rounder_io_rm = io_in_rm; // @[RoundingUnit.scala 48:19]
endmodule
module FCMA_ADD(
  input  [31:0] io_a,
  input  [31:0] io_b,
  input  [2:0]  io_rm,
  output [31:0] io_result,
  output [4:0]  io_fflags
);
  wire  far_path_mods_0_io_in_a_sign; // @[FADD.scala 247:26]
  wire [7:0] far_path_mods_0_io_in_a_exp; // @[FADD.scala 247:26]
  wire [23:0] far_path_mods_0_io_in_a_sig; // @[FADD.scala 247:26]
  wire [23:0] far_path_mods_0_io_in_b_sig; // @[FADD.scala 247:26]
  wire [7:0] far_path_mods_0_io_in_expDiff; // @[FADD.scala 247:26]
  wire  far_path_mods_0_io_in_effSub; // @[FADD.scala 247:26]
  wire  far_path_mods_0_io_in_smallAdd; // @[FADD.scala 247:26]
  wire [2:0] far_path_mods_0_io_in_rm; // @[FADD.scala 247:26]
  wire  far_path_mods_0_io_out_result_sign; // @[FADD.scala 247:26]
  wire [7:0] far_path_mods_0_io_out_result_exp; // @[FADD.scala 247:26]
  wire [25:0] far_path_mods_0_io_out_result_sig; // @[FADD.scala 247:26]
  wire  far_path_mods_0_io_out_tininess; // @[FADD.scala 247:26]
  wire [22:0] far_path_rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  far_path_rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  far_path_rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  far_path_rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] far_path_rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] far_path_rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  far_path_rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  far_path_rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire  near_path_mods_0_io_in_a_sign; // @[FADD.scala 300:27]
  wire [7:0] near_path_mods_0_io_in_a_exp; // @[FADD.scala 300:27]
  wire [23:0] near_path_mods_0_io_in_a_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_in_b_sign; // @[FADD.scala 300:27]
  wire [23:0] near_path_mods_0_io_in_b_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_in_need_shift_b; // @[FADD.scala 300:27]
  wire [2:0] near_path_mods_0_io_in_rm; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_out_result_sign; // @[FADD.scala 300:27]
  wire [7:0] near_path_mods_0_io_out_result_exp; // @[FADD.scala 300:27]
  wire [25:0] near_path_mods_0_io_out_result_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_out_sig_is_zero; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_out_a_lt_b; // @[FADD.scala 300:27]
  wire  near_path_mods_0_io_out_tininess; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_in_a_sign; // @[FADD.scala 300:27]
  wire [7:0] near_path_mods_1_io_in_a_exp; // @[FADD.scala 300:27]
  wire [23:0] near_path_mods_1_io_in_a_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_in_b_sign; // @[FADD.scala 300:27]
  wire [23:0] near_path_mods_1_io_in_b_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_in_need_shift_b; // @[FADD.scala 300:27]
  wire [2:0] near_path_mods_1_io_in_rm; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_out_result_sign; // @[FADD.scala 300:27]
  wire [7:0] near_path_mods_1_io_out_result_exp; // @[FADD.scala 300:27]
  wire [25:0] near_path_mods_1_io_out_result_sig; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_out_sig_is_zero; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_out_a_lt_b; // @[FADD.scala 300:27]
  wire  near_path_mods_1_io_out_tininess; // @[FADD.scala 300:27]
  wire [22:0] near_path_rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  near_path_rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  near_path_rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  near_path_rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] near_path_rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] near_path_rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  near_path_rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  near_path_rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire  fp_a_sign = io_a[31]; // @[package.scala 59:19]
  wire [7:0] fp_a_exp = io_a[30:23]; // @[package.scala 60:18]
  wire [22:0] fp_a_sig = io_a[22:0]; // @[package.scala 61:18]
  wire  fp_b_sign = io_b[31]; // @[package.scala 59:19]
  wire [7:0] fp_b_exp = io_b[30:23]; // @[package.scala 60:18]
  wire [22:0] fp_b_sig = io_b[22:0]; // @[package.scala 61:18]
  wire  decode_a_expNotZero = |fp_a_exp; // @[package.scala 32:31]
  wire  decode_a_expIsOnes = &fp_a_exp; // @[package.scala 33:31]
  wire  decode_a_sigNotZero = |fp_a_sig; // @[package.scala 34:31]
  wire  decode_a__expIsZero = ~decode_a_expNotZero; // @[package.scala 37:27]
  wire  decode_a__sigIsZero = ~decode_a_sigNotZero; // @[package.scala 40:27]
  wire  decode_a__isInf = decode_a_expIsOnes & decode_a__sigIsZero; // @[package.scala 42:40]
  wire  decode_a__isNaN = decode_a_expIsOnes & decode_a_sigNotZero; // @[package.scala 44:40]
  wire  decode_a__isSNaN = decode_a__isNaN & ~fp_a_sig[22]; // @[package.scala 45:37]
  wire  decode_b_expNotZero = |fp_b_exp; // @[package.scala 32:31]
  wire  decode_b_expIsOnes = &fp_b_exp; // @[package.scala 33:31]
  wire  decode_b_sigNotZero = |fp_b_sig; // @[package.scala 34:31]
  wire  decode_b__expIsZero = ~decode_b_expNotZero; // @[package.scala 37:27]
  wire  decode_b__sigIsZero = ~decode_b_sigNotZero; // @[package.scala 40:27]
  wire  decode_b__isInf = decode_b_expIsOnes & decode_b__sigIsZero; // @[package.scala 42:40]
  wire  decode_b__isNaN = decode_b_expIsOnes & decode_b_sigNotZero; // @[package.scala 44:40]
  wire  decode_b__isSNaN = decode_b__isNaN & ~fp_b_sig[22]; // @[package.scala 45:37]
  wire [7:0] _GEN_0 = {{7'd0}, decode_a__expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_a_exp = fp_a_exp | _GEN_0; // @[package.scala 83:27]
  wire [23:0] raw_a_sig = {decode_a_expNotZero,fp_a_sig}; // @[Cat.scala 31:58]
  wire [7:0] _GEN_1 = {{7'd0}, decode_b__expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_b_exp = fp_b_exp | _GEN_1; // @[package.scala 83:27]
  wire [23:0] raw_b_sig = {decode_b_expNotZero,fp_b_sig}; // @[Cat.scala 31:58]
  wire  eff_sub = fp_a_sign ^ fp_b_sign; // @[FADD.scala 197:28]
  wire  special_path_hasNaN = decode_a__isNaN | decode_b__isNaN; // @[FADD.scala 208:44]
  wire  special_path_hasSNaN = decode_a__isSNaN | decode_b__isSNaN; // @[FADD.scala 209:46]
  wire  special_path_hasInf = decode_a__isInf | decode_b__isInf; // @[FADD.scala 210:44]
  wire  special_path_inf_iv = decode_a__isInf & decode_b__isInf & eff_sub; // @[FADD.scala 211:55]
  wire  special_case_happen = special_path_hasNaN | special_path_hasInf; // @[FADD.scala 213:49]
  wire  _special_path_result_T = special_path_hasNaN | special_path_inf_iv; // @[FADD.scala 215:25]
  wire  _special_path_result_T_3 = decode_a__isInf ? fp_a_sign : fp_b_sign; // @[FADD.scala 218:10]
  wire [31:0] _special_path_result_T_5 = {_special_path_result_T_3,8'hff,23'h0}; // @[Cat.scala 31:58]
  wire [31:0] special_path_result = _special_path_result_T ? 32'h7fc00000 : _special_path_result_T_5; // @[FADD.scala 214:32]
  wire  special_path_iv = special_path_hasSNaN | special_path_inf_iv; // @[FADD.scala 223:46]
  wire [4:0] special_path_fflags = {special_path_iv,4'h0}; // @[Cat.scala 31:58]
  wire [8:0] _exp_diff_a_b_T = {1'h0,raw_a_exp}; // @[Cat.scala 31:58]
  wire [8:0] _exp_diff_a_b_T_1 = {1'h0,raw_b_exp}; // @[Cat.scala 31:58]
  wire [8:0] exp_diff_a_b = _exp_diff_a_b_T - _exp_diff_a_b_T_1; // @[FADD.scala 226:47]
  wire [8:0] exp_diff_b_a = _exp_diff_a_b_T_1 - _exp_diff_a_b_T; // @[FADD.scala 227:47]
  wire  need_swap = exp_diff_a_b[8]; // @[FADD.scala 229:36]
  wire [7:0] ea_minus_eb = need_swap ? exp_diff_b_a[7:0] : exp_diff_a_b[7:0]; // @[FADD.scala 231:24]
  wire  _sel_far_path_T = ~eff_sub; // @[FADD.scala 232:22]
  wire  sel_far_path = ~eff_sub | ea_minus_eb > 8'h1; // @[FADD.scala 232:31]
  wire  _T = ~need_swap; // @[FADD.scala 240:11]
  wire [8:0] _T_5 = _T ? exp_diff_a_b : exp_diff_b_a; // @[FADD.scala 242:10]
  wire [7:0] _GEN_2 = {{7'd0}, far_path_rounder_io_cout}; // @[FADD.scala 270:55]
  wire [7:0] far_path_exp_rounded = _GEN_2 + far_path_mods_0_io_out_result_exp; // @[FADD.scala 270:55]
  wire  far_path_mul_of = decode_b_expIsOnes & _sel_far_path_T; // @[FADD.scala 273:65]
  wire  far_path_may_uf = far_path_mods_0_io_out_tininess & ~far_path_mul_of; // @[FADD.scala 274:47]
  wire  far_path_of_before_round = far_path_mods_0_io_out_result_exp == 8'hff; // @[FADD.scala 277:18]
  wire  _far_path_of_after_round_T = far_path_mods_0_io_out_result_exp == 8'hfe; // @[FADD.scala 279:18]
  wire  far_path_of_after_round = far_path_rounder_io_cout & _far_path_of_after_round_T; // @[FADD.scala 278:58]
  wire  far_path_of = far_path_of_before_round | far_path_of_after_round | far_path_mul_of; // @[FADD.scala 282:57]
  wire  far_path_ix = far_path_rounder_io_inexact | far_path_of; // @[FADD.scala 283:49]
  wire  far_path_uf = far_path_may_uf & far_path_ix; // @[FADD.scala 284:37]
  wire [31:0] far_path_result = {far_path_mods_0_io_out_result_sign,far_path_exp_rounded,far_path_rounder_io_out}; // @[Cat.scala 31:58]
  wire  near_path_exp_neq = raw_a_exp[1:0] != raw_b_exp[1:0]; // @[FADD.scala 293:43]
  wire  _near_path_out_T_2 = need_swap | ~near_path_exp_neq & near_path_mods_0_io_out_a_lt_b; // @[FADD.scala 310:15]
  wire  near_path_out_result_sign = _near_path_out_T_2 ? near_path_mods_1_io_out_result_sign :
    near_path_mods_0_io_out_result_sign; // @[FADD.scala 309:26]
  wire [7:0] near_path_out_result_exp = _near_path_out_T_2 ? near_path_mods_1_io_out_result_exp :
    near_path_mods_0_io_out_result_exp; // @[FADD.scala 309:26]
  wire [25:0] near_path_out_result_sig = _near_path_out_T_2 ? near_path_mods_1_io_out_result_sig :
    near_path_mods_0_io_out_result_sig; // @[FADD.scala 309:26]
  wire  near_path_out_sig_is_zero = _near_path_out_T_2 ? near_path_mods_1_io_out_sig_is_zero :
    near_path_mods_0_io_out_sig_is_zero; // @[FADD.scala 309:26]
  wire  near_path_out_tininess = _near_path_out_T_2 ? near_path_mods_1_io_out_tininess :
    near_path_mods_0_io_out_tininess; // @[FADD.scala 309:26]
  wire  near_path_is_zero = near_path_out_result_exp == 8'h0 & near_path_out_sig_is_zero; // @[FADD.scala 320:49]
  wire [7:0] _GEN_3 = {{7'd0}, near_path_rounder_io_cout}; // @[FADD.scala 329:57]
  wire [7:0] near_path_exp_rounded = _GEN_3 + near_path_out_result_exp; // @[FADD.scala 329:57]
  wire  near_path_zero_sign = io_rm == 3'h2; // @[FADD.scala 331:35]
  wire  _near_path_result_T_3 = near_path_out_result_sign & ~near_path_is_zero | near_path_zero_sign & near_path_is_zero
    ; // @[FADD.scala 333:44]
  wire [31:0] near_path_result = {_near_path_result_T_3,near_path_exp_rounded,near_path_rounder_io_out}; // @[Cat.scala 31:58]
  wire  near_path_of = near_path_exp_rounded == 8'hff; // @[FADD.scala 338:44]
  wire  near_path_ix = near_path_rounder_io_inexact | near_path_of; // @[FADD.scala 339:51]
  wire  near_path_uf = near_path_out_tininess & near_path_ix; // @[FADD.scala 340:45]
  wire  _common_overflow_T_1 = ~sel_far_path; // @[FADD.scala 347:36]
  wire  common_overflow = sel_far_path & far_path_of | ~sel_far_path & near_path_of; // @[FADD.scala 347:33]
  wire  common_overflow_sign = sel_far_path ? far_path_mods_0_io_out_result_sign : near_path_out_result_sign; // @[FADD.scala 349:8]
  wire  rmin = io_rm == 3'h1 | near_path_zero_sign & ~far_path_mods_0_io_out_result_sign | io_rm == 3'h3 &
    far_path_mods_0_io_out_result_sign; // @[RoundingUnit.scala 54:41]
  wire [7:0] common_overflow_exp = rmin ? 8'hfe : 8'hff; // @[FADD.scala 351:32]
  wire [22:0] common_overflow_sig = rmin ? 23'h7fffff : 23'h0; // @[FADD.scala 357:8]
  wire  common_underflow = sel_far_path & far_path_uf | _common_overflow_T_1 & near_path_uf; // @[FADD.scala 359:33]
  wire  common_inexact = sel_far_path & far_path_ix | _common_overflow_T_1 & near_path_ix; // @[FADD.scala 361:33]
  wire [4:0] common_fflags = {2'h0,common_overflow,common_underflow,common_inexact}; // @[Cat.scala 31:58]
  wire [31:0] _io_result_T = {common_overflow_sign,common_overflow_exp,common_overflow_sig}; // @[Cat.scala 31:58]
  wire [31:0] _io_result_T_1 = sel_far_path ? far_path_result : near_path_result; // @[FADD.scala 376:10]
  wire [31:0] _io_result_T_2 = common_overflow ? _io_result_T : _io_result_T_1; // @[FADD.scala 373:8]
  FarPath far_path_mods_0 ( // @[FADD.scala 247:26]
    .io_in_a_sign(far_path_mods_0_io_in_a_sign),
    .io_in_a_exp(far_path_mods_0_io_in_a_exp),
    .io_in_a_sig(far_path_mods_0_io_in_a_sig),
    .io_in_b_sig(far_path_mods_0_io_in_b_sig),
    .io_in_expDiff(far_path_mods_0_io_in_expDiff),
    .io_in_effSub(far_path_mods_0_io_in_effSub),
    .io_in_smallAdd(far_path_mods_0_io_in_smallAdd),
    .io_in_rm(far_path_mods_0_io_in_rm),
    .io_out_result_sign(far_path_mods_0_io_out_result_sign),
    .io_out_result_exp(far_path_mods_0_io_out_result_exp),
    .io_out_result_sig(far_path_mods_0_io_out_result_sig),
    .io_out_tininess(far_path_mods_0_io_out_tininess)
  );
  RoundingUnit far_path_rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(far_path_rounder_io_in),
    .io_roundIn(far_path_rounder_io_roundIn),
    .io_stickyIn(far_path_rounder_io_stickyIn),
    .io_signIn(far_path_rounder_io_signIn),
    .io_rm(far_path_rounder_io_rm),
    .io_out(far_path_rounder_io_out),
    .io_inexact(far_path_rounder_io_inexact),
    .io_cout(far_path_rounder_io_cout)
  );
  NearPath near_path_mods_0 ( // @[FADD.scala 300:27]
    .io_in_a_sign(near_path_mods_0_io_in_a_sign),
    .io_in_a_exp(near_path_mods_0_io_in_a_exp),
    .io_in_a_sig(near_path_mods_0_io_in_a_sig),
    .io_in_b_sign(near_path_mods_0_io_in_b_sign),
    .io_in_b_sig(near_path_mods_0_io_in_b_sig),
    .io_in_need_shift_b(near_path_mods_0_io_in_need_shift_b),
    .io_in_rm(near_path_mods_0_io_in_rm),
    .io_out_result_sign(near_path_mods_0_io_out_result_sign),
    .io_out_result_exp(near_path_mods_0_io_out_result_exp),
    .io_out_result_sig(near_path_mods_0_io_out_result_sig),
    .io_out_sig_is_zero(near_path_mods_0_io_out_sig_is_zero),
    .io_out_a_lt_b(near_path_mods_0_io_out_a_lt_b),
    .io_out_tininess(near_path_mods_0_io_out_tininess)
  );
  NearPath near_path_mods_1 ( // @[FADD.scala 300:27]
    .io_in_a_sign(near_path_mods_1_io_in_a_sign),
    .io_in_a_exp(near_path_mods_1_io_in_a_exp),
    .io_in_a_sig(near_path_mods_1_io_in_a_sig),
    .io_in_b_sign(near_path_mods_1_io_in_b_sign),
    .io_in_b_sig(near_path_mods_1_io_in_b_sig),
    .io_in_need_shift_b(near_path_mods_1_io_in_need_shift_b),
    .io_in_rm(near_path_mods_1_io_in_rm),
    .io_out_result_sign(near_path_mods_1_io_out_result_sign),
    .io_out_result_exp(near_path_mods_1_io_out_result_exp),
    .io_out_result_sig(near_path_mods_1_io_out_result_sig),
    .io_out_sig_is_zero(near_path_mods_1_io_out_sig_is_zero),
    .io_out_a_lt_b(near_path_mods_1_io_out_a_lt_b),
    .io_out_tininess(near_path_mods_1_io_out_tininess)
  );
  RoundingUnit near_path_rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(near_path_rounder_io_in),
    .io_roundIn(near_path_rounder_io_roundIn),
    .io_stickyIn(near_path_rounder_io_stickyIn),
    .io_signIn(near_path_rounder_io_signIn),
    .io_rm(near_path_rounder_io_rm),
    .io_out(near_path_rounder_io_out),
    .io_inexact(near_path_rounder_io_inexact),
    .io_cout(near_path_rounder_io_cout)
  );
  assign io_result = special_case_happen ? special_path_result : _io_result_T_2; // @[FADD.scala 370:19]
  assign io_fflags = special_case_happen ? special_path_fflags : common_fflags; // @[FADD.scala 379:19]
  assign far_path_mods_0_io_in_a_sign = ~need_swap ? fp_a_sign : fp_b_sign; // @[FADD.scala 240:10]
  assign far_path_mods_0_io_in_a_exp = ~need_swap ? raw_a_exp : raw_b_exp; // @[FADD.scala 240:10]
  assign far_path_mods_0_io_in_a_sig = ~need_swap ? raw_a_sig : raw_b_sig; // @[FADD.scala 240:10]
  assign far_path_mods_0_io_in_b_sig = _T ? raw_b_sig : raw_a_sig; // @[FADD.scala 241:10]
  assign far_path_mods_0_io_in_expDiff = _T_5[7:0]; // @[FADD.scala 250:28]
  assign far_path_mods_0_io_in_effSub = fp_a_sign ^ fp_b_sign; // @[FADD.scala 197:28]
  assign far_path_mods_0_io_in_smallAdd = decode_a__expIsZero & decode_b__expIsZero; // @[FADD.scala 199:38]
  assign far_path_mods_0_io_in_rm = io_rm; // @[FADD.scala 253:23]
  assign far_path_rounder_io_in = far_path_mods_0_io_out_result_sig[24:2]; // @[RoundingUnit.scala 45:33]
  assign far_path_rounder_io_roundIn = far_path_mods_0_io_out_result_sig[1]; // @[RoundingUnit.scala 46:50]
  assign far_path_rounder_io_stickyIn = |far_path_mods_0_io_out_result_sig[0]; // @[RoundingUnit.scala 47:54]
  assign far_path_rounder_io_signIn = far_path_mods_0_io_out_result_sign; // @[RoundingUnit.scala 49:23]
  assign far_path_rounder_io_rm = io_rm; // @[RoundingUnit.scala 48:19]
  assign near_path_mods_0_io_in_a_sign = io_a[31]; // @[package.scala 59:19]
  assign near_path_mods_0_io_in_a_exp = fp_a_exp | _GEN_0; // @[package.scala 83:27]
  assign near_path_mods_0_io_in_a_sig = {decode_a_expNotZero,fp_a_sig}; // @[Cat.scala 31:58]
  assign near_path_mods_0_io_in_b_sign = io_b[31]; // @[package.scala 59:19]
  assign near_path_mods_0_io_in_b_sig = {decode_b_expNotZero,fp_b_sig}; // @[Cat.scala 31:58]
  assign near_path_mods_0_io_in_need_shift_b = raw_a_exp[1:0] != raw_b_exp[1:0]; // @[FADD.scala 293:43]
  assign near_path_mods_0_io_in_rm = io_rm; // @[FADD.scala 304:24]
  assign near_path_mods_1_io_in_a_sign = io_b[31]; // @[package.scala 59:19]
  assign near_path_mods_1_io_in_a_exp = fp_b_exp | _GEN_1; // @[package.scala 83:27]
  assign near_path_mods_1_io_in_a_sig = {decode_b_expNotZero,fp_b_sig}; // @[Cat.scala 31:58]
  assign near_path_mods_1_io_in_b_sign = io_a[31]; // @[package.scala 59:19]
  assign near_path_mods_1_io_in_b_sig = {decode_a_expNotZero,fp_a_sig}; // @[Cat.scala 31:58]
  assign near_path_mods_1_io_in_need_shift_b = raw_a_exp[1:0] != raw_b_exp[1:0]; // @[FADD.scala 293:43]
  assign near_path_mods_1_io_in_rm = io_rm; // @[FADD.scala 304:24]
  assign near_path_rounder_io_in = near_path_out_result_sig[24:2]; // @[RoundingUnit.scala 45:33]
  assign near_path_rounder_io_roundIn = near_path_out_result_sig[1]; // @[RoundingUnit.scala 46:50]
  assign near_path_rounder_io_stickyIn = |near_path_out_result_sig[0]; // @[RoundingUnit.scala 47:54]
  assign near_path_rounder_io_signIn = _near_path_out_T_2 ? near_path_mods_1_io_out_result_sign :
    near_path_mods_0_io_out_result_sign; // @[FADD.scala 309:26]
  assign near_path_rounder_io_rm = io_rm; // @[RoundingUnit.scala 48:19]
endmodule
module FADD(
  input         clock,
  input         reset,
  input  [31:0] io_a,
  input  [31:0] io_b,
  input  [2:0]  io_rm,
  output [31:0] io_result,
  output [4:0]  io_fflags
);
  wire [31:0] module__io_a; // @[FADD.scala 391:22]
  wire [31:0] module__io_b; // @[FADD.scala 391:22]
  wire [2:0] module__io_rm; // @[FADD.scala 391:22]
  wire [31:0] module__io_result; // @[FADD.scala 391:22]
  wire [4:0] module__io_fflags; // @[FADD.scala 391:22]
  FCMA_ADD module_ ( // @[FADD.scala 391:22]
    .io_a(module__io_a),
    .io_b(module__io_b),
    .io_rm(module__io_rm),
    .io_result(module__io_result),
    .io_fflags(module__io_fflags)
  );
  assign io_result = module__io_result; // @[FADD.scala 398:13]
  assign io_fflags = module__io_fflags; // @[FADD.scala 399:13]
  assign module__io_a = io_a; // @[FADD.scala 393:15]
  assign module__io_b = io_b; // @[FADD.scala 394:15]
  assign module__io_rm = io_rm; // @[FADD.scala 395:16]
endmodule
