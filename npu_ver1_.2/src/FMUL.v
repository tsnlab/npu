module CLZM(
  input  [49:0] io_in,
  output [5:0]  io_out
);
  wire [5:0] _io_out_T_50 = io_in[1] ? 6'h30 : 6'h31; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_51 = io_in[2] ? 6'h2f : _io_out_T_50; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_52 = io_in[3] ? 6'h2e : _io_out_T_51; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_53 = io_in[4] ? 6'h2d : _io_out_T_52; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_54 = io_in[5] ? 6'h2c : _io_out_T_53; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_55 = io_in[6] ? 6'h2b : _io_out_T_54; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_56 = io_in[7] ? 6'h2a : _io_out_T_55; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_57 = io_in[8] ? 6'h29 : _io_out_T_56; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_58 = io_in[9] ? 6'h28 : _io_out_T_57; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_59 = io_in[10] ? 6'h27 : _io_out_T_58; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_60 = io_in[11] ? 6'h26 : _io_out_T_59; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_61 = io_in[12] ? 6'h25 : _io_out_T_60; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_62 = io_in[13] ? 6'h24 : _io_out_T_61; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_63 = io_in[14] ? 6'h23 : _io_out_T_62; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_64 = io_in[15] ? 6'h22 : _io_out_T_63; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_65 = io_in[16] ? 6'h21 : _io_out_T_64; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_66 = io_in[17] ? 6'h20 : _io_out_T_65; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_67 = io_in[18] ? 6'h1f : _io_out_T_66; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_68 = io_in[19] ? 6'h1e : _io_out_T_67; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_69 = io_in[20] ? 6'h1d : _io_out_T_68; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_70 = io_in[21] ? 6'h1c : _io_out_T_69; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_71 = io_in[22] ? 6'h1b : _io_out_T_70; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_72 = io_in[23] ? 6'h1a : _io_out_T_71; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_73 = io_in[24] ? 6'h19 : _io_out_T_72; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_74 = io_in[25] ? 6'h18 : _io_out_T_73; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_75 = io_in[26] ? 6'h17 : _io_out_T_74; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_76 = io_in[27] ? 6'h16 : _io_out_T_75; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_77 = io_in[28] ? 6'h15 : _io_out_T_76; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_78 = io_in[29] ? 6'h14 : _io_out_T_77; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_79 = io_in[30] ? 6'h13 : _io_out_T_78; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_80 = io_in[31] ? 6'h12 : _io_out_T_79; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_81 = io_in[32] ? 6'h11 : _io_out_T_80; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_82 = io_in[33] ? 6'h10 : _io_out_T_81; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_83 = io_in[34] ? 6'hf : _io_out_T_82; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_84 = io_in[35] ? 6'he : _io_out_T_83; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_85 = io_in[36] ? 6'hd : _io_out_T_84; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_86 = io_in[37] ? 6'hc : _io_out_T_85; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_87 = io_in[38] ? 6'hb : _io_out_T_86; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_88 = io_in[39] ? 6'ha : _io_out_T_87; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_89 = io_in[40] ? 6'h9 : _io_out_T_88; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_90 = io_in[41] ? 6'h8 : _io_out_T_89; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_91 = io_in[42] ? 6'h7 : _io_out_T_90; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_92 = io_in[43] ? 6'h6 : _io_out_T_91; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_93 = io_in[44] ? 6'h5 : _io_out_T_92; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_94 = io_in[45] ? 6'h4 : _io_out_T_93; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_95 = io_in[46] ? 6'h3 : _io_out_T_94; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_96 = io_in[47] ? 6'h2 : _io_out_T_95; // @[Mux.scala 47:70]
  wire [5:0] _io_out_T_97 = io_in[48] ? 6'h1 : _io_out_T_96; // @[Mux.scala 47:70]
  assign io_out = io_in[49] ? 6'h0 : _io_out_T_97; // @[Mux.scala 47:70]
endmodule

module RoundingUnitM(
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
module FMUL(
  input         clock,
  input         reset,
  input  [31:0] io_a,
  input  [31:0] io_b,
  input  [2:0]  io_rm,
  output [31:0] io_result,
  output [4:0]  io_fflags,
  output        io_to_fadd_fp_prod_sign,
  output [7:0]  io_to_fadd_fp_prod_exp,
  output [46:0] io_to_fadd_fp_prod_sig,
  output        io_to_fadd_inter_flags_isNaN,
  output        io_to_fadd_inter_flags_isInf,
  output        io_to_fadd_inter_flags_isInv,
  output        io_to_fadd_inter_flags_overflow
);
  wire [49:0] lzc_clz_io_in; // @[CLZ.scala 22:21]
  wire [5:0] lzc_clz_io_out; // @[CLZ.scala 22:21]
  wire [22:0] tininess_rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] tininess_rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] tininess_rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  tininess_rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire [22:0] rounder_io_in; // @[RoundingUnit.scala 44:25]
  wire  rounder_io_roundIn; // @[RoundingUnit.scala 44:25]
  wire  rounder_io_stickyIn; // @[RoundingUnit.scala 44:25]
  wire  rounder_io_signIn; // @[RoundingUnit.scala 44:25]
  wire [2:0] rounder_io_rm; // @[RoundingUnit.scala 44:25]
  wire [22:0] rounder_io_out; // @[RoundingUnit.scala 44:25]
  wire  rounder_io_inexact; // @[RoundingUnit.scala 44:25]
  wire  rounder_io_cout; // @[RoundingUnit.scala 44:25]
  wire  fp_a_sign = io_a[31]; // @[package.scala 59:19]
  wire [7:0] fp_a_exp = io_a[30:23]; // @[package.scala 60:18]
  wire [22:0] fp_a_sig = io_a[22:0]; // @[package.scala 61:18]
  wire  fp_b_sign = io_b[31]; // @[package.scala 59:19]
  wire [7:0] fp_b_exp = io_b[30:23]; // @[package.scala 60:18]
  wire [22:0] fp_b_sig = io_b[22:0]; // @[package.scala 61:18]
  wire  expNotZero = |fp_a_exp; // @[package.scala 32:31]
  wire  expIsOnes = &fp_a_exp; // @[package.scala 33:31]
  wire  sigNotZero = |fp_a_sig; // @[package.scala 34:31]
  wire  decode_a_expIsZero = ~expNotZero; // @[package.scala 37:27]
  wire  decode_a_sigIsZero = ~sigNotZero; // @[package.scala 40:27]
  wire  decode_a_isInf = expIsOnes & decode_a_sigIsZero; // @[package.scala 42:40]
  wire  decode_a_isZero = decode_a_expIsZero & decode_a_sigIsZero; // @[package.scala 43:41]
  wire  decode_a_isNaN = expIsOnes & sigNotZero; // @[package.scala 44:40]
  wire  decode_a_isSNaN = decode_a_isNaN & ~fp_a_sig[22]; // @[package.scala 45:37]
  wire  expNotZero_1 = |fp_b_exp; // @[package.scala 32:31]
  wire  expIsOnes_1 = &fp_b_exp; // @[package.scala 33:31]
  wire  sigNotZero_1 = |fp_b_sig; // @[package.scala 34:31]
  wire  decode_b_expIsZero = ~expNotZero_1; // @[package.scala 37:27]
  wire  decode_b_sigIsZero = ~sigNotZero_1; // @[package.scala 40:27]
  wire  decode_b_isInf = expIsOnes_1 & decode_b_sigIsZero; // @[package.scala 42:40]
  wire  decode_b_isZero = decode_b_expIsZero & decode_b_sigIsZero; // @[package.scala 43:41]
  wire  decode_b_isNaN = expIsOnes_1 & sigNotZero_1; // @[package.scala 44:40]
  wire  decode_b_isSNaN = decode_b_isNaN & ~fp_b_sig[22]; // @[package.scala 45:37]
  wire [7:0] _GEN_0 = {{7'd0}, decode_a_expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_a_exp = fp_a_exp | _GEN_0; // @[package.scala 83:27]
  wire [23:0] raw_a_sig = {expNotZero,fp_a_sig}; // @[Cat.scala 31:58]
  wire [7:0] _GEN_1 = {{7'd0}, decode_b_expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_b_exp = fp_b_exp | _GEN_1; // @[package.scala 83:27]
  wire [23:0] raw_b_sig = {expNotZero_1,fp_b_sig}; // @[Cat.scala 31:58]
  wire  prod_sign = fp_a_sign ^ fp_b_sign; // @[FMUL.scala 35:29]
  wire [8:0] exp_sum = raw_a_exp + raw_b_exp; // @[FMUL.scala 52:27]
  wire [8:0] prod_exp = exp_sum - 9'h64; // @[FMUL.scala 53:26]
  wire [9:0] _shift_lim_sub_T = {1'h0,exp_sum}; // @[Cat.scala 31:58]
  wire [9:0] shift_lim_sub = _shift_lim_sub_T - 10'h65; // @[FMUL.scala 55:46]
  wire  prod_exp_uf = shift_lim_sub[9]; // @[FMUL.scala 56:39]
  wire [8:0] shift_lim = shift_lim_sub[8:0]; // @[FMUL.scala 57:37]
  wire  prod_exp_ov = exp_sum > 9'h17d; // @[FMUL.scala 59:29]
  wire [23:0] subnormal_sig = decode_a_expIsZero ? raw_a_sig : raw_b_sig; // @[FMUL.scala 62:26]
  wire [8:0] _GEN_2 = {{3'd0}, lzc_clz_io_out}; // @[FMUL.scala 64:30]
  wire  exceed_lim = shift_lim <= _GEN_2; // @[FMUL.scala 64:30]
  wire [8:0] _shift_amt_T = exceed_lim ? shift_lim : {{3'd0}, lzc_clz_io_out}; // @[FMUL.scala 65:44]
  wire [8:0] shift_amt = prod_exp_uf ? 9'h0 : _shift_amt_T; // @[FMUL.scala 65:22]
  wire [47:0] prod = raw_a_sig * raw_b_sig; // @[FMUL.scala 67:24]
  wire [73:0] sig_shifter_in = {26'h0,prod}; // @[Cat.scala 31:58]
  wire [584:0] _GEN_5 = {{511'd0}, sig_shifter_in}; // @[FMUL.scala 69:41]
  wire [584:0] _sig_shifted_raw_T = _GEN_5 << shift_amt; // @[FMUL.scala 69:41]
  wire [73:0] sig_shifted_raw = _sig_shifted_raw_T[73:0]; // @[FMUL.scala 69:54]
  wire [8:0] exp_shifted = prod_exp - shift_amt; // @[FMUL.scala 70:30]
  wire  exp_is_subnormal = (exceed_lim | prod_exp_uf) & ~sig_shifted_raw[73]; // @[FMUL.scala 71:54]
  wire  no_extra_shift = sig_shifted_raw[73] | exp_is_subnormal; // @[FMUL.scala 72:57]
  wire [8:0] _exp_pre_round_T_1 = exp_shifted - 9'h1; // @[FMUL.scala 74:95]
  wire [8:0] _exp_pre_round_T_2 = no_extra_shift ? exp_shifted : _exp_pre_round_T_1; // @[FMUL.scala 74:53]
  wire [8:0] exp_pre_round = exp_is_subnormal ? 9'h0 : _exp_pre_round_T_2; // @[FMUL.scala 74:26]
  wire [73:0] _sig_shifted_T_1 = {sig_shifted_raw[72:0],1'h0}; // @[Cat.scala 31:58]
  wire [73:0] sig_shifted = no_extra_shift ? sig_shifted_raw : _sig_shifted_T_1; // @[FMUL.scala 75:24]
  wire  _tininess_T_5 = sig_shifted[73:72] == 2'h1 & ~tininess_rounder_io_cout; // @[FMUL.scala 85:43]
  wire  tininess = sig_shifted[73:72] == 2'h0 | _tininess_T_5; // @[FMUL.scala 84:55]
  wire [8:0] _GEN_3 = {{8'd0}, rounder_io_cout}; // @[FMUL.scala 94:37]
  wire [8:0] exp_rounded = _GEN_3 + exp_pre_round; // @[FMUL.scala 94:37]
  wire  _common_of_T = exp_pre_round == 9'hfe; // @[FMUL.scala 99:19]
  wire  _common_of_T_1 = exp_pre_round == 9'hff; // @[FMUL.scala 100:19]
  wire  _common_of_T_2 = rounder_io_cout ? _common_of_T : _common_of_T_1; // @[FMUL.scala 97:22]
  wire  common_of = _common_of_T_2 | prod_exp_ov; // @[FMUL.scala 101:5]
  wire  common_ix = rounder_io_inexact | common_of; // @[FMUL.scala 102:38]
  wire  common_uf = tininess & common_ix; // @[FMUL.scala 103:28]
  wire  rmin = io_rm == 3'h1 | io_rm == 3'h2 & ~prod_sign | io_rm == 3'h3 & prod_sign; // @[RoundingUnit.scala 54:41]
  wire [7:0] of_exp = rmin ? 8'hfe : 8'hff; // @[FMUL.scala 107:19]
  wire [7:0] common_exp = common_of ? of_exp : exp_rounded[7:0]; // @[FMUL.scala 111:23]
  wire [22:0] _common_sig_T_1 = rmin ? 23'h7fffff : 23'h0; // @[FMUL.scala 118:8]
  wire [22:0] common_sig = common_of ? _common_sig_T_1 : rounder_io_out; // @[FMUL.scala 116:23]
  wire [31:0] common_result = {prod_sign,common_exp,common_sig}; // @[Cat.scala 31:58]
  wire [4:0] common_fflags = {2'h0,common_of,common_uf,common_ix}; // @[Cat.scala 31:58]
  wire  hasZero = decode_a_isZero | decode_b_isZero; // @[FMUL.scala 129:33]
  wire  hasNaN = decode_a_isNaN | decode_b_isNaN; // @[FMUL.scala 130:31]
  wire  hasSNaN = decode_a_isSNaN | decode_b_isSNaN; // @[FMUL.scala 131:33]
  wire  hasInf = decode_a_isInf | decode_b_isInf; // @[FMUL.scala 132:31]
  wire  special_case_happen = hasZero | hasNaN | hasInf; // @[FMUL.scala 133:47]
  wire  zero_mul_inf = hasZero & hasInf; // @[FMUL.scala 135:30]
  wire  nan_result = hasNaN | zero_mul_inf; // @[FMUL.scala 136:27]
  wire  special_iv = hasSNaN | zero_mul_inf; // @[FMUL.scala 140:28]
  wire [31:0] _special_result_T_2 = {prod_sign,8'hff,23'h0}; // @[Cat.scala 31:58]
  wire [31:0] _special_result_T_3 = {prod_sign,31'h0}; // @[Cat.scala 31:58]
  wire [31:0] _special_result_T_4 = hasInf ? _special_result_T_2 : _special_result_T_3; // @[FMUL.scala 145:8]
  wire [31:0] special_result = nan_result ? 32'h7fc00000 : _special_result_T_4; // @[FMUL.scala 143:27]
  wire [4:0] special_fflags = {special_iv,1'h0,1'h0,2'h0}; // @[Cat.scala 31:58]
  wire [8:0] _io_to_fadd_fp_prod_exp_T = hasZero ? 9'h0 : exp_pre_round; // @[FMUL.scala 156:32]
  wire [46:0] _GEN_4 = {{46'd0}, |sig_shifted[25:0]}; // @[FMUL.scala 159:49]
  wire [46:0] _io_to_fadd_fp_prod_sig_T_4 = sig_shifted[72:26] | _GEN_4; // @[FMUL.scala 159:49]
  CLZM lzc_clz ( // @[CLZ.scala 22:21]
    .io_in(lzc_clz_io_in),
    .io_out(lzc_clz_io_out)
  );
  RoundingUnitM tininess_rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(tininess_rounder_io_in),
    .io_roundIn(tininess_rounder_io_roundIn),
    .io_stickyIn(tininess_rounder_io_stickyIn),
    .io_signIn(tininess_rounder_io_signIn),
    .io_rm(tininess_rounder_io_rm),
    .io_out(tininess_rounder_io_out),
    .io_inexact(tininess_rounder_io_inexact),
    .io_cout(tininess_rounder_io_cout)
  );
  RoundingUnitM rounder ( // @[RoundingUnit.scala 44:25]
    .io_in(rounder_io_in),
    .io_roundIn(rounder_io_roundIn),
    .io_stickyIn(rounder_io_stickyIn),
    .io_signIn(rounder_io_signIn),
    .io_rm(rounder_io_rm),
    .io_out(rounder_io_out),
    .io_inexact(rounder_io_inexact),
    .io_cout(rounder_io_cout)
  );
  assign io_result = special_case_happen ? special_result : common_result; // @[FMUL.scala 152:19]
  assign io_fflags = special_case_happen ? special_fflags : common_fflags; // @[FMUL.scala 153:19]
  assign io_to_fadd_fp_prod_sign = fp_a_sign ^ fp_b_sign; // @[FMUL.scala 35:29]
  assign io_to_fadd_fp_prod_exp = _io_to_fadd_fp_prod_exp_T[7:0]; // @[FMUL.scala 156:26]
  assign io_to_fadd_fp_prod_sig = hasZero ? 47'h0 : _io_to_fadd_fp_prod_sig_T_4; // @[FMUL.scala 157:32]
  assign io_to_fadd_inter_flags_isNaN = hasNaN | zero_mul_inf; // @[FMUL.scala 136:27]
  assign io_to_fadd_inter_flags_isInf = hasInf & ~nan_result; // @[FMUL.scala 162:42]
  assign io_to_fadd_inter_flags_isInv = hasSNaN | zero_mul_inf; // @[FMUL.scala 140:28]
  assign io_to_fadd_inter_flags_overflow = exp_pre_round > 9'hff; // @[FMUL.scala 164:52]
  assign lzc_clz_io_in = {26'h0,subnormal_sig}; // @[Cat.scala 31:58]
  assign tininess_rounder_io_in = sig_shifted[71:49]; // @[RoundingUnit.scala 45:33]
  assign tininess_rounder_io_roundIn = sig_shifted[48]; // @[RoundingUnit.scala 46:50]
  assign tininess_rounder_io_stickyIn = |sig_shifted[47:0]; // @[RoundingUnit.scala 47:54]
  assign tininess_rounder_io_signIn = fp_a_sign ^ fp_b_sign; // @[FMUL.scala 35:29]
  assign tininess_rounder_io_rm = io_rm; // @[RoundingUnit.scala 48:19]
  assign rounder_io_in = sig_shifted[72:50]; // @[RoundingUnit.scala 45:33]
  assign rounder_io_roundIn = sig_shifted[49]; // @[RoundingUnit.scala 46:50]
  assign rounder_io_stickyIn = |sig_shifted[48:0]; // @[RoundingUnit.scala 47:54]
  assign rounder_io_signIn = fp_a_sign ^ fp_b_sign; // @[FMUL.scala 35:29]
  assign rounder_io_rm = io_rm; // @[RoundingUnit.scala 48:19]
endmodule
