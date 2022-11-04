module CLZD(
  input  [23:0] io_in,
  output [4:0]  io_out
);
  wire [4:0] _io_out_T_24 = io_in[1] ? 5'h16 : 5'h17; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_25 = io_in[2] ? 5'h15 : _io_out_T_24; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_26 = io_in[3] ? 5'h14 : _io_out_T_25; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_27 = io_in[4] ? 5'h13 : _io_out_T_26; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_28 = io_in[5] ? 5'h12 : _io_out_T_27; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_29 = io_in[6] ? 5'h11 : _io_out_T_28; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_30 = io_in[7] ? 5'h10 : _io_out_T_29; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_31 = io_in[8] ? 5'hf : _io_out_T_30; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_32 = io_in[9] ? 5'he : _io_out_T_31; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_33 = io_in[10] ? 5'hd : _io_out_T_32; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_34 = io_in[11] ? 5'hc : _io_out_T_33; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_35 = io_in[12] ? 5'hb : _io_out_T_34; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_36 = io_in[13] ? 5'ha : _io_out_T_35; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_37 = io_in[14] ? 5'h9 : _io_out_T_36; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_38 = io_in[15] ? 5'h8 : _io_out_T_37; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_39 = io_in[16] ? 5'h7 : _io_out_T_38; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_40 = io_in[17] ? 5'h6 : _io_out_T_39; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_41 = io_in[18] ? 5'h5 : _io_out_T_40; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_42 = io_in[19] ? 5'h4 : _io_out_T_41; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_43 = io_in[20] ? 5'h3 : _io_out_T_42; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_44 = io_in[21] ? 5'h2 : _io_out_T_43; // @[Mux.scala 47:70]
  wire [4:0] _io_out_T_45 = io_in[22] ? 5'h1 : _io_out_T_44; // @[Mux.scala 47:70]
  assign io_out = io_in[23] ? 5'h0 : _io_out_T_45; // @[Mux.scala 47:70]
endmodule
module CSA3_2(
  input  [9:0] io_in_0,
  input  [9:0] io_in_1,
  input  [9:0] io_in_2,
  output [9:0] io_out_0,
  output [9:0] io_out_1
);
  wire  a = io_in_0[0]; // @[FDIV.scala 669:32]
  wire  b = io_in_1[0]; // @[FDIV.scala 669:45]
  wire  cin = io_in_2[0]; // @[FDIV.scala 669:58]
  wire  a_xor_b = a ^ b; // @[FDIV.scala 670:21]
  wire  a_and_b = a & b; // @[FDIV.scala 671:21]
  wire  sum = a_xor_b ^ cin; // @[FDIV.scala 672:23]
  wire  cout = a_and_b | a_xor_b & cin; // @[FDIV.scala 673:24]
  wire [1:0] temp_0 = {cout,sum}; // @[Cat.scala 31:58]
  wire  a_1 = io_in_0[1]; // @[FDIV.scala 669:32]
  wire  b_1 = io_in_1[1]; // @[FDIV.scala 669:45]
  wire  cin_1 = io_in_2[1]; // @[FDIV.scala 669:58]
  wire  a_xor_b_1 = a_1 ^ b_1; // @[FDIV.scala 670:21]
  wire  a_and_b_1 = a_1 & b_1; // @[FDIV.scala 671:21]
  wire  sum_1 = a_xor_b_1 ^ cin_1; // @[FDIV.scala 672:23]
  wire  cout_1 = a_and_b_1 | a_xor_b_1 & cin_1; // @[FDIV.scala 673:24]
  wire [1:0] temp_1 = {cout_1,sum_1}; // @[Cat.scala 31:58]
  wire  a_2 = io_in_0[2]; // @[FDIV.scala 669:32]
  wire  b_2 = io_in_1[2]; // @[FDIV.scala 669:45]
  wire  cin_2 = io_in_2[2]; // @[FDIV.scala 669:58]
  wire  a_xor_b_2 = a_2 ^ b_2; // @[FDIV.scala 670:21]
  wire  a_and_b_2 = a_2 & b_2; // @[FDIV.scala 671:21]
  wire  sum_2 = a_xor_b_2 ^ cin_2; // @[FDIV.scala 672:23]
  wire  cout_2 = a_and_b_2 | a_xor_b_2 & cin_2; // @[FDIV.scala 673:24]
  wire [1:0] temp_2 = {cout_2,sum_2}; // @[Cat.scala 31:58]
  wire  a_3 = io_in_0[3]; // @[FDIV.scala 669:32]
  wire  b_3 = io_in_1[3]; // @[FDIV.scala 669:45]
  wire  cin_3 = io_in_2[3]; // @[FDIV.scala 669:58]
  wire  a_xor_b_3 = a_3 ^ b_3; // @[FDIV.scala 670:21]
  wire  a_and_b_3 = a_3 & b_3; // @[FDIV.scala 671:21]
  wire  sum_3 = a_xor_b_3 ^ cin_3; // @[FDIV.scala 672:23]
  wire  cout_3 = a_and_b_3 | a_xor_b_3 & cin_3; // @[FDIV.scala 673:24]
  wire [1:0] temp_3 = {cout_3,sum_3}; // @[Cat.scala 31:58]
  wire  a_4 = io_in_0[4]; // @[FDIV.scala 669:32]
  wire  b_4 = io_in_1[4]; // @[FDIV.scala 669:45]
  wire  cin_4 = io_in_2[4]; // @[FDIV.scala 669:58]
  wire  a_xor_b_4 = a_4 ^ b_4; // @[FDIV.scala 670:21]
  wire  a_and_b_4 = a_4 & b_4; // @[FDIV.scala 671:21]
  wire  sum_4 = a_xor_b_4 ^ cin_4; // @[FDIV.scala 672:23]
  wire  cout_4 = a_and_b_4 | a_xor_b_4 & cin_4; // @[FDIV.scala 673:24]
  wire [1:0] temp_4 = {cout_4,sum_4}; // @[Cat.scala 31:58]
  wire  a_5 = io_in_0[5]; // @[FDIV.scala 669:32]
  wire  b_5 = io_in_1[5]; // @[FDIV.scala 669:45]
  wire  cin_5 = io_in_2[5]; // @[FDIV.scala 669:58]
  wire  a_xor_b_5 = a_5 ^ b_5; // @[FDIV.scala 670:21]
  wire  a_and_b_5 = a_5 & b_5; // @[FDIV.scala 671:21]
  wire  sum_5 = a_xor_b_5 ^ cin_5; // @[FDIV.scala 672:23]
  wire  cout_5 = a_and_b_5 | a_xor_b_5 & cin_5; // @[FDIV.scala 673:24]
  wire [1:0] temp_5 = {cout_5,sum_5}; // @[Cat.scala 31:58]
  wire  a_6 = io_in_0[6]; // @[FDIV.scala 669:32]
  wire  b_6 = io_in_1[6]; // @[FDIV.scala 669:45]
  wire  cin_6 = io_in_2[6]; // @[FDIV.scala 669:58]
  wire  a_xor_b_6 = a_6 ^ b_6; // @[FDIV.scala 670:21]
  wire  a_and_b_6 = a_6 & b_6; // @[FDIV.scala 671:21]
  wire  sum_6 = a_xor_b_6 ^ cin_6; // @[FDIV.scala 672:23]
  wire  cout_6 = a_and_b_6 | a_xor_b_6 & cin_6; // @[FDIV.scala 673:24]
  wire [1:0] temp_6 = {cout_6,sum_6}; // @[Cat.scala 31:58]
  wire  a_7 = io_in_0[7]; // @[FDIV.scala 669:32]
  wire  b_7 = io_in_1[7]; // @[FDIV.scala 669:45]
  wire  cin_7 = io_in_2[7]; // @[FDIV.scala 669:58]
  wire  a_xor_b_7 = a_7 ^ b_7; // @[FDIV.scala 670:21]
  wire  a_and_b_7 = a_7 & b_7; // @[FDIV.scala 671:21]
  wire  sum_7 = a_xor_b_7 ^ cin_7; // @[FDIV.scala 672:23]
  wire  cout_7 = a_and_b_7 | a_xor_b_7 & cin_7; // @[FDIV.scala 673:24]
  wire [1:0] temp_7 = {cout_7,sum_7}; // @[Cat.scala 31:58]
  wire  a_8 = io_in_0[8]; // @[FDIV.scala 669:32]
  wire  b_8 = io_in_1[8]; // @[FDIV.scala 669:45]
  wire  cin_8 = io_in_2[8]; // @[FDIV.scala 669:58]
  wire  a_xor_b_8 = a_8 ^ b_8; // @[FDIV.scala 670:21]
  wire  a_and_b_8 = a_8 & b_8; // @[FDIV.scala 671:21]
  wire  sum_8 = a_xor_b_8 ^ cin_8; // @[FDIV.scala 672:23]
  wire  cout_8 = a_and_b_8 | a_xor_b_8 & cin_8; // @[FDIV.scala 673:24]
  wire [1:0] temp_8 = {cout_8,sum_8}; // @[Cat.scala 31:58]
  wire  a_9 = io_in_0[9]; // @[FDIV.scala 669:32]
  wire  b_9 = io_in_1[9]; // @[FDIV.scala 669:45]
  wire  cin_9 = io_in_2[9]; // @[FDIV.scala 669:58]
  wire  a_xor_b_9 = a_9 ^ b_9; // @[FDIV.scala 670:21]
  wire  a_and_b_9 = a_9 & b_9; // @[FDIV.scala 671:21]
  wire  sum_9 = a_xor_b_9 ^ cin_9; // @[FDIV.scala 672:23]
  wire  cout_9 = a_and_b_9 | a_xor_b_9 & cin_9; // @[FDIV.scala 673:24]
  wire [1:0] temp_9 = {cout_9,sum_9}; // @[Cat.scala 31:58]
  wire [4:0] io_out_0_lo = {temp_4[0],temp_3[0],temp_2[0],temp_1[0],temp_0[0]}; // @[Cat.scala 31:58]
  wire [4:0] io_out_0_hi = {temp_9[0],temp_8[0],temp_7[0],temp_6[0],temp_5[0]}; // @[Cat.scala 31:58]
  wire [4:0] io_out_1_lo = {temp_4[1],temp_3[1],temp_2[1],temp_1[1],temp_0[1]}; // @[Cat.scala 31:58]
  wire [4:0] io_out_1_hi = {temp_9[1],temp_8[1],temp_7[1],temp_6[1],temp_5[1]}; // @[Cat.scala 31:58]
  assign io_out_0 = {io_out_0_hi,io_out_0_lo}; // @[Cat.scala 31:58]
  assign io_out_1 = {io_out_1_hi,io_out_1_lo}; // @[Cat.scala 31:58]
endmodule
module CSA3_2_4(
  input  [27:0] io_in_0,
  input  [27:0] io_in_1,
  input  [27:0] io_in_2,
  output [27:0] io_out_0,
  output [27:0] io_out_1
);
  wire  a = io_in_0[0]; // @[FDIV.scala 669:32]
  wire  b = io_in_1[0]; // @[FDIV.scala 669:45]
  wire  cin = io_in_2[0]; // @[FDIV.scala 669:58]
  wire  a_xor_b = a ^ b; // @[FDIV.scala 670:21]
  wire  a_and_b = a & b; // @[FDIV.scala 671:21]
  wire  sum = a_xor_b ^ cin; // @[FDIV.scala 672:23]
  wire  cout = a_and_b | a_xor_b & cin; // @[FDIV.scala 673:24]
  wire [1:0] temp_0 = {cout,sum}; // @[Cat.scala 31:58]
  wire  a_1 = io_in_0[1]; // @[FDIV.scala 669:32]
  wire  b_1 = io_in_1[1]; // @[FDIV.scala 669:45]
  wire  cin_1 = io_in_2[1]; // @[FDIV.scala 669:58]
  wire  a_xor_b_1 = a_1 ^ b_1; // @[FDIV.scala 670:21]
  wire  a_and_b_1 = a_1 & b_1; // @[FDIV.scala 671:21]
  wire  sum_1 = a_xor_b_1 ^ cin_1; // @[FDIV.scala 672:23]
  wire  cout_1 = a_and_b_1 | a_xor_b_1 & cin_1; // @[FDIV.scala 673:24]
  wire [1:0] temp_1 = {cout_1,sum_1}; // @[Cat.scala 31:58]
  wire  a_2 = io_in_0[2]; // @[FDIV.scala 669:32]
  wire  b_2 = io_in_1[2]; // @[FDIV.scala 669:45]
  wire  cin_2 = io_in_2[2]; // @[FDIV.scala 669:58]
  wire  a_xor_b_2 = a_2 ^ b_2; // @[FDIV.scala 670:21]
  wire  a_and_b_2 = a_2 & b_2; // @[FDIV.scala 671:21]
  wire  sum_2 = a_xor_b_2 ^ cin_2; // @[FDIV.scala 672:23]
  wire  cout_2 = a_and_b_2 | a_xor_b_2 & cin_2; // @[FDIV.scala 673:24]
  wire [1:0] temp_2 = {cout_2,sum_2}; // @[Cat.scala 31:58]
  wire  a_3 = io_in_0[3]; // @[FDIV.scala 669:32]
  wire  b_3 = io_in_1[3]; // @[FDIV.scala 669:45]
  wire  cin_3 = io_in_2[3]; // @[FDIV.scala 669:58]
  wire  a_xor_b_3 = a_3 ^ b_3; // @[FDIV.scala 670:21]
  wire  a_and_b_3 = a_3 & b_3; // @[FDIV.scala 671:21]
  wire  sum_3 = a_xor_b_3 ^ cin_3; // @[FDIV.scala 672:23]
  wire  cout_3 = a_and_b_3 | a_xor_b_3 & cin_3; // @[FDIV.scala 673:24]
  wire [1:0] temp_3 = {cout_3,sum_3}; // @[Cat.scala 31:58]
  wire  a_4 = io_in_0[4]; // @[FDIV.scala 669:32]
  wire  b_4 = io_in_1[4]; // @[FDIV.scala 669:45]
  wire  cin_4 = io_in_2[4]; // @[FDIV.scala 669:58]
  wire  a_xor_b_4 = a_4 ^ b_4; // @[FDIV.scala 670:21]
  wire  a_and_b_4 = a_4 & b_4; // @[FDIV.scala 671:21]
  wire  sum_4 = a_xor_b_4 ^ cin_4; // @[FDIV.scala 672:23]
  wire  cout_4 = a_and_b_4 | a_xor_b_4 & cin_4; // @[FDIV.scala 673:24]
  wire [1:0] temp_4 = {cout_4,sum_4}; // @[Cat.scala 31:58]
  wire  a_5 = io_in_0[5]; // @[FDIV.scala 669:32]
  wire  b_5 = io_in_1[5]; // @[FDIV.scala 669:45]
  wire  cin_5 = io_in_2[5]; // @[FDIV.scala 669:58]
  wire  a_xor_b_5 = a_5 ^ b_5; // @[FDIV.scala 670:21]
  wire  a_and_b_5 = a_5 & b_5; // @[FDIV.scala 671:21]
  wire  sum_5 = a_xor_b_5 ^ cin_5; // @[FDIV.scala 672:23]
  wire  cout_5 = a_and_b_5 | a_xor_b_5 & cin_5; // @[FDIV.scala 673:24]
  wire [1:0] temp_5 = {cout_5,sum_5}; // @[Cat.scala 31:58]
  wire  a_6 = io_in_0[6]; // @[FDIV.scala 669:32]
  wire  b_6 = io_in_1[6]; // @[FDIV.scala 669:45]
  wire  cin_6 = io_in_2[6]; // @[FDIV.scala 669:58]
  wire  a_xor_b_6 = a_6 ^ b_6; // @[FDIV.scala 670:21]
  wire  a_and_b_6 = a_6 & b_6; // @[FDIV.scala 671:21]
  wire  sum_6 = a_xor_b_6 ^ cin_6; // @[FDIV.scala 672:23]
  wire  cout_6 = a_and_b_6 | a_xor_b_6 & cin_6; // @[FDIV.scala 673:24]
  wire [1:0] temp_6 = {cout_6,sum_6}; // @[Cat.scala 31:58]
  wire  a_7 = io_in_0[7]; // @[FDIV.scala 669:32]
  wire  b_7 = io_in_1[7]; // @[FDIV.scala 669:45]
  wire  cin_7 = io_in_2[7]; // @[FDIV.scala 669:58]
  wire  a_xor_b_7 = a_7 ^ b_7; // @[FDIV.scala 670:21]
  wire  a_and_b_7 = a_7 & b_7; // @[FDIV.scala 671:21]
  wire  sum_7 = a_xor_b_7 ^ cin_7; // @[FDIV.scala 672:23]
  wire  cout_7 = a_and_b_7 | a_xor_b_7 & cin_7; // @[FDIV.scala 673:24]
  wire [1:0] temp_7 = {cout_7,sum_7}; // @[Cat.scala 31:58]
  wire  a_8 = io_in_0[8]; // @[FDIV.scala 669:32]
  wire  b_8 = io_in_1[8]; // @[FDIV.scala 669:45]
  wire  cin_8 = io_in_2[8]; // @[FDIV.scala 669:58]
  wire  a_xor_b_8 = a_8 ^ b_8; // @[FDIV.scala 670:21]
  wire  a_and_b_8 = a_8 & b_8; // @[FDIV.scala 671:21]
  wire  sum_8 = a_xor_b_8 ^ cin_8; // @[FDIV.scala 672:23]
  wire  cout_8 = a_and_b_8 | a_xor_b_8 & cin_8; // @[FDIV.scala 673:24]
  wire [1:0] temp_8 = {cout_8,sum_8}; // @[Cat.scala 31:58]
  wire  a_9 = io_in_0[9]; // @[FDIV.scala 669:32]
  wire  b_9 = io_in_1[9]; // @[FDIV.scala 669:45]
  wire  cin_9 = io_in_2[9]; // @[FDIV.scala 669:58]
  wire  a_xor_b_9 = a_9 ^ b_9; // @[FDIV.scala 670:21]
  wire  a_and_b_9 = a_9 & b_9; // @[FDIV.scala 671:21]
  wire  sum_9 = a_xor_b_9 ^ cin_9; // @[FDIV.scala 672:23]
  wire  cout_9 = a_and_b_9 | a_xor_b_9 & cin_9; // @[FDIV.scala 673:24]
  wire [1:0] temp_9 = {cout_9,sum_9}; // @[Cat.scala 31:58]
  wire  a_10 = io_in_0[10]; // @[FDIV.scala 669:32]
  wire  b_10 = io_in_1[10]; // @[FDIV.scala 669:45]
  wire  cin_10 = io_in_2[10]; // @[FDIV.scala 669:58]
  wire  a_xor_b_10 = a_10 ^ b_10; // @[FDIV.scala 670:21]
  wire  a_and_b_10 = a_10 & b_10; // @[FDIV.scala 671:21]
  wire  sum_10 = a_xor_b_10 ^ cin_10; // @[FDIV.scala 672:23]
  wire  cout_10 = a_and_b_10 | a_xor_b_10 & cin_10; // @[FDIV.scala 673:24]
  wire [1:0] temp_10 = {cout_10,sum_10}; // @[Cat.scala 31:58]
  wire  a_11 = io_in_0[11]; // @[FDIV.scala 669:32]
  wire  b_11 = io_in_1[11]; // @[FDIV.scala 669:45]
  wire  cin_11 = io_in_2[11]; // @[FDIV.scala 669:58]
  wire  a_xor_b_11 = a_11 ^ b_11; // @[FDIV.scala 670:21]
  wire  a_and_b_11 = a_11 & b_11; // @[FDIV.scala 671:21]
  wire  sum_11 = a_xor_b_11 ^ cin_11; // @[FDIV.scala 672:23]
  wire  cout_11 = a_and_b_11 | a_xor_b_11 & cin_11; // @[FDIV.scala 673:24]
  wire [1:0] temp_11 = {cout_11,sum_11}; // @[Cat.scala 31:58]
  wire  a_12 = io_in_0[12]; // @[FDIV.scala 669:32]
  wire  b_12 = io_in_1[12]; // @[FDIV.scala 669:45]
  wire  cin_12 = io_in_2[12]; // @[FDIV.scala 669:58]
  wire  a_xor_b_12 = a_12 ^ b_12; // @[FDIV.scala 670:21]
  wire  a_and_b_12 = a_12 & b_12; // @[FDIV.scala 671:21]
  wire  sum_12 = a_xor_b_12 ^ cin_12; // @[FDIV.scala 672:23]
  wire  cout_12 = a_and_b_12 | a_xor_b_12 & cin_12; // @[FDIV.scala 673:24]
  wire [1:0] temp_12 = {cout_12,sum_12}; // @[Cat.scala 31:58]
  wire  a_13 = io_in_0[13]; // @[FDIV.scala 669:32]
  wire  b_13 = io_in_1[13]; // @[FDIV.scala 669:45]
  wire  cin_13 = io_in_2[13]; // @[FDIV.scala 669:58]
  wire  a_xor_b_13 = a_13 ^ b_13; // @[FDIV.scala 670:21]
  wire  a_and_b_13 = a_13 & b_13; // @[FDIV.scala 671:21]
  wire  sum_13 = a_xor_b_13 ^ cin_13; // @[FDIV.scala 672:23]
  wire  cout_13 = a_and_b_13 | a_xor_b_13 & cin_13; // @[FDIV.scala 673:24]
  wire [1:0] temp_13 = {cout_13,sum_13}; // @[Cat.scala 31:58]
  wire  a_14 = io_in_0[14]; // @[FDIV.scala 669:32]
  wire  b_14 = io_in_1[14]; // @[FDIV.scala 669:45]
  wire  cin_14 = io_in_2[14]; // @[FDIV.scala 669:58]
  wire  a_xor_b_14 = a_14 ^ b_14; // @[FDIV.scala 670:21]
  wire  a_and_b_14 = a_14 & b_14; // @[FDIV.scala 671:21]
  wire  sum_14 = a_xor_b_14 ^ cin_14; // @[FDIV.scala 672:23]
  wire  cout_14 = a_and_b_14 | a_xor_b_14 & cin_14; // @[FDIV.scala 673:24]
  wire [1:0] temp_14 = {cout_14,sum_14}; // @[Cat.scala 31:58]
  wire  a_15 = io_in_0[15]; // @[FDIV.scala 669:32]
  wire  b_15 = io_in_1[15]; // @[FDIV.scala 669:45]
  wire  cin_15 = io_in_2[15]; // @[FDIV.scala 669:58]
  wire  a_xor_b_15 = a_15 ^ b_15; // @[FDIV.scala 670:21]
  wire  a_and_b_15 = a_15 & b_15; // @[FDIV.scala 671:21]
  wire  sum_15 = a_xor_b_15 ^ cin_15; // @[FDIV.scala 672:23]
  wire  cout_15 = a_and_b_15 | a_xor_b_15 & cin_15; // @[FDIV.scala 673:24]
  wire [1:0] temp_15 = {cout_15,sum_15}; // @[Cat.scala 31:58]
  wire  a_16 = io_in_0[16]; // @[FDIV.scala 669:32]
  wire  b_16 = io_in_1[16]; // @[FDIV.scala 669:45]
  wire  cin_16 = io_in_2[16]; // @[FDIV.scala 669:58]
  wire  a_xor_b_16 = a_16 ^ b_16; // @[FDIV.scala 670:21]
  wire  a_and_b_16 = a_16 & b_16; // @[FDIV.scala 671:21]
  wire  sum_16 = a_xor_b_16 ^ cin_16; // @[FDIV.scala 672:23]
  wire  cout_16 = a_and_b_16 | a_xor_b_16 & cin_16; // @[FDIV.scala 673:24]
  wire [1:0] temp_16 = {cout_16,sum_16}; // @[Cat.scala 31:58]
  wire  a_17 = io_in_0[17]; // @[FDIV.scala 669:32]
  wire  b_17 = io_in_1[17]; // @[FDIV.scala 669:45]
  wire  cin_17 = io_in_2[17]; // @[FDIV.scala 669:58]
  wire  a_xor_b_17 = a_17 ^ b_17; // @[FDIV.scala 670:21]
  wire  a_and_b_17 = a_17 & b_17; // @[FDIV.scala 671:21]
  wire  sum_17 = a_xor_b_17 ^ cin_17; // @[FDIV.scala 672:23]
  wire  cout_17 = a_and_b_17 | a_xor_b_17 & cin_17; // @[FDIV.scala 673:24]
  wire [1:0] temp_17 = {cout_17,sum_17}; // @[Cat.scala 31:58]
  wire  a_18 = io_in_0[18]; // @[FDIV.scala 669:32]
  wire  b_18 = io_in_1[18]; // @[FDIV.scala 669:45]
  wire  cin_18 = io_in_2[18]; // @[FDIV.scala 669:58]
  wire  a_xor_b_18 = a_18 ^ b_18; // @[FDIV.scala 670:21]
  wire  a_and_b_18 = a_18 & b_18; // @[FDIV.scala 671:21]
  wire  sum_18 = a_xor_b_18 ^ cin_18; // @[FDIV.scala 672:23]
  wire  cout_18 = a_and_b_18 | a_xor_b_18 & cin_18; // @[FDIV.scala 673:24]
  wire [1:0] temp_18 = {cout_18,sum_18}; // @[Cat.scala 31:58]
  wire  a_19 = io_in_0[19]; // @[FDIV.scala 669:32]
  wire  b_19 = io_in_1[19]; // @[FDIV.scala 669:45]
  wire  cin_19 = io_in_2[19]; // @[FDIV.scala 669:58]
  wire  a_xor_b_19 = a_19 ^ b_19; // @[FDIV.scala 670:21]
  wire  a_and_b_19 = a_19 & b_19; // @[FDIV.scala 671:21]
  wire  sum_19 = a_xor_b_19 ^ cin_19; // @[FDIV.scala 672:23]
  wire  cout_19 = a_and_b_19 | a_xor_b_19 & cin_19; // @[FDIV.scala 673:24]
  wire [1:0] temp_19 = {cout_19,sum_19}; // @[Cat.scala 31:58]
  wire  a_20 = io_in_0[20]; // @[FDIV.scala 669:32]
  wire  b_20 = io_in_1[20]; // @[FDIV.scala 669:45]
  wire  cin_20 = io_in_2[20]; // @[FDIV.scala 669:58]
  wire  a_xor_b_20 = a_20 ^ b_20; // @[FDIV.scala 670:21]
  wire  a_and_b_20 = a_20 & b_20; // @[FDIV.scala 671:21]
  wire  sum_20 = a_xor_b_20 ^ cin_20; // @[FDIV.scala 672:23]
  wire  cout_20 = a_and_b_20 | a_xor_b_20 & cin_20; // @[FDIV.scala 673:24]
  wire [1:0] temp_20 = {cout_20,sum_20}; // @[Cat.scala 31:58]
  wire  a_21 = io_in_0[21]; // @[FDIV.scala 669:32]
  wire  b_21 = io_in_1[21]; // @[FDIV.scala 669:45]
  wire  cin_21 = io_in_2[21]; // @[FDIV.scala 669:58]
  wire  a_xor_b_21 = a_21 ^ b_21; // @[FDIV.scala 670:21]
  wire  a_and_b_21 = a_21 & b_21; // @[FDIV.scala 671:21]
  wire  sum_21 = a_xor_b_21 ^ cin_21; // @[FDIV.scala 672:23]
  wire  cout_21 = a_and_b_21 | a_xor_b_21 & cin_21; // @[FDIV.scala 673:24]
  wire [1:0] temp_21 = {cout_21,sum_21}; // @[Cat.scala 31:58]
  wire  a_22 = io_in_0[22]; // @[FDIV.scala 669:32]
  wire  b_22 = io_in_1[22]; // @[FDIV.scala 669:45]
  wire  cin_22 = io_in_2[22]; // @[FDIV.scala 669:58]
  wire  a_xor_b_22 = a_22 ^ b_22; // @[FDIV.scala 670:21]
  wire  a_and_b_22 = a_22 & b_22; // @[FDIV.scala 671:21]
  wire  sum_22 = a_xor_b_22 ^ cin_22; // @[FDIV.scala 672:23]
  wire  cout_22 = a_and_b_22 | a_xor_b_22 & cin_22; // @[FDIV.scala 673:24]
  wire [1:0] temp_22 = {cout_22,sum_22}; // @[Cat.scala 31:58]
  wire  a_23 = io_in_0[23]; // @[FDIV.scala 669:32]
  wire  b_23 = io_in_1[23]; // @[FDIV.scala 669:45]
  wire  cin_23 = io_in_2[23]; // @[FDIV.scala 669:58]
  wire  a_xor_b_23 = a_23 ^ b_23; // @[FDIV.scala 670:21]
  wire  a_and_b_23 = a_23 & b_23; // @[FDIV.scala 671:21]
  wire  sum_23 = a_xor_b_23 ^ cin_23; // @[FDIV.scala 672:23]
  wire  cout_23 = a_and_b_23 | a_xor_b_23 & cin_23; // @[FDIV.scala 673:24]
  wire [1:0] temp_23 = {cout_23,sum_23}; // @[Cat.scala 31:58]
  wire  a_24 = io_in_0[24]; // @[FDIV.scala 669:32]
  wire  b_24 = io_in_1[24]; // @[FDIV.scala 669:45]
  wire  cin_24 = io_in_2[24]; // @[FDIV.scala 669:58]
  wire  a_xor_b_24 = a_24 ^ b_24; // @[FDIV.scala 670:21]
  wire  a_and_b_24 = a_24 & b_24; // @[FDIV.scala 671:21]
  wire  sum_24 = a_xor_b_24 ^ cin_24; // @[FDIV.scala 672:23]
  wire  cout_24 = a_and_b_24 | a_xor_b_24 & cin_24; // @[FDIV.scala 673:24]
  wire [1:0] temp_24 = {cout_24,sum_24}; // @[Cat.scala 31:58]
  wire  a_25 = io_in_0[25]; // @[FDIV.scala 669:32]
  wire  b_25 = io_in_1[25]; // @[FDIV.scala 669:45]
  wire  cin_25 = io_in_2[25]; // @[FDIV.scala 669:58]
  wire  a_xor_b_25 = a_25 ^ b_25; // @[FDIV.scala 670:21]
  wire  a_and_b_25 = a_25 & b_25; // @[FDIV.scala 671:21]
  wire  sum_25 = a_xor_b_25 ^ cin_25; // @[FDIV.scala 672:23]
  wire  cout_25 = a_and_b_25 | a_xor_b_25 & cin_25; // @[FDIV.scala 673:24]
  wire [1:0] temp_25 = {cout_25,sum_25}; // @[Cat.scala 31:58]
  wire  a_26 = io_in_0[26]; // @[FDIV.scala 669:32]
  wire  b_26 = io_in_1[26]; // @[FDIV.scala 669:45]
  wire  cin_26 = io_in_2[26]; // @[FDIV.scala 669:58]
  wire  a_xor_b_26 = a_26 ^ b_26; // @[FDIV.scala 670:21]
  wire  a_and_b_26 = a_26 & b_26; // @[FDIV.scala 671:21]
  wire  sum_26 = a_xor_b_26 ^ cin_26; // @[FDIV.scala 672:23]
  wire  cout_26 = a_and_b_26 | a_xor_b_26 & cin_26; // @[FDIV.scala 673:24]
  wire [1:0] temp_26 = {cout_26,sum_26}; // @[Cat.scala 31:58]
  wire  a_27 = io_in_0[27]; // @[FDIV.scala 669:32]
  wire  b_27 = io_in_1[27]; // @[FDIV.scala 669:45]
  wire  cin_27 = io_in_2[27]; // @[FDIV.scala 669:58]
  wire  a_xor_b_27 = a_27 ^ b_27; // @[FDIV.scala 670:21]
  wire  a_and_b_27 = a_27 & b_27; // @[FDIV.scala 671:21]
  wire  sum_27 = a_xor_b_27 ^ cin_27; // @[FDIV.scala 672:23]
  wire  cout_27 = a_and_b_27 | a_xor_b_27 & cin_27; // @[FDIV.scala 673:24]
  wire [1:0] temp_27 = {cout_27,sum_27}; // @[Cat.scala 31:58]
  wire [6:0] io_out_0_lo_lo = {temp_6[0],temp_5[0],temp_4[0],temp_3[0],temp_2[0],temp_1[0],temp_0[0]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_0_lo = {temp_13[0],temp_12[0],temp_11[0],temp_10[0],temp_9[0],temp_8[0],temp_7[0],io_out_0_lo_lo}; // @[Cat.scala 31:58]
  wire [6:0] io_out_0_hi_lo = {temp_20[0],temp_19[0],temp_18[0],temp_17[0],temp_16[0],temp_15[0],temp_14[0]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_0_hi = {temp_27[0],temp_26[0],temp_25[0],temp_24[0],temp_23[0],temp_22[0],temp_21[0],io_out_0_hi_lo
    }; // @[Cat.scala 31:58]
  wire [6:0] io_out_1_lo_lo = {temp_6[1],temp_5[1],temp_4[1],temp_3[1],temp_2[1],temp_1[1],temp_0[1]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_1_lo = {temp_13[1],temp_12[1],temp_11[1],temp_10[1],temp_9[1],temp_8[1],temp_7[1],io_out_1_lo_lo}; // @[Cat.scala 31:58]
  wire [6:0] io_out_1_hi_lo = {temp_20[1],temp_19[1],temp_18[1],temp_17[1],temp_16[1],temp_15[1],temp_14[1]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_1_hi = {temp_27[1],temp_26[1],temp_25[1],temp_24[1],temp_23[1],temp_22[1],temp_21[1],io_out_1_hi_lo
    }; // @[Cat.scala 31:58]
  assign io_out_0 = {io_out_0_hi,io_out_0_lo}; // @[Cat.scala 31:58]
  assign io_out_1 = {io_out_1_hi,io_out_1_lo}; // @[Cat.scala 31:58]
endmodule
module CSA3_2_6(
  input  [12:0] io_in_0,
  input  [12:0] io_in_1,
  input  [12:0] io_in_2,
  output [12:0] io_out_0,
  output [12:0] io_out_1
);
  wire  a = io_in_0[0]; // @[FDIV.scala 669:32]
  wire  b = io_in_1[0]; // @[FDIV.scala 669:45]
  wire  cin = io_in_2[0]; // @[FDIV.scala 669:58]
  wire  a_xor_b = a ^ b; // @[FDIV.scala 670:21]
  wire  a_and_b = a & b; // @[FDIV.scala 671:21]
  wire  sum = a_xor_b ^ cin; // @[FDIV.scala 672:23]
  wire  cout = a_and_b | a_xor_b & cin; // @[FDIV.scala 673:24]
  wire [1:0] temp_0 = {cout,sum}; // @[Cat.scala 31:58]
  wire  a_1 = io_in_0[1]; // @[FDIV.scala 669:32]
  wire  b_1 = io_in_1[1]; // @[FDIV.scala 669:45]
  wire  cin_1 = io_in_2[1]; // @[FDIV.scala 669:58]
  wire  a_xor_b_1 = a_1 ^ b_1; // @[FDIV.scala 670:21]
  wire  a_and_b_1 = a_1 & b_1; // @[FDIV.scala 671:21]
  wire  sum_1 = a_xor_b_1 ^ cin_1; // @[FDIV.scala 672:23]
  wire  cout_1 = a_and_b_1 | a_xor_b_1 & cin_1; // @[FDIV.scala 673:24]
  wire [1:0] temp_1 = {cout_1,sum_1}; // @[Cat.scala 31:58]
  wire  a_2 = io_in_0[2]; // @[FDIV.scala 669:32]
  wire  b_2 = io_in_1[2]; // @[FDIV.scala 669:45]
  wire  cin_2 = io_in_2[2]; // @[FDIV.scala 669:58]
  wire  a_xor_b_2 = a_2 ^ b_2; // @[FDIV.scala 670:21]
  wire  a_and_b_2 = a_2 & b_2; // @[FDIV.scala 671:21]
  wire  sum_2 = a_xor_b_2 ^ cin_2; // @[FDIV.scala 672:23]
  wire  cout_2 = a_and_b_2 | a_xor_b_2 & cin_2; // @[FDIV.scala 673:24]
  wire [1:0] temp_2 = {cout_2,sum_2}; // @[Cat.scala 31:58]
  wire  a_3 = io_in_0[3]; // @[FDIV.scala 669:32]
  wire  b_3 = io_in_1[3]; // @[FDIV.scala 669:45]
  wire  cin_3 = io_in_2[3]; // @[FDIV.scala 669:58]
  wire  a_xor_b_3 = a_3 ^ b_3; // @[FDIV.scala 670:21]
  wire  a_and_b_3 = a_3 & b_3; // @[FDIV.scala 671:21]
  wire  sum_3 = a_xor_b_3 ^ cin_3; // @[FDIV.scala 672:23]
  wire  cout_3 = a_and_b_3 | a_xor_b_3 & cin_3; // @[FDIV.scala 673:24]
  wire [1:0] temp_3 = {cout_3,sum_3}; // @[Cat.scala 31:58]
  wire  a_4 = io_in_0[4]; // @[FDIV.scala 669:32]
  wire  b_4 = io_in_1[4]; // @[FDIV.scala 669:45]
  wire  cin_4 = io_in_2[4]; // @[FDIV.scala 669:58]
  wire  a_xor_b_4 = a_4 ^ b_4; // @[FDIV.scala 670:21]
  wire  a_and_b_4 = a_4 & b_4; // @[FDIV.scala 671:21]
  wire  sum_4 = a_xor_b_4 ^ cin_4; // @[FDIV.scala 672:23]
  wire  cout_4 = a_and_b_4 | a_xor_b_4 & cin_4; // @[FDIV.scala 673:24]
  wire [1:0] temp_4 = {cout_4,sum_4}; // @[Cat.scala 31:58]
  wire  a_5 = io_in_0[5]; // @[FDIV.scala 669:32]
  wire  b_5 = io_in_1[5]; // @[FDIV.scala 669:45]
  wire  cin_5 = io_in_2[5]; // @[FDIV.scala 669:58]
  wire  a_xor_b_5 = a_5 ^ b_5; // @[FDIV.scala 670:21]
  wire  a_and_b_5 = a_5 & b_5; // @[FDIV.scala 671:21]
  wire  sum_5 = a_xor_b_5 ^ cin_5; // @[FDIV.scala 672:23]
  wire  cout_5 = a_and_b_5 | a_xor_b_5 & cin_5; // @[FDIV.scala 673:24]
  wire [1:0] temp_5 = {cout_5,sum_5}; // @[Cat.scala 31:58]
  wire  a_6 = io_in_0[6]; // @[FDIV.scala 669:32]
  wire  b_6 = io_in_1[6]; // @[FDIV.scala 669:45]
  wire  cin_6 = io_in_2[6]; // @[FDIV.scala 669:58]
  wire  a_xor_b_6 = a_6 ^ b_6; // @[FDIV.scala 670:21]
  wire  a_and_b_6 = a_6 & b_6; // @[FDIV.scala 671:21]
  wire  sum_6 = a_xor_b_6 ^ cin_6; // @[FDIV.scala 672:23]
  wire  cout_6 = a_and_b_6 | a_xor_b_6 & cin_6; // @[FDIV.scala 673:24]
  wire [1:0] temp_6 = {cout_6,sum_6}; // @[Cat.scala 31:58]
  wire  a_7 = io_in_0[7]; // @[FDIV.scala 669:32]
  wire  b_7 = io_in_1[7]; // @[FDIV.scala 669:45]
  wire  cin_7 = io_in_2[7]; // @[FDIV.scala 669:58]
  wire  a_xor_b_7 = a_7 ^ b_7; // @[FDIV.scala 670:21]
  wire  a_and_b_7 = a_7 & b_7; // @[FDIV.scala 671:21]
  wire  sum_7 = a_xor_b_7 ^ cin_7; // @[FDIV.scala 672:23]
  wire  cout_7 = a_and_b_7 | a_xor_b_7 & cin_7; // @[FDIV.scala 673:24]
  wire [1:0] temp_7 = {cout_7,sum_7}; // @[Cat.scala 31:58]
  wire  a_8 = io_in_0[8]; // @[FDIV.scala 669:32]
  wire  b_8 = io_in_1[8]; // @[FDIV.scala 669:45]
  wire  cin_8 = io_in_2[8]; // @[FDIV.scala 669:58]
  wire  a_xor_b_8 = a_8 ^ b_8; // @[FDIV.scala 670:21]
  wire  a_and_b_8 = a_8 & b_8; // @[FDIV.scala 671:21]
  wire  sum_8 = a_xor_b_8 ^ cin_8; // @[FDIV.scala 672:23]
  wire  cout_8 = a_and_b_8 | a_xor_b_8 & cin_8; // @[FDIV.scala 673:24]
  wire [1:0] temp_8 = {cout_8,sum_8}; // @[Cat.scala 31:58]
  wire  a_9 = io_in_0[9]; // @[FDIV.scala 669:32]
  wire  b_9 = io_in_1[9]; // @[FDIV.scala 669:45]
  wire  cin_9 = io_in_2[9]; // @[FDIV.scala 669:58]
  wire  a_xor_b_9 = a_9 ^ b_9; // @[FDIV.scala 670:21]
  wire  a_and_b_9 = a_9 & b_9; // @[FDIV.scala 671:21]
  wire  sum_9 = a_xor_b_9 ^ cin_9; // @[FDIV.scala 672:23]
  wire  cout_9 = a_and_b_9 | a_xor_b_9 & cin_9; // @[FDIV.scala 673:24]
  wire [1:0] temp_9 = {cout_9,sum_9}; // @[Cat.scala 31:58]
  wire  a_10 = io_in_0[10]; // @[FDIV.scala 669:32]
  wire  b_10 = io_in_1[10]; // @[FDIV.scala 669:45]
  wire  cin_10 = io_in_2[10]; // @[FDIV.scala 669:58]
  wire  a_xor_b_10 = a_10 ^ b_10; // @[FDIV.scala 670:21]
  wire  a_and_b_10 = a_10 & b_10; // @[FDIV.scala 671:21]
  wire  sum_10 = a_xor_b_10 ^ cin_10; // @[FDIV.scala 672:23]
  wire  cout_10 = a_and_b_10 | a_xor_b_10 & cin_10; // @[FDIV.scala 673:24]
  wire [1:0] temp_10 = {cout_10,sum_10}; // @[Cat.scala 31:58]
  wire  a_11 = io_in_0[11]; // @[FDIV.scala 669:32]
  wire  b_11 = io_in_1[11]; // @[FDIV.scala 669:45]
  wire  cin_11 = io_in_2[11]; // @[FDIV.scala 669:58]
  wire  a_xor_b_11 = a_11 ^ b_11; // @[FDIV.scala 670:21]
  wire  a_and_b_11 = a_11 & b_11; // @[FDIV.scala 671:21]
  wire  sum_11 = a_xor_b_11 ^ cin_11; // @[FDIV.scala 672:23]
  wire  cout_11 = a_and_b_11 | a_xor_b_11 & cin_11; // @[FDIV.scala 673:24]
  wire [1:0] temp_11 = {cout_11,sum_11}; // @[Cat.scala 31:58]
  wire  a_12 = io_in_0[12]; // @[FDIV.scala 669:32]
  wire  b_12 = io_in_1[12]; // @[FDIV.scala 669:45]
  wire  cin_12 = io_in_2[12]; // @[FDIV.scala 669:58]
  wire  a_xor_b_12 = a_12 ^ b_12; // @[FDIV.scala 670:21]
  wire  a_and_b_12 = a_12 & b_12; // @[FDIV.scala 671:21]
  wire  sum_12 = a_xor_b_12 ^ cin_12; // @[FDIV.scala 672:23]
  wire  cout_12 = a_and_b_12 | a_xor_b_12 & cin_12; // @[FDIV.scala 673:24]
  wire [1:0] temp_12 = {cout_12,sum_12}; // @[Cat.scala 31:58]
  wire [5:0] io_out_0_lo = {temp_5[0],temp_4[0],temp_3[0],temp_2[0],temp_1[0],temp_0[0]}; // @[Cat.scala 31:58]
  wire [6:0] io_out_0_hi = {temp_12[0],temp_11[0],temp_10[0],temp_9[0],temp_8[0],temp_7[0],temp_6[0]}; // @[Cat.scala 31:58]
  wire [5:0] io_out_1_lo = {temp_5[1],temp_4[1],temp_3[1],temp_2[1],temp_1[1],temp_0[1]}; // @[Cat.scala 31:58]
  wire [6:0] io_out_1_hi = {temp_12[1],temp_11[1],temp_10[1],temp_9[1],temp_8[1],temp_7[1],temp_6[1]}; // @[Cat.scala 31:58]
  assign io_out_0 = {io_out_0_hi,io_out_0_lo}; // @[Cat.scala 31:58]
  assign io_out_1 = {io_out_1_hi,io_out_1_lo}; // @[Cat.scala 31:58]
endmodule
module DivIterModule(
  input         clock,
  input  [23:0] io_a,
  input  [23:0] io_d,
  input  [1:0]  io_state,
  input         io_lastIterDoHalf,
  input         io_sigCmp,
  output [27:0] io_rem,
  output [25:0] io_quot,
  output [25:0] io_quotM1
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
  reg [31:0] _RAND_5;
  reg [31:0] _RAND_6;
  reg [31:0] _RAND_7;
  reg [31:0] _RAND_8;
  reg [31:0] _RAND_9;
  reg [31:0] _RAND_10;
  reg [31:0] _RAND_11;
  reg [31:0] _RAND_12;
  reg [31:0] _RAND_13;
  reg [31:0] _RAND_14;
  reg [31:0] _RAND_15;
  reg [31:0] _RAND_16;
  reg [31:0] _RAND_17;
  reg [31:0] _RAND_18;
  reg [31:0] _RAND_19;
  reg [31:0] _RAND_20;
  reg [31:0] _RAND_21;
  reg [31:0] _RAND_22;
  reg [31:0] _RAND_23;
  reg [31:0] _RAND_24;
  reg [31:0] _RAND_25;
  reg [31:0] _RAND_26;
  reg [31:0] _RAND_27;
  reg [31:0] _RAND_28;
  reg [31:0] _RAND_29;
  reg [31:0] _RAND_30;
  reg [31:0] _RAND_31;
  reg [31:0] _RAND_32;
  reg [31:0] _RAND_33;
  reg [31:0] _RAND_34;
  reg [31:0] _RAND_35;
  reg [31:0] _RAND_36;
  reg [31:0] _RAND_37;
  reg [31:0] _RAND_38;
  reg [31:0] _RAND_39;
  reg [31:0] _RAND_40;
  reg [31:0] _RAND_41;
  reg [31:0] _RAND_42;
  reg [31:0] _RAND_43;
  reg [31:0] _RAND_44;
  reg [31:0] _RAND_45;
  reg [31:0] _RAND_46;
  reg [31:0] _RAND_47;
  reg [31:0] _RAND_48;
`endif // RANDOMIZE_REG_INIT
  wire [9:0] signs_csa_sel_0_io_in_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_0_io_in_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_0_io_in_2; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_0_io_out_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_0_io_out_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_1_io_in_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_1_io_in_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_1_io_in_2; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_1_io_out_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_1_io_out_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_2_io_in_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_2_io_in_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_2_io_in_2; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_2_io_out_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_2_io_out_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_3_io_in_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_3_io_in_1; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_3_io_in_2; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_3_io_out_0; // @[FDIV.scala 447:21]
  wire [9:0] signs_csa_sel_3_io_out_1; // @[FDIV.scala 447:21]
  wire [27:0] csa_sel_wide_1_io_in_0; // @[FDIV.scala 455:24]
  wire [27:0] csa_sel_wide_1_io_in_1; // @[FDIV.scala 455:24]
  wire [27:0] csa_sel_wide_1_io_in_2; // @[FDIV.scala 455:24]
  wire [27:0] csa_sel_wide_1_io_out_0; // @[FDIV.scala 455:24]
  wire [27:0] csa_sel_wide_1_io_out_1; // @[FDIV.scala 455:24]
  wire [27:0] csa_sel_wide_2_io_in_0; // @[FDIV.scala 456:24]
  wire [27:0] csa_sel_wide_2_io_in_1; // @[FDIV.scala 456:24]
  wire [27:0] csa_sel_wide_2_io_in_2; // @[FDIV.scala 456:24]
  wire [27:0] csa_sel_wide_2_io_out_0; // @[FDIV.scala 456:24]
  wire [27:0] csa_sel_wide_2_io_out_1; // @[FDIV.scala 456:24]
  wire [12:0] csa_spec_0_io_in_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_0_io_in_1; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_0_io_in_2; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_0_io_out_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_0_io_out_1; // @[FDIV.scala 469:22]
  wire [12:0] signs2_csa_spec_0_0_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_0_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_0_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_0_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_0_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_1_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_1_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_1_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_1_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_1_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_2_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_2_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_2_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_2_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_2_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_3_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_3_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_3_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_3_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_0_3_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] csa_spec_1_io_in_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_1_io_in_1; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_1_io_in_2; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_1_io_out_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_1_io_out_1; // @[FDIV.scala 469:22]
  wire [12:0] signs2_csa_spec_1_0_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_0_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_0_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_0_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_0_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_1_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_1_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_1_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_1_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_1_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_2_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_2_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_2_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_2_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_2_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_3_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_3_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_3_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_3_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_1_3_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] csa_spec_2_io_in_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_2_io_in_1; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_2_io_in_2; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_2_io_out_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_2_io_out_1; // @[FDIV.scala 469:22]
  wire [12:0] signs2_csa_spec_2_0_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_0_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_0_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_0_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_0_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_1_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_1_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_1_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_1_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_1_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_2_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_2_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_2_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_2_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_2_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_3_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_3_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_3_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_3_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_2_3_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] csa_spec_3_io_in_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_3_io_in_1; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_3_io_in_2; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_3_io_out_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_3_io_out_1; // @[FDIV.scala 469:22]
  wire [12:0] signs2_csa_spec_3_0_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_0_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_0_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_0_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_0_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_1_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_1_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_1_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_1_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_1_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_2_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_2_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_2_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_2_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_2_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_3_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_3_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_3_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_3_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_3_3_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] csa_spec_4_io_in_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_4_io_in_1; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_4_io_in_2; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_4_io_out_0; // @[FDIV.scala 469:22]
  wire [12:0] csa_spec_4_io_out_1; // @[FDIV.scala 469:22]
  wire [12:0] signs2_csa_spec_4_0_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_0_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_0_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_0_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_0_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_1_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_1_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_1_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_1_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_1_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_2_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_2_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_2_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_2_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_2_io_out_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_3_io_in_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_3_io_in_1; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_3_io_in_2; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_3_io_out_0; // @[FDIV.scala 474:24]
  wire [12:0] signs2_csa_spec_4_3_io_out_1; // @[FDIV.scala 474:24]
  wire [27:0] _wsInit_T = {2'h0,io_a,2'h0}; // @[Cat.scala 31:58]
  wire [27:0] _wsInit_T_1 = {3'h0,io_a,1'h0}; // @[Cat.scala 31:58]
  wire [27:0] wsInit = io_sigCmp ? _wsInit_T : _wsInit_T_1; // @[FDIV.scala 367:19]
  wire [2:0] lookup = io_d[22:20]; // @[FDIV.scala 370:17]
  wire [2:0] smallerThanM1_invInputs = ~lookup; // @[pla.scala 78:21]
  wire  smallerThanM1_andMatrixInput_0 = smallerThanM1_invInputs[0]; // @[pla.scala 91:29]
  wire  smallerThanM1_andMatrixInput_1 = smallerThanM1_invInputs[1]; // @[pla.scala 91:29]
  wire [1:0] _smallerThanM1_T_2 = {smallerThanM1_andMatrixInput_0,smallerThanM1_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _smallerThanM1_T_3 = &_smallerThanM1_T_2; // @[pla.scala 98:74]
  wire  smallerThanM1_andMatrixInput_0_1 = smallerThanM1_invInputs[2]; // @[pla.scala 91:29]
  wire  _smallerThanM1_T_4 = &smallerThanM1_andMatrixInput_0_1; // @[pla.scala 98:74]
  wire  smallerThanM1_andMatrixInput_0_2 = lookup[2]; // @[pla.scala 90:45]
  wire  _smallerThanM1_T_5 = &smallerThanM1_andMatrixInput_0_2; // @[pla.scala 98:74]
  wire  _smallerThanM1_orMatrixOutputs_T = |_smallerThanM1_T_3; // @[pla.scala 114:39]
  wire  _smallerThanM1_orMatrixOutputs_T_1 = |_smallerThanM1_T_5; // @[pla.scala 114:39]
  wire  _smallerThanM1_orMatrixOutputs_T_2 = |_smallerThanM1_T_4; // @[pla.scala 114:39]
  wire [5:0] smallerThanM1_orMatrixOutputs = {3'h7,_smallerThanM1_orMatrixOutputs_T_2,_smallerThanM1_orMatrixOutputs_T_1
    ,_smallerThanM1_orMatrixOutputs_T}; // @[Cat.scala 31:58]
  wire [5:0] smallerThanM1_invMatrixOutputs = {smallerThanM1_orMatrixOutputs[5],smallerThanM1_orMatrixOutputs[4],
    smallerThanM1_orMatrixOutputs[3],smallerThanM1_orMatrixOutputs[2],smallerThanM1_orMatrixOutputs[1],
    smallerThanM1_orMatrixOutputs[0]}; // @[Cat.scala 31:58]
  wire [5:0] _GEN_50 = {{1'd0}, wsInit[25:21]}; // @[FDIV.scala 372:45]
  wire [5:0] _smallerThanM1_T_7 = _GEN_50 + smallerThanM1_invMatrixOutputs; // @[FDIV.scala 372:45]
  wire  smallerThanM1 = _smallerThanM1_T_7[5]; // @[FDIV.scala 372:107]
  wire [1:0] _smallerThanM2_T_4 = {smallerThanM1_andMatrixInput_0,smallerThanM1_andMatrixInput_0_1}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_5 = &_smallerThanM2_T_4; // @[pla.scala 98:74]
  wire [1:0] _smallerThanM2_T_6 = {smallerThanM1_andMatrixInput_1,smallerThanM1_andMatrixInput_0_1}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_7 = &_smallerThanM2_T_6; // @[pla.scala 98:74]
  wire  smallerThanM2_andMatrixInput_0_3 = lookup[0]; // @[pla.scala 90:45]
  wire [1:0] _smallerThanM2_T_8 = {smallerThanM2_andMatrixInput_0_3,smallerThanM1_andMatrixInput_0_1}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_9 = &_smallerThanM2_T_8; // @[pla.scala 98:74]
  wire  smallerThanM2_andMatrixInput_0_4 = lookup[1]; // @[pla.scala 90:45]
  wire  _smallerThanM2_T_10 = &smallerThanM2_andMatrixInput_0_4; // @[pla.scala 98:74]
  wire [1:0] _smallerThanM2_T_11 = {smallerThanM2_andMatrixInput_0_3,smallerThanM2_andMatrixInput_0_4}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_12 = &_smallerThanM2_T_11; // @[pla.scala 98:74]
  wire [2:0] _smallerThanM2_T_13 = {smallerThanM2_andMatrixInput_0_3,smallerThanM2_andMatrixInput_0_4,
    smallerThanM1_andMatrixInput_0_1}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_14 = &_smallerThanM2_T_13; // @[pla.scala 98:74]
  wire [1:0] _smallerThanM2_T_16 = {smallerThanM1_andMatrixInput_0,smallerThanM1_andMatrixInput_0_2}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_17 = &_smallerThanM2_T_16; // @[pla.scala 98:74]
  wire [1:0] _smallerThanM2_T_18 = {smallerThanM1_andMatrixInput_1,smallerThanM1_andMatrixInput_0_2}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_T_19 = &_smallerThanM2_T_18; // @[pla.scala 98:74]
  wire  _smallerThanM2_orMatrixOutputs_T = |_smallerThanM2_T_10; // @[pla.scala 114:39]
  wire [1:0] _smallerThanM2_orMatrixOutputs_T_1 = {_smallerThanM2_T_9,_smallerThanM2_T_17}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_orMatrixOutputs_T_2 = |_smallerThanM2_orMatrixOutputs_T_1; // @[pla.scala 114:39]
  wire [2:0] _smallerThanM2_orMatrixOutputs_T_3 = {_smallerThanM1_T_3,_smallerThanM2_T_14,_smallerThanM2_T_19}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_orMatrixOutputs_T_4 = |_smallerThanM2_orMatrixOutputs_T_3; // @[pla.scala 114:39]
  wire [1:0] _smallerThanM2_orMatrixOutputs_T_5 = {_smallerThanM2_T_12,_smallerThanM1_T_5}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_orMatrixOutputs_T_6 = |_smallerThanM2_orMatrixOutputs_T_5; // @[pla.scala 114:39]
  wire [1:0] _smallerThanM2_orMatrixOutputs_T_7 = {_smallerThanM2_T_5,_smallerThanM2_T_7}; // @[Cat.scala 31:58]
  wire  _smallerThanM2_orMatrixOutputs_T_8 = |_smallerThanM2_orMatrixOutputs_T_7; // @[pla.scala 114:39]
  wire [5:0] smallerThanM2_orMatrixOutputs = {1'h1,_smallerThanM2_orMatrixOutputs_T_8,_smallerThanM2_orMatrixOutputs_T_6
    ,_smallerThanM2_orMatrixOutputs_T_4,_smallerThanM2_orMatrixOutputs_T_2,_smallerThanM2_orMatrixOutputs_T}; // @[Cat.scala 31:58]
  wire [5:0] smallerThanM2_invMatrixOutputs = {smallerThanM2_orMatrixOutputs[5],smallerThanM2_orMatrixOutputs[4],
    smallerThanM2_orMatrixOutputs[3],smallerThanM2_orMatrixOutputs[2],smallerThanM2_orMatrixOutputs[1],
    smallerThanM2_orMatrixOutputs[0]}; // @[Cat.scala 31:58]
  wire [5:0] _smallerThanM2_T_21 = _GEN_50 + smallerThanM2_invMatrixOutputs; // @[FDIV.scala 373:45]
  wire  smallerThanM2 = _smallerThanM2_T_21[5]; // @[FDIV.scala 373:107]
  wire [24:0] dPos = {1'h0,io_d}; // @[Cat.scala 31:58]
  wire [24:0] dNeg = 25'h0 - dPos; // @[FDIV.scala 380:14]
  wire [9:0] _signs_T_16 = {signs_csa_sel_3_io_out_1[8:0], 1'h0}; // @[FDIV.scala 452:43]
  wire [9:0] _signs_T_18 = signs_csa_sel_3_io_out_0 + _signs_T_16; // @[FDIV.scala 452:20]
  wire  signs_3 = _signs_T_18[9]; // @[FDIV.scala 452:49]
  wire [9:0] _signs_T_11 = {signs_csa_sel_2_io_out_1[8:0], 1'h0}; // @[FDIV.scala 452:43]
  wire [9:0] _signs_T_13 = signs_csa_sel_2_io_out_0 + _signs_T_11; // @[FDIV.scala 452:20]
  wire  signs_2 = _signs_T_13[9]; // @[FDIV.scala 452:49]
  wire [9:0] _signs_T_6 = {signs_csa_sel_1_io_out_1[8:0], 1'h0}; // @[FDIV.scala 452:43]
  wire [9:0] _signs_T_8 = signs_csa_sel_1_io_out_0 + _signs_T_6; // @[FDIV.scala 452:20]
  wire  signs_1 = _signs_T_8[9]; // @[FDIV.scala 452:49]
  wire [9:0] _signs_T_1 = {signs_csa_sel_0_io_out_1[8:0], 1'h0}; // @[FDIV.scala 452:43]
  wire [9:0] _signs_T_3 = signs_csa_sel_0_io_out_0 + _signs_T_1; // @[FDIV.scala 452:20]
  wire  signs_0 = _signs_T_3[9]; // @[FDIV.scala 452:49]
  wire [3:0] _qNext_T = {signs_3,signs_2,signs_1,signs_0}; // @[FDIV.scala 454:29]
  wire  _qNext_sel_q_4_T_3 = ~_qNext_T[2]; // @[FDIV.scala 634:32]
  wire  _qNext_sel_q_4_T_6 = ~_qNext_T[1]; // @[FDIV.scala 634:47]
  wire  qNext_sel_q_4 = ~_qNext_T[3] & ~_qNext_T[2] & ~_qNext_T[1]; // @[FDIV.scala 634:44]
  wire  qNext_sel_q_3 = _qNext_T[3] & _qNext_sel_q_4_T_3 & _qNext_sel_q_4_T_6; // @[FDIV.scala 633:42]
  wire  qNext_sel_q_2 = _qNext_T[2] & _qNext_sel_q_4_T_6; // @[FDIV.scala 632:27]
  wire  qNext_sel_q_1 = ~_qNext_T[0] & _qNext_T[1] & _qNext_T[2]; // @[FDIV.scala 631:42]
  wire  qNext_sel_q_0 = _qNext_T[0] & _qNext_T[1] & _qNext_T[2]; // @[FDIV.scala 630:40]
  wire [4:0] qNext = {qNext_sel_q_4,qNext_sel_q_3,qNext_sel_q_2,qNext_sel_q_1,qNext_sel_q_0}; // @[FDIV.scala 635:10]
  wire [12:0] _signs2_T_16 = {signs2_csa_spec_0_3_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_18 = signs2_csa_spec_0_3_io_out_0 + _signs2_T_16; // @[FDIV.scala 478:23]
  wire  signs2__3 = _signs2_T_18[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_11 = {signs2_csa_spec_0_2_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_13 = signs2_csa_spec_0_2_io_out_0 + _signs2_T_11; // @[FDIV.scala 478:23]
  wire  signs2__2 = _signs2_T_13[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_6 = {signs2_csa_spec_0_1_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_8 = signs2_csa_spec_0_1_io_out_0 + _signs2_T_6; // @[FDIV.scala 478:23]
  wire  signs2__1 = _signs2_T_8[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_1 = {signs2_csa_spec_0_0_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_3 = signs2_csa_spec_0_0_io_out_0 + _signs2_T_1; // @[FDIV.scala 478:23]
  wire  signs2__0 = _signs2_T_3[12]; // @[FDIV.scala 478:54]
  wire [3:0] _qVec2_T = {signs2__3,signs2__2,signs2__1,signs2__0}; // @[FDIV.scala 480:35]
  wire  _qVec2_spec_q_0_4_T_3 = ~_qVec2_T[2]; // @[FDIV.scala 634:32]
  wire  _qVec2_spec_q_0_4_T_6 = ~_qVec2_T[1]; // @[FDIV.scala 634:47]
  wire  qVec2_spec_q_0_4 = ~_qVec2_T[3] & ~_qVec2_T[2] & ~_qVec2_T[1]; // @[FDIV.scala 634:44]
  wire  qVec2_spec_q_0_3 = _qVec2_T[3] & _qVec2_spec_q_0_4_T_3 & _qVec2_spec_q_0_4_T_6; // @[FDIV.scala 633:42]
  wire  qVec2_spec_q_0_2 = _qVec2_T[2] & _qVec2_spec_q_0_4_T_6; // @[FDIV.scala 632:27]
  wire  qVec2_spec_q_0_1 = ~_qVec2_T[0] & _qVec2_T[1] & _qVec2_T[2]; // @[FDIV.scala 631:42]
  wire  qVec2_spec_q_0_0 = _qVec2_T[0] & _qVec2_T[1] & _qVec2_T[2]; // @[FDIV.scala 630:40]
  wire [4:0] qVec2 = {qVec2_spec_q_0_4,qVec2_spec_q_0_3,qVec2_spec_q_0_2,qVec2_spec_q_0_1,qVec2_spec_q_0_0}; // @[FDIV.scala 635:10]
  wire [4:0] _qNext2_T_5 = qNext[0] ? qVec2 : 5'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_T_36 = {signs2_csa_spec_1_3_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_38 = signs2_csa_spec_1_3_io_out_0 + _signs2_T_36; // @[FDIV.scala 478:23]
  wire  signs2_1_3 = _signs2_T_38[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_31 = {signs2_csa_spec_1_2_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_33 = signs2_csa_spec_1_2_io_out_0 + _signs2_T_31; // @[FDIV.scala 478:23]
  wire  signs2_1_2 = _signs2_T_33[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_26 = {signs2_csa_spec_1_1_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_28 = signs2_csa_spec_1_1_io_out_0 + _signs2_T_26; // @[FDIV.scala 478:23]
  wire  signs2_1_1 = _signs2_T_28[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_21 = {signs2_csa_spec_1_0_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_23 = signs2_csa_spec_1_0_io_out_0 + _signs2_T_21; // @[FDIV.scala 478:23]
  wire  signs2_1_0 = _signs2_T_23[12]; // @[FDIV.scala 478:54]
  wire [3:0] _qVec2_T_1 = {signs2_1_3,signs2_1_2,signs2_1_1,signs2_1_0}; // @[FDIV.scala 480:35]
  wire  _qVec2_spec_q_1_4_T_3 = ~_qVec2_T_1[2]; // @[FDIV.scala 634:32]
  wire  _qVec2_spec_q_1_4_T_6 = ~_qVec2_T_1[1]; // @[FDIV.scala 634:47]
  wire  qVec2_spec_q_1_4 = ~_qVec2_T_1[3] & ~_qVec2_T_1[2] & ~_qVec2_T_1[1]; // @[FDIV.scala 634:44]
  wire  qVec2_spec_q_1_3 = _qVec2_T_1[3] & _qVec2_spec_q_1_4_T_3 & _qVec2_spec_q_1_4_T_6; // @[FDIV.scala 633:42]
  wire  qVec2_spec_q_1_2 = _qVec2_T_1[2] & _qVec2_spec_q_1_4_T_6; // @[FDIV.scala 632:27]
  wire  qVec2_spec_q_1_1 = ~_qVec2_T_1[0] & _qVec2_T_1[1] & _qVec2_T_1[2]; // @[FDIV.scala 631:42]
  wire  qVec2_spec_q_1_0 = _qVec2_T_1[0] & _qVec2_T_1[1] & _qVec2_T_1[2]; // @[FDIV.scala 630:40]
  wire [4:0] qVec2_1 = {qVec2_spec_q_1_4,qVec2_spec_q_1_3,qVec2_spec_q_1_2,qVec2_spec_q_1_1,qVec2_spec_q_1_0}; // @[FDIV.scala 635:10]
  wire [4:0] _qNext2_T_6 = qNext[1] ? qVec2_1 : 5'h0; // @[Mux.scala 27:73]
  wire [4:0] _qNext2_T_10 = _qNext2_T_5 | _qNext2_T_6; // @[Mux.scala 27:73]
  wire [12:0] _signs2_T_56 = {signs2_csa_spec_2_3_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_58 = signs2_csa_spec_2_3_io_out_0 + _signs2_T_56; // @[FDIV.scala 478:23]
  wire  signs2_2_3 = _signs2_T_58[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_51 = {signs2_csa_spec_2_2_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_53 = signs2_csa_spec_2_2_io_out_0 + _signs2_T_51; // @[FDIV.scala 478:23]
  wire  signs2_2_2 = _signs2_T_53[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_46 = {signs2_csa_spec_2_1_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_48 = signs2_csa_spec_2_1_io_out_0 + _signs2_T_46; // @[FDIV.scala 478:23]
  wire  signs2_2_1 = _signs2_T_48[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_41 = {signs2_csa_spec_2_0_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_43 = signs2_csa_spec_2_0_io_out_0 + _signs2_T_41; // @[FDIV.scala 478:23]
  wire  signs2_2_0 = _signs2_T_43[12]; // @[FDIV.scala 478:54]
  wire [3:0] _qVec2_T_2 = {signs2_2_3,signs2_2_2,signs2_2_1,signs2_2_0}; // @[FDIV.scala 480:35]
  wire  _qVec2_spec_q_2_4_T_3 = ~_qVec2_T_2[2]; // @[FDIV.scala 634:32]
  wire  _qVec2_spec_q_2_4_T_6 = ~_qVec2_T_2[1]; // @[FDIV.scala 634:47]
  wire  qVec2_spec_q_2_4 = ~_qVec2_T_2[3] & ~_qVec2_T_2[2] & ~_qVec2_T_2[1]; // @[FDIV.scala 634:44]
  wire  qVec2_spec_q_2_3 = _qVec2_T_2[3] & _qVec2_spec_q_2_4_T_3 & _qVec2_spec_q_2_4_T_6; // @[FDIV.scala 633:42]
  wire  qVec2_spec_q_2_2 = _qVec2_T_2[2] & _qVec2_spec_q_2_4_T_6; // @[FDIV.scala 632:27]
  wire  qVec2_spec_q_2_1 = ~_qVec2_T_2[0] & _qVec2_T_2[1] & _qVec2_T_2[2]; // @[FDIV.scala 631:42]
  wire  qVec2_spec_q_2_0 = _qVec2_T_2[0] & _qVec2_T_2[1] & _qVec2_T_2[2]; // @[FDIV.scala 630:40]
  wire [4:0] qVec2_2 = {qVec2_spec_q_2_4,qVec2_spec_q_2_3,qVec2_spec_q_2_2,qVec2_spec_q_2_1,qVec2_spec_q_2_0}; // @[FDIV.scala 635:10]
  wire [4:0] _qNext2_T_7 = qNext[2] ? qVec2_2 : 5'h0; // @[Mux.scala 27:73]
  wire [4:0] _qNext2_T_11 = _qNext2_T_10 | _qNext2_T_7; // @[Mux.scala 27:73]
  wire [12:0] _signs2_T_76 = {signs2_csa_spec_3_3_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_78 = signs2_csa_spec_3_3_io_out_0 + _signs2_T_76; // @[FDIV.scala 478:23]
  wire  signs2_3_3 = _signs2_T_78[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_71 = {signs2_csa_spec_3_2_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_73 = signs2_csa_spec_3_2_io_out_0 + _signs2_T_71; // @[FDIV.scala 478:23]
  wire  signs2_3_2 = _signs2_T_73[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_66 = {signs2_csa_spec_3_1_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_68 = signs2_csa_spec_3_1_io_out_0 + _signs2_T_66; // @[FDIV.scala 478:23]
  wire  signs2_3_1 = _signs2_T_68[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_61 = {signs2_csa_spec_3_0_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_63 = signs2_csa_spec_3_0_io_out_0 + _signs2_T_61; // @[FDIV.scala 478:23]
  wire  signs2_3_0 = _signs2_T_63[12]; // @[FDIV.scala 478:54]
  wire [3:0] _qVec2_T_3 = {signs2_3_3,signs2_3_2,signs2_3_1,signs2_3_0}; // @[FDIV.scala 480:35]
  wire  _qVec2_spec_q_3_4_T_3 = ~_qVec2_T_3[2]; // @[FDIV.scala 634:32]
  wire  _qVec2_spec_q_3_4_T_6 = ~_qVec2_T_3[1]; // @[FDIV.scala 634:47]
  wire  qVec2_spec_q_3_4 = ~_qVec2_T_3[3] & ~_qVec2_T_3[2] & ~_qVec2_T_3[1]; // @[FDIV.scala 634:44]
  wire  qVec2_spec_q_3_3 = _qVec2_T_3[3] & _qVec2_spec_q_3_4_T_3 & _qVec2_spec_q_3_4_T_6; // @[FDIV.scala 633:42]
  wire  qVec2_spec_q_3_2 = _qVec2_T_3[2] & _qVec2_spec_q_3_4_T_6; // @[FDIV.scala 632:27]
  wire  qVec2_spec_q_3_1 = ~_qVec2_T_3[0] & _qVec2_T_3[1] & _qVec2_T_3[2]; // @[FDIV.scala 631:42]
  wire  qVec2_spec_q_3_0 = _qVec2_T_3[0] & _qVec2_T_3[1] & _qVec2_T_3[2]; // @[FDIV.scala 630:40]
  wire [4:0] qVec2_3 = {qVec2_spec_q_3_4,qVec2_spec_q_3_3,qVec2_spec_q_3_2,qVec2_spec_q_3_1,qVec2_spec_q_3_0}; // @[FDIV.scala 635:10]
  wire [4:0] _qNext2_T_8 = qNext[3] ? qVec2_3 : 5'h0; // @[Mux.scala 27:73]
  wire [4:0] _qNext2_T_12 = _qNext2_T_11 | _qNext2_T_8; // @[Mux.scala 27:73]
  wire [12:0] _signs2_T_96 = {signs2_csa_spec_4_3_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_98 = signs2_csa_spec_4_3_io_out_0 + _signs2_T_96; // @[FDIV.scala 478:23]
  wire  signs2_4_3 = _signs2_T_98[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_91 = {signs2_csa_spec_4_2_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_93 = signs2_csa_spec_4_2_io_out_0 + _signs2_T_91; // @[FDIV.scala 478:23]
  wire  signs2_4_2 = _signs2_T_93[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_86 = {signs2_csa_spec_4_1_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_88 = signs2_csa_spec_4_1_io_out_0 + _signs2_T_86; // @[FDIV.scala 478:23]
  wire  signs2_4_1 = _signs2_T_88[12]; // @[FDIV.scala 478:54]
  wire [12:0] _signs2_T_81 = {signs2_csa_spec_4_0_io_out_1[11:0], 1'h0}; // @[FDIV.scala 478:48]
  wire [12:0] _signs2_T_83 = signs2_csa_spec_4_0_io_out_0 + _signs2_T_81; // @[FDIV.scala 478:23]
  wire  signs2_4_0 = _signs2_T_83[12]; // @[FDIV.scala 478:54]
  wire [3:0] _qVec2_T_4 = {signs2_4_3,signs2_4_2,signs2_4_1,signs2_4_0}; // @[FDIV.scala 480:35]
  wire  _qVec2_spec_q_4_4_T_3 = ~_qVec2_T_4[2]; // @[FDIV.scala 634:32]
  wire  _qVec2_spec_q_4_4_T_6 = ~_qVec2_T_4[1]; // @[FDIV.scala 634:47]
  wire  qVec2_spec_q_4_4 = ~_qVec2_T_4[3] & ~_qVec2_T_4[2] & ~_qVec2_T_4[1]; // @[FDIV.scala 634:44]
  wire  qVec2_spec_q_4_3 = _qVec2_T_4[3] & _qVec2_spec_q_4_4_T_3 & _qVec2_spec_q_4_4_T_6; // @[FDIV.scala 633:42]
  wire  qVec2_spec_q_4_2 = _qVec2_T_4[2] & _qVec2_spec_q_4_4_T_6; // @[FDIV.scala 632:27]
  wire  qVec2_spec_q_4_1 = ~_qVec2_T_4[0] & _qVec2_T_4[1] & _qVec2_T_4[2]; // @[FDIV.scala 631:42]
  wire  qVec2_spec_q_4_0 = _qVec2_T_4[0] & _qVec2_T_4[1] & _qVec2_T_4[2]; // @[FDIV.scala 630:40]
  wire [4:0] qVec2_4 = {qVec2_spec_q_4_4,qVec2_spec_q_4_3,qVec2_spec_q_4_2,qVec2_spec_q_4_1,qVec2_spec_q_4_0}; // @[FDIV.scala 635:10]
  wire [4:0] _qNext2_T_9 = qNext[4] ? qVec2_4 : 5'h0; // @[Mux.scala 27:73]
  wire [4:0] qNext2 = _qNext2_T_12 | _qNext2_T_9; // @[Mux.scala 27:73]
  wire  _qPrevReg_T_4 = io_state[0] | io_state[1]; // @[FDIV.scala 410:79]
  reg [7:0] qPrevReg; // @[Reg.scala 16:16]
  reg [27:0] wsReg; // @[Reg.scala 16:16]
  wire [28:0] _wcIter_T = {csa_sel_wide_1_io_out_1, 1'h0}; // @[FDIV.scala 464:56]
  wire [28:0] _wcIter_T_2 = {csa_sel_wide_2_io_out_1, 1'h0}; // @[FDIV.scala 464:97]
  reg [27:0] wcReg; // @[Reg.scala 16:16]
  reg [25:0] quotIterReg; // @[Reg.scala 16:16]
  wire [27:0] _quotHalfIter_quotNext_T_1 = {quotIterReg, 2'h0}; // @[FDIV.scala 642:21]
  wire [27:0] _quotHalfIter_quotNext_T_2 = _quotHalfIter_quotNext_T_1 | 28'h2; // @[FDIV.scala 642:26]
  wire [27:0] _quotHalfIter_quotNext_T_15 = qPrevReg[4] ? _quotHalfIter_quotNext_T_2 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_5 = _quotHalfIter_quotNext_T_1 | 28'h1; // @[FDIV.scala 643:26]
  wire [27:0] _quotHalfIter_quotNext_T_16 = qPrevReg[3] ? _quotHalfIter_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_20 = _quotHalfIter_quotNext_T_15 | _quotHalfIter_quotNext_T_16; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_17 = qPrevReg[2] ? _quotHalfIter_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_21 = _quotHalfIter_quotNext_T_20 | _quotHalfIter_quotNext_T_17; // @[Mux.scala 27:73]
  reg [25:0] quotM1IterReg; // @[Reg.scala 16:16]
  wire [27:0] _quotHalfIter_quotNext_T_10 = {quotM1IterReg, 2'h0}; // @[FDIV.scala 645:23]
  wire [27:0] _quotHalfIter_quotNext_T_11 = _quotHalfIter_quotNext_T_10 | 28'h3; // @[FDIV.scala 645:28]
  wire [27:0] _quotHalfIter_quotNext_T_18 = qPrevReg[1] ? _quotHalfIter_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_22 = _quotHalfIter_quotNext_T_21 | _quotHalfIter_quotNext_T_18; // @[Mux.scala 27:73]
  wire [27:0] _quotHalfIter_quotNext_T_14 = _quotHalfIter_quotNext_T_10 | 28'h2; // @[FDIV.scala 646:28]
  wire [27:0] _quotHalfIter_quotNext_T_19 = qPrevReg[0] ? _quotHalfIter_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] quotHalfIter_quotNext = _quotHalfIter_quotNext_T_22 | _quotHalfIter_quotNext_T_19; // @[Mux.scala 27:73]
  wire [25:0] quotHalfIter = quotHalfIter_quotNext[25:0]; // @[FDIV.scala 655:14]
  wire [27:0] _quotIterNext_quotNext_T_1 = {quotHalfIter, 2'h0}; // @[FDIV.scala 642:21]
  wire [27:0] _quotIterNext_quotNext_T_2 = _quotIterNext_quotNext_T_1 | 28'h2; // @[FDIV.scala 642:26]
  wire [27:0] _quotIterNext_quotNext_T_15 = qNext[4] ? _quotIterNext_quotNext_T_2 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_5 = _quotIterNext_quotNext_T_1 | 28'h1; // @[FDIV.scala 643:26]
  wire [27:0] _quotIterNext_quotNext_T_16 = qNext[3] ? _quotIterNext_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_20 = _quotIterNext_quotNext_T_15 | _quotIterNext_quotNext_T_16; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_17 = qNext[2] ? _quotIterNext_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_21 = _quotIterNext_quotNext_T_20 | _quotIterNext_quotNext_T_17; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_15 = qPrevReg[4] ? _quotHalfIter_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_16 = qPrevReg[3] ? _quotHalfIter_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_20 = _quotM1HalfIter_quotM1Next_T_15 | _quotM1HalfIter_quotM1Next_T_16; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_17 = qPrevReg[2] ? _quotHalfIter_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_21 = _quotM1HalfIter_quotM1Next_T_20 | _quotM1HalfIter_quotM1Next_T_17; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_18 = qPrevReg[1] ? _quotHalfIter_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_22 = _quotM1HalfIter_quotM1Next_T_21 | _quotM1HalfIter_quotM1Next_T_18; // @[Mux.scala 27:73]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_14 = _quotHalfIter_quotNext_T_10 | 28'h1; // @[FDIV.scala 653:28]
  wire [27:0] _quotM1HalfIter_quotM1Next_T_19 = qPrevReg[0] ? _quotM1HalfIter_quotM1Next_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] quotM1HalfIter_quotM1Next = _quotM1HalfIter_quotM1Next_T_22 | _quotM1HalfIter_quotM1Next_T_19; // @[Mux.scala 27:73]
  wire [25:0] quotM1HalfIter = quotM1HalfIter_quotM1Next[25:0]; // @[FDIV.scala 655:38]
  wire [27:0] _quotIterNext_quotNext_T_10 = {quotM1HalfIter, 2'h0}; // @[FDIV.scala 645:23]
  wire [27:0] _quotIterNext_quotNext_T_11 = _quotIterNext_quotNext_T_10 | 28'h3; // @[FDIV.scala 645:28]
  wire [27:0] _quotIterNext_quotNext_T_18 = qNext[1] ? _quotIterNext_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_22 = _quotIterNext_quotNext_T_21 | _quotIterNext_quotNext_T_18; // @[Mux.scala 27:73]
  wire [27:0] _quotIterNext_quotNext_T_14 = _quotIterNext_quotNext_T_10 | 28'h2; // @[FDIV.scala 646:28]
  wire [27:0] _quotIterNext_quotNext_T_19 = qNext[0] ? _quotIterNext_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] quotIterNext_quotNext = _quotIterNext_quotNext_T_22 | _quotIterNext_quotNext_T_19; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_15 = qNext[4] ? _quotIterNext_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_16 = qNext[3] ? _quotIterNext_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_20 = _quotM1IterNext_quotM1Next_T_15 | _quotM1IterNext_quotM1Next_T_16; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_17 = qNext[2] ? _quotIterNext_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_21 = _quotM1IterNext_quotM1Next_T_20 | _quotM1IterNext_quotM1Next_T_17; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_18 = qNext[1] ? _quotIterNext_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_22 = _quotM1IterNext_quotM1Next_T_21 | _quotM1IterNext_quotM1Next_T_18; // @[Mux.scala 27:73]
  wire [27:0] _quotM1IterNext_quotM1Next_T_14 = _quotIterNext_quotNext_T_10 | 28'h1; // @[FDIV.scala 653:28]
  wire [27:0] _quotM1IterNext_quotM1Next_T_19 = qNext[0] ? _quotM1IterNext_quotM1Next_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] quotM1IterNext_quotM1Next = _quotM1IterNext_quotM1Next_T_22 | _quotM1IterNext_quotM1Next_T_19; // @[Mux.scala 27:73]
  wire  _T = &smallerThanM1_andMatrixInput_1; // @[pla.scala 98:74]
  wire [1:0] _T_3 = {smallerThanM2_andMatrixInput_0_3,smallerThanM1_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _T_4 = &_T_3; // @[pla.scala 98:74]
  wire [2:0] _T_11 = {smallerThanM1_andMatrixInput_0,smallerThanM2_andMatrixInput_0_4,smallerThanM1_andMatrixInput_0_2}; // @[Cat.scala 31:58]
  wire  _T_12 = &_T_11; // @[pla.scala 98:74]
  wire [2:0] _T_13 = {smallerThanM2_andMatrixInput_0_3,smallerThanM2_andMatrixInput_0_4,smallerThanM1_andMatrixInput_0_2
    }; // @[Cat.scala 31:58]
  wire  _T_14 = &_T_13; // @[pla.scala 98:74]
  wire  _orMatrixOutputs_T = |_T; // @[pla.scala 114:39]
  wire [2:0] _orMatrixOutputs_T_3 = {_smallerThanM2_T_7,_T_4,_T_12}; // @[Cat.scala 31:58]
  wire  _orMatrixOutputs_T_4 = |_orMatrixOutputs_T_3; // @[pla.scala 114:39]
  wire [1:0] _orMatrixOutputs_T_5 = {_smallerThanM2_T_7,_T_14}; // @[Cat.scala 31:58]
  wire  _orMatrixOutputs_T_6 = |_orMatrixOutputs_T_5; // @[pla.scala 114:39]
  wire [1:0] _orMatrixOutputs_T_7 = {_smallerThanM2_T_10,_smallerThanM1_T_5}; // @[Cat.scala 31:58]
  wire  _orMatrixOutputs_T_8 = |_orMatrixOutputs_T_7; // @[pla.scala 114:39]
  wire [5:0] orMatrixOutputs = {1'h0,_orMatrixOutputs_T_8,_orMatrixOutputs_T_6,_orMatrixOutputs_T_4,
    _smallerThanM2_orMatrixOutputs_T_2,_orMatrixOutputs_T}; // @[Cat.scala 31:58]
  wire [5:0] invMatrixOutputs = {orMatrixOutputs[5],orMatrixOutputs[4],orMatrixOutputs[3],orMatrixOutputs[2],
    orMatrixOutputs[1],orMatrixOutputs[0]}; // @[Cat.scala 31:58]
  wire  signBit = invMatrixOutputs[5]; // @[FDIV.scala 132:20]
  wire [3:0] _T_16 = signBit ? 4'hf : 4'h0; // @[Bitwise.scala 74:12]
  wire [11:0] mNeg_0 = {_T_16,orMatrixOutputs[5],orMatrixOutputs[4],orMatrixOutputs[3],orMatrixOutputs[2],
    orMatrixOutputs[1],orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire [1:0] _T_22 = {smallerThanM2_andMatrixInput_0_4,smallerThanM1_andMatrixInput_0_1}; // @[Cat.scala 31:58]
  wire  _T_23 = &_T_22; // @[pla.scala 98:74]
  wire [1:0] _orMatrixOutputs_T_9 = {_smallerThanM2_T_9,_T_23}; // @[Cat.scala 31:58]
  wire  _orMatrixOutputs_T_10 = |_orMatrixOutputs_T_9; // @[pla.scala 114:39]
  wire [5:0] orMatrixOutputs_1 = {2'h0,_smallerThanM1_orMatrixOutputs_T_1,_smallerThanM1_orMatrixOutputs_T_2,
    _orMatrixOutputs_T_10,1'h0}; // @[Cat.scala 31:58]
  wire [5:0] invMatrixOutputs_1 = {orMatrixOutputs_1[5],orMatrixOutputs_1[4],orMatrixOutputs_1[3],orMatrixOutputs_1[2],
    orMatrixOutputs_1[1],orMatrixOutputs_1[0]}; // @[Cat.scala 31:58]
  wire  signBit_1 = invMatrixOutputs_1[5]; // @[FDIV.scala 132:20]
  wire [3:0] _T_26 = signBit_1 ? 4'hf : 4'h0; // @[Bitwise.scala 74:12]
  wire [11:0] mNeg_1 = {_T_26,orMatrixOutputs_1[5],orMatrixOutputs_1[4],orMatrixOutputs_1[3],orMatrixOutputs_1[2],
    orMatrixOutputs_1[1],orMatrixOutputs_1[0],2'h0}; // @[Cat.scala 31:58]
  wire  signBit_2 = smallerThanM1_invMatrixOutputs[5]; // @[FDIV.scala 132:20]
  wire [3:0] _T_34 = signBit_2 ? 4'hf : 4'h0; // @[Bitwise.scala 74:12]
  wire [11:0] mNeg_2 = {_T_34,smallerThanM1_orMatrixOutputs[5],smallerThanM1_orMatrixOutputs[4],
    smallerThanM1_orMatrixOutputs[3],smallerThanM1_orMatrixOutputs[2],smallerThanM1_orMatrixOutputs[1],
    smallerThanM1_orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire  signBit_3 = smallerThanM2_invMatrixOutputs[5]; // @[FDIV.scala 132:20]
  wire [3:0] _T_56 = signBit_3 ? 4'hf : 4'h0; // @[Bitwise.scala 74:12]
  wire [11:0] mNeg_3 = {_T_56,smallerThanM2_orMatrixOutputs[5],smallerThanM2_orMatrixOutputs[4],
    smallerThanM2_orMatrixOutputs[3],smallerThanM2_orMatrixOutputs[2],smallerThanM2_orMatrixOutputs[1],
    smallerThanM2_orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire  signBit_4 = dPos[24]; // @[FDIV.scala 132:20]
  wire [27:0] udNeg_0 = {signBit_4,1'h0,io_d,2'h0}; // @[Cat.scala 31:58]
  wire [1:0] _T_62 = signBit_4 ? 2'h3 : 2'h0; // @[Bitwise.scala 74:12]
  wire [27:0] udNeg_1 = {_T_62,1'h0,io_d,1'h0}; // @[Cat.scala 31:58]
  wire  signBit_6 = dNeg[24]; // @[FDIV.scala 132:20]
  wire [1:0] _T_66 = signBit_6 ? 2'h3 : 2'h0; // @[Bitwise.scala 74:12]
  wire [27:0] udNeg_3 = {_T_66,dNeg,1'h0}; // @[Cat.scala 31:58]
  wire [27:0] udNeg_4 = {signBit_6,dNeg,2'h0}; // @[Cat.scala 31:58]
  wire [9:0] rudNeg_0 = udNeg_0[26:17]; // @[FDIV.scala 429:50]
  wire [9:0] rudNeg_1 = udNeg_1[26:17]; // @[FDIV.scala 429:50]
  wire [9:0] rudNeg_3 = udNeg_3[26:17]; // @[FDIV.scala 429:50]
  wire [9:0] rudNeg_4 = udNeg_4[26:17]; // @[FDIV.scala 429:50]
  wire [11:0] r2udNeg_0 = udNeg_0[26:15]; // @[FDIV.scala 430:51]
  wire [11:0] r2udNeg_1 = udNeg_1[26:15]; // @[FDIV.scala 430:51]
  wire [11:0] r2udNeg_3 = udNeg_3[26:15]; // @[FDIV.scala 430:51]
  wire [11:0] r2udNeg_4 = udNeg_4[26:15]; // @[FDIV.scala 430:51]
  wire  signBit_8 = rudNeg_0[9]; // @[FDIV.scala 132:20]
  wire [9:0] _T_82 = {signBit_8,rudNeg_0[9:1]}; // @[Cat.scala 31:58]
  wire [9:0] rudPmNeg_0_0 = _T_82 + mNeg_0[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_0_1 = _T_82 + mNeg_1[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_0_2 = _T_82 + mNeg_2[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_0_3 = _T_82 + mNeg_3[10:1]; // @[FDIV.scala 432:103]
  wire  signBit_12 = rudNeg_1[9]; // @[FDIV.scala 132:20]
  wire [9:0] _T_102 = {signBit_12,rudNeg_1[9:1]}; // @[Cat.scala 31:58]
  wire [9:0] rudPmNeg_1_0 = _T_102 + mNeg_0[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_1_1 = _T_102 + mNeg_1[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_1_2 = _T_102 + mNeg_2[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_1_3 = _T_102 + mNeg_3[10:1]; // @[FDIV.scala 432:103]
  wire [10:0] _T_124 = {{1'd0}, mNeg_0[10:1]}; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_2_0 = _T_124[9:0]; // @[FDIV.scala 432:103]
  wire [10:0] _T_129 = {{1'd0}, mNeg_1[10:1]}; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_2_1 = _T_129[9:0]; // @[FDIV.scala 432:103]
  wire [10:0] _T_134 = {{1'd0}, mNeg_2[10:1]}; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_2_2 = _T_134[9:0]; // @[FDIV.scala 432:103]
  wire [10:0] _T_139 = {{1'd0}, mNeg_3[10:1]}; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_2_3 = _T_139[9:0]; // @[FDIV.scala 432:103]
  wire  signBit_20 = rudNeg_3[9]; // @[FDIV.scala 132:20]
  wire [9:0] _T_142 = {signBit_20,rudNeg_3[9:1]}; // @[Cat.scala 31:58]
  wire [9:0] rudPmNeg_3_0 = _T_142 + mNeg_0[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_3_1 = _T_142 + mNeg_1[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_3_2 = _T_142 + mNeg_2[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_3_3 = _T_142 + mNeg_3[10:1]; // @[FDIV.scala 432:103]
  wire  signBit_24 = rudNeg_4[9]; // @[FDIV.scala 132:20]
  wire [9:0] _T_162 = {signBit_24,rudNeg_4[9:1]}; // @[Cat.scala 31:58]
  wire [9:0] rudPmNeg_4_0 = _T_162 + mNeg_0[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_4_1 = _T_162 + mNeg_1[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_4_2 = _T_162 + mNeg_2[10:1]; // @[FDIV.scala 432:103]
  wire [9:0] rudPmNeg_4_3 = _T_162 + mNeg_3[10:1]; // @[FDIV.scala 432:103]
  wire  signBit_28 = r2udNeg_0[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_181 = {signBit_28,r2udNeg_0}; // @[Cat.scala 31:58]
  wire  signBit_29 = mNeg_0[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_182 = {signBit_29,_T_16,orMatrixOutputs[5],orMatrixOutputs[4],orMatrixOutputs[3],orMatrixOutputs[2],
    orMatrixOutputs[1],orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_0_0 = _T_181 + _T_182; // @[FDIV.scala 433:99]
  wire  signBit_31 = mNeg_1[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_186 = {signBit_31,_T_26,orMatrixOutputs_1[5],orMatrixOutputs_1[4],orMatrixOutputs_1[3],
    orMatrixOutputs_1[2],orMatrixOutputs_1[1],orMatrixOutputs_1[0],2'h0}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_0_1 = _T_181 + _T_186; // @[FDIV.scala 433:99]
  wire  signBit_33 = mNeg_2[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_190 = {signBit_33,_T_34,smallerThanM1_orMatrixOutputs[5],smallerThanM1_orMatrixOutputs[4],
    smallerThanM1_orMatrixOutputs[3],smallerThanM1_orMatrixOutputs[2],smallerThanM1_orMatrixOutputs[1],
    smallerThanM1_orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_0_2 = _T_181 + _T_190; // @[FDIV.scala 433:99]
  wire  signBit_35 = mNeg_3[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_194 = {signBit_35,_T_56,smallerThanM2_orMatrixOutputs[5],smallerThanM2_orMatrixOutputs[4],
    smallerThanM2_orMatrixOutputs[3],smallerThanM2_orMatrixOutputs[2],smallerThanM2_orMatrixOutputs[1],
    smallerThanM2_orMatrixOutputs[0],2'h0}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_0_3 = _T_181 + _T_194; // @[FDIV.scala 433:99]
  wire  signBit_36 = r2udNeg_1[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_197 = {signBit_36,r2udNeg_1}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_1_0 = _T_197 + _T_182; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_1_1 = _T_197 + _T_186; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_1_2 = _T_197 + _T_190; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_1_3 = _T_197 + _T_194; // @[FDIV.scala 433:99]
  wire [13:0] _T_215 = {{1'd0}, _T_182}; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_2_0 = _T_215[12:0]; // @[FDIV.scala 433:99]
  wire [13:0] _T_219 = {{1'd0}, _T_186}; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_2_1 = _T_219[12:0]; // @[FDIV.scala 433:99]
  wire [13:0] _T_223 = {{1'd0}, _T_190}; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_2_2 = _T_223[12:0]; // @[FDIV.scala 433:99]
  wire [13:0] _T_227 = {{1'd0}, _T_194}; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_2_3 = _T_227[12:0]; // @[FDIV.scala 433:99]
  wire  signBit_52 = r2udNeg_3[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_229 = {signBit_52,r2udNeg_3}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_3_0 = _T_229 + _T_182; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_3_1 = _T_229 + _T_186; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_3_2 = _T_229 + _T_190; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_3_3 = _T_229 + _T_194; // @[FDIV.scala 433:99]
  wire  signBit_60 = r2udNeg_4[11]; // @[FDIV.scala 132:20]
  wire [12:0] _T_245 = {signBit_60,r2udNeg_4}; // @[Cat.scala 31:58]
  wire [12:0] r2udPmNeg_4_0 = _T_245 + _T_182; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_4_1 = _T_245 + _T_186; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_4_2 = _T_245 + _T_190; // @[FDIV.scala 433:99]
  wire [12:0] r2udPmNeg_4_3 = _T_245 + _T_194; // @[FDIV.scala 433:99]
  reg [27:0] udNegReg_0; // @[Reg.scala 16:16]
  reg [27:0] udNegReg_1; // @[Reg.scala 16:16]
  reg [27:0] udNegReg_3; // @[Reg.scala 16:16]
  reg [27:0] udNegReg_4; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_0_0; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_0_1; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_0_2; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_0_3; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_1_0; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_1_1; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_1_2; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_1_3; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_2_0; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_2_1; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_2_2; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_2_3; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_3_0; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_3_1; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_3_2; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_3_3; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_4_0; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_4_1; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_4_2; // @[Reg.scala 16:16]
  reg [9:0] rudPmNegReg_4_3; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_0_0; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_0_1; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_0_2; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_0_3; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_1_0; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_1_1; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_1_2; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_1_3; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_2_0; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_2_1; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_2_2; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_2_3; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_3_0; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_3_1; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_3_2; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_3_3; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_4_0; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_4_1; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_4_2; // @[Reg.scala 16:16]
  reg [12:0] r2udPmNegReg_4_3; // @[Reg.scala 16:16]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_5 = qPrevReg[0] ? rudPmNegReg_0_0 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_6 = qPrevReg[1] ? rudPmNegReg_1_0 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_7 = qPrevReg[2] ? rudPmNegReg_2_0 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_8 = qPrevReg[3] ? rudPmNegReg_3_0 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_9 = qPrevReg[4] ? rudPmNegReg_4_0 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_10 = _signs_csa_sel_0_io_in_2_T_5 | _signs_csa_sel_0_io_in_2_T_6; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_11 = _signs_csa_sel_0_io_in_2_T_10 | _signs_csa_sel_0_io_in_2_T_7; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_12 = _signs_csa_sel_0_io_in_2_T_11 | _signs_csa_sel_0_io_in_2_T_8; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_14 = qPrevReg[0] ? rudPmNegReg_0_1 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_15 = qPrevReg[1] ? rudPmNegReg_1_1 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_16 = qPrevReg[2] ? rudPmNegReg_2_1 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_17 = qPrevReg[3] ? rudPmNegReg_3_1 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_18 = qPrevReg[4] ? rudPmNegReg_4_1 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_19 = _signs_csa_sel_0_io_in_2_T_14 | _signs_csa_sel_0_io_in_2_T_15; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_20 = _signs_csa_sel_0_io_in_2_T_19 | _signs_csa_sel_0_io_in_2_T_16; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_21 = _signs_csa_sel_0_io_in_2_T_20 | _signs_csa_sel_0_io_in_2_T_17; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_23 = qPrevReg[0] ? rudPmNegReg_0_2 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_24 = qPrevReg[1] ? rudPmNegReg_1_2 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_25 = qPrevReg[2] ? rudPmNegReg_2_2 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_26 = qPrevReg[3] ? rudPmNegReg_3_2 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_27 = qPrevReg[4] ? rudPmNegReg_4_2 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_28 = _signs_csa_sel_0_io_in_2_T_23 | _signs_csa_sel_0_io_in_2_T_24; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_29 = _signs_csa_sel_0_io_in_2_T_28 | _signs_csa_sel_0_io_in_2_T_25; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_30 = _signs_csa_sel_0_io_in_2_T_29 | _signs_csa_sel_0_io_in_2_T_26; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_32 = qPrevReg[0] ? rudPmNegReg_0_3 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_33 = qPrevReg[1] ? rudPmNegReg_1_3 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_34 = qPrevReg[2] ? rudPmNegReg_2_3 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_35 = qPrevReg[3] ? rudPmNegReg_3_3 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_36 = qPrevReg[4] ? rudPmNegReg_4_3 : 10'h0; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_37 = _signs_csa_sel_0_io_in_2_T_32 | _signs_csa_sel_0_io_in_2_T_33; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_38 = _signs_csa_sel_0_io_in_2_T_37 | _signs_csa_sel_0_io_in_2_T_34; // @[Mux.scala 27:73]
  wire [9:0] _signs_csa_sel_0_io_in_2_T_39 = _signs_csa_sel_0_io_in_2_T_38 | _signs_csa_sel_0_io_in_2_T_35; // @[Mux.scala 27:73]
  wire [29:0] _csa_sel_wide_1_io_in_0_T = {wsReg, 2'h0}; // @[FDIV.scala 457:30]
  wire [29:0] _csa_sel_wide_1_io_in_1_T = {wcReg, 2'h0}; // @[FDIV.scala 458:30]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_5 = qPrevReg[0] ? udNegReg_0 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_6 = qPrevReg[1] ? udNegReg_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_8 = qPrevReg[3] ? udNegReg_3 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_9 = qPrevReg[4] ? udNegReg_4 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_10 = _csa_sel_wide_1_io_in_2_T_5 | _csa_sel_wide_1_io_in_2_T_6; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_12 = _csa_sel_wide_1_io_in_2_T_10 | _csa_sel_wide_1_io_in_2_T_8; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_1_io_in_2_T_13 = _csa_sel_wide_1_io_in_2_T_12 | _csa_sel_wide_1_io_in_2_T_9; // @[Mux.scala 27:73]
  wire [29:0] _csa_sel_wide_1_io_in_2_T_14 = {_csa_sel_wide_1_io_in_2_T_13, 2'h0}; // @[FDIV.scala 459:56]
  wire [29:0] _csa_sel_wide_2_io_in_0_T = {csa_sel_wide_1_io_out_0, 2'h0}; // @[FDIV.scala 460:43]
  wire [29:0] _csa_sel_wide_2_io_in_1_T_2 = {_wcIter_T[27:0], 2'h0}; // @[FDIV.scala 461:64]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_5 = qNext[0] ? udNegReg_0 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_6 = qNext[1] ? udNegReg_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_8 = qNext[3] ? udNegReg_3 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_9 = qNext[4] ? udNegReg_4 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_10 = _csa_sel_wide_2_io_in_2_T_5 | _csa_sel_wide_2_io_in_2_T_6; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_12 = _csa_sel_wide_2_io_in_2_T_10 | _csa_sel_wide_2_io_in_2_T_8; // @[Mux.scala 27:73]
  wire [27:0] _csa_sel_wide_2_io_in_2_T_13 = _csa_sel_wide_2_io_in_2_T_12 | _csa_sel_wide_2_io_in_2_T_9; // @[Mux.scala 27:73]
  wire [29:0] _csa_sel_wide_2_io_in_2_T_14 = {_csa_sel_wide_2_io_in_2_T_13, 2'h0}; // @[FDIV.scala 462:53]
  wire  csa_spec_0_io_in_2_signBit = udNegReg_0[26]; // @[FDIV.scala 132:20]
  wire [2:0] _csa_spec_0_io_in_2_T_2 = csa_spec_0_io_in_2_signBit ? 3'h7 : 3'h0; // @[Bitwise.scala 74:12]
  wire [13:0] _signs2_csa_spec_0_0_io_in_1_T = {csa_spec_0_io_out_1, 1'h0}; // @[FDIV.scala 476:40]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_5 = qPrevReg[0] ? r2udPmNegReg_0_0 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_6 = qPrevReg[1] ? r2udPmNegReg_1_0 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_7 = qPrevReg[2] ? r2udPmNegReg_2_0 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_8 = qPrevReg[3] ? r2udPmNegReg_3_0 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_9 = qPrevReg[4] ? r2udPmNegReg_4_0 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_10 = _signs2_csa_spec_0_0_io_in_2_T_5 | _signs2_csa_spec_0_0_io_in_2_T_6; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_11 = _signs2_csa_spec_0_0_io_in_2_T_10 | _signs2_csa_spec_0_0_io_in_2_T_7; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_12 = _signs2_csa_spec_0_0_io_in_2_T_11 | _signs2_csa_spec_0_0_io_in_2_T_8; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_14 = qPrevReg[0] ? r2udPmNegReg_0_1 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_15 = qPrevReg[1] ? r2udPmNegReg_1_1 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_16 = qPrevReg[2] ? r2udPmNegReg_2_1 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_17 = qPrevReg[3] ? r2udPmNegReg_3_1 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_18 = qPrevReg[4] ? r2udPmNegReg_4_1 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_19 = _signs2_csa_spec_0_0_io_in_2_T_14 | _signs2_csa_spec_0_0_io_in_2_T_15; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_20 = _signs2_csa_spec_0_0_io_in_2_T_19 | _signs2_csa_spec_0_0_io_in_2_T_16; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_21 = _signs2_csa_spec_0_0_io_in_2_T_20 | _signs2_csa_spec_0_0_io_in_2_T_17; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_23 = qPrevReg[0] ? r2udPmNegReg_0_2 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_24 = qPrevReg[1] ? r2udPmNegReg_1_2 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_25 = qPrevReg[2] ? r2udPmNegReg_2_2 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_26 = qPrevReg[3] ? r2udPmNegReg_3_2 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_27 = qPrevReg[4] ? r2udPmNegReg_4_2 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_28 = _signs2_csa_spec_0_0_io_in_2_T_23 | _signs2_csa_spec_0_0_io_in_2_T_24; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_29 = _signs2_csa_spec_0_0_io_in_2_T_28 | _signs2_csa_spec_0_0_io_in_2_T_25; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_30 = _signs2_csa_spec_0_0_io_in_2_T_29 | _signs2_csa_spec_0_0_io_in_2_T_26; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_32 = qPrevReg[0] ? r2udPmNegReg_0_3 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_33 = qPrevReg[1] ? r2udPmNegReg_1_3 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_34 = qPrevReg[2] ? r2udPmNegReg_2_3 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_35 = qPrevReg[3] ? r2udPmNegReg_3_3 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_36 = qPrevReg[4] ? r2udPmNegReg_4_3 : 13'h0; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_37 = _signs2_csa_spec_0_0_io_in_2_T_32 | _signs2_csa_spec_0_0_io_in_2_T_33; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_38 = _signs2_csa_spec_0_0_io_in_2_T_37 | _signs2_csa_spec_0_0_io_in_2_T_34; // @[Mux.scala 27:73]
  wire [12:0] _signs2_csa_spec_0_0_io_in_2_T_39 = _signs2_csa_spec_0_0_io_in_2_T_38 | _signs2_csa_spec_0_0_io_in_2_T_35; // @[Mux.scala 27:73]
  wire  csa_spec_1_io_in_2_signBit = udNegReg_1[26]; // @[FDIV.scala 132:20]
  wire [2:0] _csa_spec_1_io_in_2_T_2 = csa_spec_1_io_in_2_signBit ? 3'h7 : 3'h0; // @[Bitwise.scala 74:12]
  wire [13:0] _signs2_csa_spec_1_0_io_in_1_T = {csa_spec_1_io_out_1, 1'h0}; // @[FDIV.scala 476:40]
  wire [13:0] _signs2_csa_spec_2_0_io_in_1_T = {csa_spec_2_io_out_1, 1'h0}; // @[FDIV.scala 476:40]
  wire  csa_spec_3_io_in_2_signBit = udNegReg_3[26]; // @[FDIV.scala 132:20]
  wire [2:0] _csa_spec_3_io_in_2_T_2 = csa_spec_3_io_in_2_signBit ? 3'h7 : 3'h0; // @[Bitwise.scala 74:12]
  wire [13:0] _signs2_csa_spec_3_0_io_in_1_T = {csa_spec_3_io_out_1, 1'h0}; // @[FDIV.scala 476:40]
  wire  csa_spec_4_io_in_2_signBit = udNegReg_4[26]; // @[FDIV.scala 132:20]
  wire [2:0] _csa_spec_4_io_in_2_T_2 = csa_spec_4_io_in_2_signBit ? 3'h7 : 3'h0; // @[Bitwise.scala 74:12]
  wire [13:0] _signs2_csa_spec_4_0_io_in_1_T = {csa_spec_4_io_out_1, 1'h0}; // @[FDIV.scala 476:40]
  CSA3_2 signs_csa_sel_0 ( // @[FDIV.scala 447:21]
    .io_in_0(signs_csa_sel_0_io_in_0),
    .io_in_1(signs_csa_sel_0_io_in_1),
    .io_in_2(signs_csa_sel_0_io_in_2),
    .io_out_0(signs_csa_sel_0_io_out_0),
    .io_out_1(signs_csa_sel_0_io_out_1)
  );
  CSA3_2 signs_csa_sel_1 ( // @[FDIV.scala 447:21]
    .io_in_0(signs_csa_sel_1_io_in_0),
    .io_in_1(signs_csa_sel_1_io_in_1),
    .io_in_2(signs_csa_sel_1_io_in_2),
    .io_out_0(signs_csa_sel_1_io_out_0),
    .io_out_1(signs_csa_sel_1_io_out_1)
  );
  CSA3_2 signs_csa_sel_2 ( // @[FDIV.scala 447:21]
    .io_in_0(signs_csa_sel_2_io_in_0),
    .io_in_1(signs_csa_sel_2_io_in_1),
    .io_in_2(signs_csa_sel_2_io_in_2),
    .io_out_0(signs_csa_sel_2_io_out_0),
    .io_out_1(signs_csa_sel_2_io_out_1)
  );
  CSA3_2 signs_csa_sel_3 ( // @[FDIV.scala 447:21]
    .io_in_0(signs_csa_sel_3_io_in_0),
    .io_in_1(signs_csa_sel_3_io_in_1),
    .io_in_2(signs_csa_sel_3_io_in_2),
    .io_out_0(signs_csa_sel_3_io_out_0),
    .io_out_1(signs_csa_sel_3_io_out_1)
  );
  CSA3_2_4 csa_sel_wide_1 ( // @[FDIV.scala 455:24]
    .io_in_0(csa_sel_wide_1_io_in_0),
    .io_in_1(csa_sel_wide_1_io_in_1),
    .io_in_2(csa_sel_wide_1_io_in_2),
    .io_out_0(csa_sel_wide_1_io_out_0),
    .io_out_1(csa_sel_wide_1_io_out_1)
  );
  CSA3_2_4 csa_sel_wide_2 ( // @[FDIV.scala 456:24]
    .io_in_0(csa_sel_wide_2_io_in_0),
    .io_in_1(csa_sel_wide_2_io_in_1),
    .io_in_2(csa_sel_wide_2_io_in_2),
    .io_out_0(csa_sel_wide_2_io_out_0),
    .io_out_1(csa_sel_wide_2_io_out_1)
  );
  CSA3_2_6 csa_spec_0 ( // @[FDIV.scala 469:22]
    .io_in_0(csa_spec_0_io_in_0),
    .io_in_1(csa_spec_0_io_in_1),
    .io_in_2(csa_spec_0_io_in_2),
    .io_out_0(csa_spec_0_io_out_0),
    .io_out_1(csa_spec_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_0_0 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_0_0_io_in_0),
    .io_in_1(signs2_csa_spec_0_0_io_in_1),
    .io_in_2(signs2_csa_spec_0_0_io_in_2),
    .io_out_0(signs2_csa_spec_0_0_io_out_0),
    .io_out_1(signs2_csa_spec_0_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_0_1 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_0_1_io_in_0),
    .io_in_1(signs2_csa_spec_0_1_io_in_1),
    .io_in_2(signs2_csa_spec_0_1_io_in_2),
    .io_out_0(signs2_csa_spec_0_1_io_out_0),
    .io_out_1(signs2_csa_spec_0_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_0_2 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_0_2_io_in_0),
    .io_in_1(signs2_csa_spec_0_2_io_in_1),
    .io_in_2(signs2_csa_spec_0_2_io_in_2),
    .io_out_0(signs2_csa_spec_0_2_io_out_0),
    .io_out_1(signs2_csa_spec_0_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_0_3 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_0_3_io_in_0),
    .io_in_1(signs2_csa_spec_0_3_io_in_1),
    .io_in_2(signs2_csa_spec_0_3_io_in_2),
    .io_out_0(signs2_csa_spec_0_3_io_out_0),
    .io_out_1(signs2_csa_spec_0_3_io_out_1)
  );
  CSA3_2_6 csa_spec_1 ( // @[FDIV.scala 469:22]
    .io_in_0(csa_spec_1_io_in_0),
    .io_in_1(csa_spec_1_io_in_1),
    .io_in_2(csa_spec_1_io_in_2),
    .io_out_0(csa_spec_1_io_out_0),
    .io_out_1(csa_spec_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_1_0 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_1_0_io_in_0),
    .io_in_1(signs2_csa_spec_1_0_io_in_1),
    .io_in_2(signs2_csa_spec_1_0_io_in_2),
    .io_out_0(signs2_csa_spec_1_0_io_out_0),
    .io_out_1(signs2_csa_spec_1_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_1_1 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_1_1_io_in_0),
    .io_in_1(signs2_csa_spec_1_1_io_in_1),
    .io_in_2(signs2_csa_spec_1_1_io_in_2),
    .io_out_0(signs2_csa_spec_1_1_io_out_0),
    .io_out_1(signs2_csa_spec_1_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_1_2 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_1_2_io_in_0),
    .io_in_1(signs2_csa_spec_1_2_io_in_1),
    .io_in_2(signs2_csa_spec_1_2_io_in_2),
    .io_out_0(signs2_csa_spec_1_2_io_out_0),
    .io_out_1(signs2_csa_spec_1_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_1_3 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_1_3_io_in_0),
    .io_in_1(signs2_csa_spec_1_3_io_in_1),
    .io_in_2(signs2_csa_spec_1_3_io_in_2),
    .io_out_0(signs2_csa_spec_1_3_io_out_0),
    .io_out_1(signs2_csa_spec_1_3_io_out_1)
  );
  CSA3_2_6 csa_spec_2 ( // @[FDIV.scala 469:22]
    .io_in_0(csa_spec_2_io_in_0),
    .io_in_1(csa_spec_2_io_in_1),
    .io_in_2(csa_spec_2_io_in_2),
    .io_out_0(csa_spec_2_io_out_0),
    .io_out_1(csa_spec_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_2_0 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_2_0_io_in_0),
    .io_in_1(signs2_csa_spec_2_0_io_in_1),
    .io_in_2(signs2_csa_spec_2_0_io_in_2),
    .io_out_0(signs2_csa_spec_2_0_io_out_0),
    .io_out_1(signs2_csa_spec_2_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_2_1 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_2_1_io_in_0),
    .io_in_1(signs2_csa_spec_2_1_io_in_1),
    .io_in_2(signs2_csa_spec_2_1_io_in_2),
    .io_out_0(signs2_csa_spec_2_1_io_out_0),
    .io_out_1(signs2_csa_spec_2_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_2_2 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_2_2_io_in_0),
    .io_in_1(signs2_csa_spec_2_2_io_in_1),
    .io_in_2(signs2_csa_spec_2_2_io_in_2),
    .io_out_0(signs2_csa_spec_2_2_io_out_0),
    .io_out_1(signs2_csa_spec_2_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_2_3 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_2_3_io_in_0),
    .io_in_1(signs2_csa_spec_2_3_io_in_1),
    .io_in_2(signs2_csa_spec_2_3_io_in_2),
    .io_out_0(signs2_csa_spec_2_3_io_out_0),
    .io_out_1(signs2_csa_spec_2_3_io_out_1)
  );
  CSA3_2_6 csa_spec_3 ( // @[FDIV.scala 469:22]
    .io_in_0(csa_spec_3_io_in_0),
    .io_in_1(csa_spec_3_io_in_1),
    .io_in_2(csa_spec_3_io_in_2),
    .io_out_0(csa_spec_3_io_out_0),
    .io_out_1(csa_spec_3_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_3_0 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_3_0_io_in_0),
    .io_in_1(signs2_csa_spec_3_0_io_in_1),
    .io_in_2(signs2_csa_spec_3_0_io_in_2),
    .io_out_0(signs2_csa_spec_3_0_io_out_0),
    .io_out_1(signs2_csa_spec_3_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_3_1 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_3_1_io_in_0),
    .io_in_1(signs2_csa_spec_3_1_io_in_1),
    .io_in_2(signs2_csa_spec_3_1_io_in_2),
    .io_out_0(signs2_csa_spec_3_1_io_out_0),
    .io_out_1(signs2_csa_spec_3_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_3_2 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_3_2_io_in_0),
    .io_in_1(signs2_csa_spec_3_2_io_in_1),
    .io_in_2(signs2_csa_spec_3_2_io_in_2),
    .io_out_0(signs2_csa_spec_3_2_io_out_0),
    .io_out_1(signs2_csa_spec_3_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_3_3 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_3_3_io_in_0),
    .io_in_1(signs2_csa_spec_3_3_io_in_1),
    .io_in_2(signs2_csa_spec_3_3_io_in_2),
    .io_out_0(signs2_csa_spec_3_3_io_out_0),
    .io_out_1(signs2_csa_spec_3_3_io_out_1)
  );
  CSA3_2_6 csa_spec_4 ( // @[FDIV.scala 469:22]
    .io_in_0(csa_spec_4_io_in_0),
    .io_in_1(csa_spec_4_io_in_1),
    .io_in_2(csa_spec_4_io_in_2),
    .io_out_0(csa_spec_4_io_out_0),
    .io_out_1(csa_spec_4_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_4_0 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_4_0_io_in_0),
    .io_in_1(signs2_csa_spec_4_0_io_in_1),
    .io_in_2(signs2_csa_spec_4_0_io_in_2),
    .io_out_0(signs2_csa_spec_4_0_io_out_0),
    .io_out_1(signs2_csa_spec_4_0_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_4_1 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_4_1_io_in_0),
    .io_in_1(signs2_csa_spec_4_1_io_in_1),
    .io_in_2(signs2_csa_spec_4_1_io_in_2),
    .io_out_0(signs2_csa_spec_4_1_io_out_0),
    .io_out_1(signs2_csa_spec_4_1_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_4_2 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_4_2_io_in_0),
    .io_in_1(signs2_csa_spec_4_2_io_in_1),
    .io_in_2(signs2_csa_spec_4_2_io_in_2),
    .io_out_0(signs2_csa_spec_4_2_io_out_0),
    .io_out_1(signs2_csa_spec_4_2_io_out_1)
  );
  CSA3_2_6 signs2_csa_spec_4_3 ( // @[FDIV.scala 474:24]
    .io_in_0(signs2_csa_spec_4_3_io_in_0),
    .io_in_1(signs2_csa_spec_4_3_io_in_1),
    .io_in_2(signs2_csa_spec_4_3_io_in_2),
    .io_out_0(signs2_csa_spec_4_3_io_out_0),
    .io_out_1(signs2_csa_spec_4_3_io_out_1)
  );
  assign io_rem = wsReg + wcReg; // @[FDIV.scala 500:19]
  assign io_quot = quotIterReg; // @[FDIV.scala 501:11]
  assign io_quotM1 = quotM1IterReg; // @[FDIV.scala 502:13]
  assign signs_csa_sel_0_io_in_0 = wsReg[27:18]; // @[FDIV.scala 437:16]
  assign signs_csa_sel_0_io_in_1 = wcReg[27:18]; // @[FDIV.scala 438:16]
  assign signs_csa_sel_0_io_in_2 = _signs_csa_sel_0_io_in_2_T_12 | _signs_csa_sel_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs_csa_sel_1_io_in_0 = wsReg[27:18]; // @[FDIV.scala 437:16]
  assign signs_csa_sel_1_io_in_1 = wcReg[27:18]; // @[FDIV.scala 438:16]
  assign signs_csa_sel_1_io_in_2 = _signs_csa_sel_0_io_in_2_T_21 | _signs_csa_sel_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs_csa_sel_2_io_in_0 = wsReg[27:18]; // @[FDIV.scala 437:16]
  assign signs_csa_sel_2_io_in_1 = wcReg[27:18]; // @[FDIV.scala 438:16]
  assign signs_csa_sel_2_io_in_2 = _signs_csa_sel_0_io_in_2_T_30 | _signs_csa_sel_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs_csa_sel_3_io_in_0 = wsReg[27:18]; // @[FDIV.scala 437:16]
  assign signs_csa_sel_3_io_in_1 = wcReg[27:18]; // @[FDIV.scala 438:16]
  assign signs_csa_sel_3_io_in_2 = _signs_csa_sel_0_io_in_2_T_39 | _signs_csa_sel_0_io_in_2_T_36; // @[Mux.scala 27:73]
  assign csa_sel_wide_1_io_in_0 = _csa_sel_wide_1_io_in_0_T[27:0]; // @[FDIV.scala 457:21]
  assign csa_sel_wide_1_io_in_1 = _csa_sel_wide_1_io_in_1_T[27:0]; // @[FDIV.scala 458:21]
  assign csa_sel_wide_1_io_in_2 = _csa_sel_wide_1_io_in_2_T_14[27:0]; // @[FDIV.scala 459:21]
  assign csa_sel_wide_2_io_in_0 = _csa_sel_wide_2_io_in_0_T[27:0]; // @[FDIV.scala 460:21]
  assign csa_sel_wide_2_io_in_1 = _csa_sel_wide_2_io_in_1_T_2[27:0]; // @[FDIV.scala 461:21]
  assign csa_sel_wide_2_io_in_2 = _csa_sel_wide_2_io_in_2_T_14[27:0]; // @[FDIV.scala 462:21]
  assign csa_spec_0_io_in_0 = wsReg[27:15]; // @[FDIV.scala 434:16]
  assign csa_spec_0_io_in_1 = wcReg[27:15]; // @[FDIV.scala 435:16]
  assign csa_spec_0_io_in_2 = {_csa_spec_0_io_in_2_T_2,udNegReg_0[26:17]}; // @[Cat.scala 31:58]
  assign signs2_csa_spec_0_0_io_in_0 = csa_spec_0_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_0_0_io_in_1 = _signs2_csa_spec_0_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_0_0_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_12 | _signs2_csa_spec_0_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs2_csa_spec_0_1_io_in_0 = csa_spec_0_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_0_1_io_in_1 = _signs2_csa_spec_0_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_0_1_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_21 | _signs2_csa_spec_0_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs2_csa_spec_0_2_io_in_0 = csa_spec_0_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_0_2_io_in_1 = _signs2_csa_spec_0_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_0_2_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_30 | _signs2_csa_spec_0_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs2_csa_spec_0_3_io_in_0 = csa_spec_0_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_0_3_io_in_1 = _signs2_csa_spec_0_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_0_3_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_39 | _signs2_csa_spec_0_0_io_in_2_T_36; // @[Mux.scala 27:73]
  assign csa_spec_1_io_in_0 = wsReg[27:15]; // @[FDIV.scala 434:16]
  assign csa_spec_1_io_in_1 = wcReg[27:15]; // @[FDIV.scala 435:16]
  assign csa_spec_1_io_in_2 = {_csa_spec_1_io_in_2_T_2,udNegReg_1[26:17]}; // @[Cat.scala 31:58]
  assign signs2_csa_spec_1_0_io_in_0 = csa_spec_1_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_1_0_io_in_1 = _signs2_csa_spec_1_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_1_0_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_12 | _signs2_csa_spec_0_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs2_csa_spec_1_1_io_in_0 = csa_spec_1_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_1_1_io_in_1 = _signs2_csa_spec_1_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_1_1_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_21 | _signs2_csa_spec_0_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs2_csa_spec_1_2_io_in_0 = csa_spec_1_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_1_2_io_in_1 = _signs2_csa_spec_1_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_1_2_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_30 | _signs2_csa_spec_0_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs2_csa_spec_1_3_io_in_0 = csa_spec_1_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_1_3_io_in_1 = _signs2_csa_spec_1_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_1_3_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_39 | _signs2_csa_spec_0_0_io_in_2_T_36; // @[Mux.scala 27:73]
  assign csa_spec_2_io_in_0 = wsReg[27:15]; // @[FDIV.scala 434:16]
  assign csa_spec_2_io_in_1 = wcReg[27:15]; // @[FDIV.scala 435:16]
  assign csa_spec_2_io_in_2 = 13'h0; // @[Cat.scala 31:58]
  assign signs2_csa_spec_2_0_io_in_0 = csa_spec_2_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_2_0_io_in_1 = _signs2_csa_spec_2_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_2_0_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_12 | _signs2_csa_spec_0_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs2_csa_spec_2_1_io_in_0 = csa_spec_2_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_2_1_io_in_1 = _signs2_csa_spec_2_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_2_1_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_21 | _signs2_csa_spec_0_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs2_csa_spec_2_2_io_in_0 = csa_spec_2_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_2_2_io_in_1 = _signs2_csa_spec_2_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_2_2_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_30 | _signs2_csa_spec_0_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs2_csa_spec_2_3_io_in_0 = csa_spec_2_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_2_3_io_in_1 = _signs2_csa_spec_2_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_2_3_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_39 | _signs2_csa_spec_0_0_io_in_2_T_36; // @[Mux.scala 27:73]
  assign csa_spec_3_io_in_0 = wsReg[27:15]; // @[FDIV.scala 434:16]
  assign csa_spec_3_io_in_1 = wcReg[27:15]; // @[FDIV.scala 435:16]
  assign csa_spec_3_io_in_2 = {_csa_spec_3_io_in_2_T_2,udNegReg_3[26:17]}; // @[Cat.scala 31:58]
  assign signs2_csa_spec_3_0_io_in_0 = csa_spec_3_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_3_0_io_in_1 = _signs2_csa_spec_3_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_3_0_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_12 | _signs2_csa_spec_0_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs2_csa_spec_3_1_io_in_0 = csa_spec_3_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_3_1_io_in_1 = _signs2_csa_spec_3_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_3_1_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_21 | _signs2_csa_spec_0_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs2_csa_spec_3_2_io_in_0 = csa_spec_3_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_3_2_io_in_1 = _signs2_csa_spec_3_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_3_2_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_30 | _signs2_csa_spec_0_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs2_csa_spec_3_3_io_in_0 = csa_spec_3_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_3_3_io_in_1 = _signs2_csa_spec_3_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_3_3_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_39 | _signs2_csa_spec_0_0_io_in_2_T_36; // @[Mux.scala 27:73]
  assign csa_spec_4_io_in_0 = wsReg[27:15]; // @[FDIV.scala 434:16]
  assign csa_spec_4_io_in_1 = wcReg[27:15]; // @[FDIV.scala 435:16]
  assign csa_spec_4_io_in_2 = {_csa_spec_4_io_in_2_T_2,udNegReg_4[26:17]}; // @[Cat.scala 31:58]
  assign signs2_csa_spec_4_0_io_in_0 = csa_spec_4_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_4_0_io_in_1 = _signs2_csa_spec_4_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_4_0_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_12 | _signs2_csa_spec_0_0_io_in_2_T_9; // @[Mux.scala 27:73]
  assign signs2_csa_spec_4_1_io_in_0 = csa_spec_4_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_4_1_io_in_1 = _signs2_csa_spec_4_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_4_1_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_21 | _signs2_csa_spec_0_0_io_in_2_T_18; // @[Mux.scala 27:73]
  assign signs2_csa_spec_4_2_io_in_0 = csa_spec_4_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_4_2_io_in_1 = _signs2_csa_spec_4_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_4_2_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_30 | _signs2_csa_spec_0_0_io_in_2_T_27; // @[Mux.scala 27:73]
  assign signs2_csa_spec_4_3_io_in_0 = csa_spec_4_io_out_0; // @[FDIV.scala 475:21]
  assign signs2_csa_spec_4_3_io_in_1 = _signs2_csa_spec_4_0_io_in_1_T[12:0]; // @[FDIV.scala 476:45]
  assign signs2_csa_spec_4_3_io_in_2 = _signs2_csa_spec_0_0_io_in_2_T_39 | _signs2_csa_spec_0_0_io_in_2_T_36; // @[Mux.scala 27:73]
  always @(posedge clock) begin
    if (_qPrevReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 410:31]
        if (smallerThanM1) begin // @[FDIV.scala 375:18]
          qPrevReg <= 8'h4;
        end else if (smallerThanM2) begin // @[FDIV.scala 375:55]
          qPrevReg <= 8'h8;
        end else begin
          qPrevReg <= 8'h10;
        end
      end else begin
        qPrevReg <= {{3'd0}, qNext2};
      end
    end
    if (_qPrevReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 411:28]
        if (io_sigCmp) begin // @[FDIV.scala 367:19]
          wsReg <= _wsInit_T;
        end else begin
          wsReg <= _wsInit_T_1;
        end
      end else if (io_lastIterDoHalf) begin // @[FDIV.scala 463:16]
        wsReg <= csa_sel_wide_1_io_out_0;
      end else begin
        wsReg <= csa_sel_wide_2_io_out_0;
      end
    end
    if (_qPrevReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 412:28]
        wcReg <= 28'h0;
      end else if (io_lastIterDoHalf) begin // @[FDIV.scala 464:16]
        wcReg <= _wcIter_T[27:0];
      end else begin
        wcReg <= _wcIter_T_2[27:0];
      end
    end
    if (_qPrevReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[1]) begin // @[FDIV.scala 497:18]
        if (io_lastIterDoHalf) begin // @[FDIV.scala 494:22]
          quotIterReg <= quotHalfIter;
        end else begin
          quotIterReg <= quotIterNext_quotNext[25:0];
        end
      end else begin
        quotIterReg <= 26'h0;
      end
    end
    if (_qPrevReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[1]) begin // @[FDIV.scala 498:20]
        if (io_lastIterDoHalf) begin // @[FDIV.scala 495:24]
          quotM1IterReg <= quotM1HalfIter;
        end else begin
          quotM1IterReg <= quotM1IterNext_quotM1Next[25:0];
        end
      end else begin
        quotM1IterReg <= 26'h0;
      end
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      udNegReg_0 <= udNeg_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      udNegReg_1 <= udNeg_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      udNegReg_3 <= udNeg_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      udNegReg_4 <= udNeg_4; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_0_0 <= rudPmNeg_0_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_0_1 <= rudPmNeg_0_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_0_2 <= rudPmNeg_0_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_0_3 <= rudPmNeg_0_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_1_0 <= rudPmNeg_1_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_1_1 <= rudPmNeg_1_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_1_2 <= rudPmNeg_1_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_1_3 <= rudPmNeg_1_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_2_0 <= rudPmNeg_2_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_2_1 <= rudPmNeg_2_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_2_2 <= rudPmNeg_2_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_2_3 <= rudPmNeg_2_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_3_0 <= rudPmNeg_3_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_3_1 <= rudPmNeg_3_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_3_2 <= rudPmNeg_3_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_3_3 <= rudPmNeg_3_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_4_0 <= rudPmNeg_4_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_4_1 <= rudPmNeg_4_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_4_2 <= rudPmNeg_4_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      rudPmNegReg_4_3 <= rudPmNeg_4_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_0_0 <= r2udPmNeg_0_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_0_1 <= r2udPmNeg_0_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_0_2 <= r2udPmNeg_0_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_0_3 <= r2udPmNeg_0_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_1_0 <= r2udPmNeg_1_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_1_1 <= r2udPmNeg_1_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_1_2 <= r2udPmNeg_1_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_1_3 <= r2udPmNeg_1_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_2_0 <= r2udPmNeg_2_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_2_1 <= r2udPmNeg_2_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_2_2 <= r2udPmNeg_2_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_2_3 <= r2udPmNeg_2_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_3_0 <= r2udPmNeg_3_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_3_1 <= r2udPmNeg_3_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_3_2 <= r2udPmNeg_3_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_3_3 <= r2udPmNeg_3_3; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_4_0 <= r2udPmNeg_4_0; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_4_1 <= r2udPmNeg_4_1; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_4_2 <= r2udPmNeg_4_2; // @[Reg.scala 17:22]
    end
    if (io_state[0]) begin // @[Reg.scala 17:18]
      r2udPmNegReg_4_3 <= r2udPmNeg_4_3; // @[Reg.scala 17:22]
    end
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  qPrevReg = _RAND_0[7:0];
  _RAND_1 = {1{`RANDOM}};
  wsReg = _RAND_1[27:0];
  _RAND_2 = {1{`RANDOM}};
  wcReg = _RAND_2[27:0];
  _RAND_3 = {1{`RANDOM}};
  quotIterReg = _RAND_3[25:0];
  _RAND_4 = {1{`RANDOM}};
  quotM1IterReg = _RAND_4[25:0];
  _RAND_5 = {1{`RANDOM}};
  udNegReg_0 = _RAND_5[27:0];
  _RAND_6 = {1{`RANDOM}};
  udNegReg_1 = _RAND_6[27:0];
  _RAND_7 = {1{`RANDOM}};
  udNegReg_3 = _RAND_7[27:0];
  _RAND_8 = {1{`RANDOM}};
  udNegReg_4 = _RAND_8[27:0];
  _RAND_9 = {1{`RANDOM}};
  rudPmNegReg_0_0 = _RAND_9[9:0];
  _RAND_10 = {1{`RANDOM}};
  rudPmNegReg_0_1 = _RAND_10[9:0];
  _RAND_11 = {1{`RANDOM}};
  rudPmNegReg_0_2 = _RAND_11[9:0];
  _RAND_12 = {1{`RANDOM}};
  rudPmNegReg_0_3 = _RAND_12[9:0];
  _RAND_13 = {1{`RANDOM}};
  rudPmNegReg_1_0 = _RAND_13[9:0];
  _RAND_14 = {1{`RANDOM}};
  rudPmNegReg_1_1 = _RAND_14[9:0];
  _RAND_15 = {1{`RANDOM}};
  rudPmNegReg_1_2 = _RAND_15[9:0];
  _RAND_16 = {1{`RANDOM}};
  rudPmNegReg_1_3 = _RAND_16[9:0];
  _RAND_17 = {1{`RANDOM}};
  rudPmNegReg_2_0 = _RAND_17[9:0];
  _RAND_18 = {1{`RANDOM}};
  rudPmNegReg_2_1 = _RAND_18[9:0];
  _RAND_19 = {1{`RANDOM}};
  rudPmNegReg_2_2 = _RAND_19[9:0];
  _RAND_20 = {1{`RANDOM}};
  rudPmNegReg_2_3 = _RAND_20[9:0];
  _RAND_21 = {1{`RANDOM}};
  rudPmNegReg_3_0 = _RAND_21[9:0];
  _RAND_22 = {1{`RANDOM}};
  rudPmNegReg_3_1 = _RAND_22[9:0];
  _RAND_23 = {1{`RANDOM}};
  rudPmNegReg_3_2 = _RAND_23[9:0];
  _RAND_24 = {1{`RANDOM}};
  rudPmNegReg_3_3 = _RAND_24[9:0];
  _RAND_25 = {1{`RANDOM}};
  rudPmNegReg_4_0 = _RAND_25[9:0];
  _RAND_26 = {1{`RANDOM}};
  rudPmNegReg_4_1 = _RAND_26[9:0];
  _RAND_27 = {1{`RANDOM}};
  rudPmNegReg_4_2 = _RAND_27[9:0];
  _RAND_28 = {1{`RANDOM}};
  rudPmNegReg_4_3 = _RAND_28[9:0];
  _RAND_29 = {1{`RANDOM}};
  r2udPmNegReg_0_0 = _RAND_29[12:0];
  _RAND_30 = {1{`RANDOM}};
  r2udPmNegReg_0_1 = _RAND_30[12:0];
  _RAND_31 = {1{`RANDOM}};
  r2udPmNegReg_0_2 = _RAND_31[12:0];
  _RAND_32 = {1{`RANDOM}};
  r2udPmNegReg_0_3 = _RAND_32[12:0];
  _RAND_33 = {1{`RANDOM}};
  r2udPmNegReg_1_0 = _RAND_33[12:0];
  _RAND_34 = {1{`RANDOM}};
  r2udPmNegReg_1_1 = _RAND_34[12:0];
  _RAND_35 = {1{`RANDOM}};
  r2udPmNegReg_1_2 = _RAND_35[12:0];
  _RAND_36 = {1{`RANDOM}};
  r2udPmNegReg_1_3 = _RAND_36[12:0];
  _RAND_37 = {1{`RANDOM}};
  r2udPmNegReg_2_0 = _RAND_37[12:0];
  _RAND_38 = {1{`RANDOM}};
  r2udPmNegReg_2_1 = _RAND_38[12:0];
  _RAND_39 = {1{`RANDOM}};
  r2udPmNegReg_2_2 = _RAND_39[12:0];
  _RAND_40 = {1{`RANDOM}};
  r2udPmNegReg_2_3 = _RAND_40[12:0];
  _RAND_41 = {1{`RANDOM}};
  r2udPmNegReg_3_0 = _RAND_41[12:0];
  _RAND_42 = {1{`RANDOM}};
  r2udPmNegReg_3_1 = _RAND_42[12:0];
  _RAND_43 = {1{`RANDOM}};
  r2udPmNegReg_3_2 = _RAND_43[12:0];
  _RAND_44 = {1{`RANDOM}};
  r2udPmNegReg_3_3 = _RAND_44[12:0];
  _RAND_45 = {1{`RANDOM}};
  r2udPmNegReg_4_0 = _RAND_45[12:0];
  _RAND_46 = {1{`RANDOM}};
  r2udPmNegReg_4_1 = _RAND_46[12:0];
  _RAND_47 = {1{`RANDOM}};
  r2udPmNegReg_4_2 = _RAND_47[12:0];
  _RAND_48 = {1{`RANDOM}};
  r2udPmNegReg_4_3 = _RAND_48[12:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module CSA3_2_31(
  input  [7:0] io_in_0,
  input  [7:0] io_in_1,
  input  [7:0] io_in_2,
  output [7:0] io_out_0,
  output [7:0] io_out_1
);
  wire  a = io_in_0[0]; // @[FDIV.scala 669:32]
  wire  b = io_in_1[0]; // @[FDIV.scala 669:45]
  wire  cin = io_in_2[0]; // @[FDIV.scala 669:58]
  wire  a_xor_b = a ^ b; // @[FDIV.scala 670:21]
  wire  a_and_b = a & b; // @[FDIV.scala 671:21]
  wire  sum = a_xor_b ^ cin; // @[FDIV.scala 672:23]
  wire  cout = a_and_b | a_xor_b & cin; // @[FDIV.scala 673:24]
  wire [1:0] temp_0 = {cout,sum}; // @[Cat.scala 31:58]
  wire  a_1 = io_in_0[1]; // @[FDIV.scala 669:32]
  wire  b_1 = io_in_1[1]; // @[FDIV.scala 669:45]
  wire  cin_1 = io_in_2[1]; // @[FDIV.scala 669:58]
  wire  a_xor_b_1 = a_1 ^ b_1; // @[FDIV.scala 670:21]
  wire  a_and_b_1 = a_1 & b_1; // @[FDIV.scala 671:21]
  wire  sum_1 = a_xor_b_1 ^ cin_1; // @[FDIV.scala 672:23]
  wire  cout_1 = a_and_b_1 | a_xor_b_1 & cin_1; // @[FDIV.scala 673:24]
  wire [1:0] temp_1 = {cout_1,sum_1}; // @[Cat.scala 31:58]
  wire  a_2 = io_in_0[2]; // @[FDIV.scala 669:32]
  wire  b_2 = io_in_1[2]; // @[FDIV.scala 669:45]
  wire  cin_2 = io_in_2[2]; // @[FDIV.scala 669:58]
  wire  a_xor_b_2 = a_2 ^ b_2; // @[FDIV.scala 670:21]
  wire  a_and_b_2 = a_2 & b_2; // @[FDIV.scala 671:21]
  wire  sum_2 = a_xor_b_2 ^ cin_2; // @[FDIV.scala 672:23]
  wire  cout_2 = a_and_b_2 | a_xor_b_2 & cin_2; // @[FDIV.scala 673:24]
  wire [1:0] temp_2 = {cout_2,sum_2}; // @[Cat.scala 31:58]
  wire  a_3 = io_in_0[3]; // @[FDIV.scala 669:32]
  wire  b_3 = io_in_1[3]; // @[FDIV.scala 669:45]
  wire  cin_3 = io_in_2[3]; // @[FDIV.scala 669:58]
  wire  a_xor_b_3 = a_3 ^ b_3; // @[FDIV.scala 670:21]
  wire  a_and_b_3 = a_3 & b_3; // @[FDIV.scala 671:21]
  wire  sum_3 = a_xor_b_3 ^ cin_3; // @[FDIV.scala 672:23]
  wire  cout_3 = a_and_b_3 | a_xor_b_3 & cin_3; // @[FDIV.scala 673:24]
  wire [1:0] temp_3 = {cout_3,sum_3}; // @[Cat.scala 31:58]
  wire  a_4 = io_in_0[4]; // @[FDIV.scala 669:32]
  wire  b_4 = io_in_1[4]; // @[FDIV.scala 669:45]
  wire  cin_4 = io_in_2[4]; // @[FDIV.scala 669:58]
  wire  a_xor_b_4 = a_4 ^ b_4; // @[FDIV.scala 670:21]
  wire  a_and_b_4 = a_4 & b_4; // @[FDIV.scala 671:21]
  wire  sum_4 = a_xor_b_4 ^ cin_4; // @[FDIV.scala 672:23]
  wire  cout_4 = a_and_b_4 | a_xor_b_4 & cin_4; // @[FDIV.scala 673:24]
  wire [1:0] temp_4 = {cout_4,sum_4}; // @[Cat.scala 31:58]
  wire  a_5 = io_in_0[5]; // @[FDIV.scala 669:32]
  wire  b_5 = io_in_1[5]; // @[FDIV.scala 669:45]
  wire  cin_5 = io_in_2[5]; // @[FDIV.scala 669:58]
  wire  a_xor_b_5 = a_5 ^ b_5; // @[FDIV.scala 670:21]
  wire  a_and_b_5 = a_5 & b_5; // @[FDIV.scala 671:21]
  wire  sum_5 = a_xor_b_5 ^ cin_5; // @[FDIV.scala 672:23]
  wire  cout_5 = a_and_b_5 | a_xor_b_5 & cin_5; // @[FDIV.scala 673:24]
  wire [1:0] temp_5 = {cout_5,sum_5}; // @[Cat.scala 31:58]
  wire  a_6 = io_in_0[6]; // @[FDIV.scala 669:32]
  wire  b_6 = io_in_1[6]; // @[FDIV.scala 669:45]
  wire  cin_6 = io_in_2[6]; // @[FDIV.scala 669:58]
  wire  a_xor_b_6 = a_6 ^ b_6; // @[FDIV.scala 670:21]
  wire  a_and_b_6 = a_6 & b_6; // @[FDIV.scala 671:21]
  wire  sum_6 = a_xor_b_6 ^ cin_6; // @[FDIV.scala 672:23]
  wire  cout_6 = a_and_b_6 | a_xor_b_6 & cin_6; // @[FDIV.scala 673:24]
  wire [1:0] temp_6 = {cout_6,sum_6}; // @[Cat.scala 31:58]
  wire  a_7 = io_in_0[7]; // @[FDIV.scala 669:32]
  wire  b_7 = io_in_1[7]; // @[FDIV.scala 669:45]
  wire  cin_7 = io_in_2[7]; // @[FDIV.scala 669:58]
  wire  a_xor_b_7 = a_7 ^ b_7; // @[FDIV.scala 670:21]
  wire  a_and_b_7 = a_7 & b_7; // @[FDIV.scala 671:21]
  wire  sum_7 = a_xor_b_7 ^ cin_7; // @[FDIV.scala 672:23]
  wire  cout_7 = a_and_b_7 | a_xor_b_7 & cin_7; // @[FDIV.scala 673:24]
  wire [1:0] temp_7 = {cout_7,sum_7}; // @[Cat.scala 31:58]
  wire [3:0] io_out_0_lo = {temp_3[0],temp_2[0],temp_1[0],temp_0[0]}; // @[Cat.scala 31:58]
  wire [3:0] io_out_0_hi = {temp_7[0],temp_6[0],temp_5[0],temp_4[0]}; // @[Cat.scala 31:58]
  wire [3:0] io_out_1_lo = {temp_3[1],temp_2[1],temp_1[1],temp_0[1]}; // @[Cat.scala 31:58]
  wire [3:0] io_out_1_hi = {temp_7[1],temp_6[1],temp_5[1],temp_4[1]}; // @[Cat.scala 31:58]
  assign io_out_0 = {io_out_0_hi,io_out_0_lo}; // @[Cat.scala 31:58]
  assign io_out_1 = {io_out_1_hi,io_out_1_lo}; // @[Cat.scala 31:58]
endmodule
module CSA3_2_35(
  input  [26:0] io_in_0,
  input  [26:0] io_in_1,
  input  [26:0] io_in_2,
  output [26:0] io_out_0,
  output [26:0] io_out_1
);
  wire  a = io_in_0[0]; // @[FDIV.scala 669:32]
  wire  b = io_in_1[0]; // @[FDIV.scala 669:45]
  wire  cin = io_in_2[0]; // @[FDIV.scala 669:58]
  wire  a_xor_b = a ^ b; // @[FDIV.scala 670:21]
  wire  a_and_b = a & b; // @[FDIV.scala 671:21]
  wire  sum = a_xor_b ^ cin; // @[FDIV.scala 672:23]
  wire  cout = a_and_b | a_xor_b & cin; // @[FDIV.scala 673:24]
  wire [1:0] temp_0 = {cout,sum}; // @[Cat.scala 31:58]
  wire  a_1 = io_in_0[1]; // @[FDIV.scala 669:32]
  wire  b_1 = io_in_1[1]; // @[FDIV.scala 669:45]
  wire  cin_1 = io_in_2[1]; // @[FDIV.scala 669:58]
  wire  a_xor_b_1 = a_1 ^ b_1; // @[FDIV.scala 670:21]
  wire  a_and_b_1 = a_1 & b_1; // @[FDIV.scala 671:21]
  wire  sum_1 = a_xor_b_1 ^ cin_1; // @[FDIV.scala 672:23]
  wire  cout_1 = a_and_b_1 | a_xor_b_1 & cin_1; // @[FDIV.scala 673:24]
  wire [1:0] temp_1 = {cout_1,sum_1}; // @[Cat.scala 31:58]
  wire  a_2 = io_in_0[2]; // @[FDIV.scala 669:32]
  wire  b_2 = io_in_1[2]; // @[FDIV.scala 669:45]
  wire  cin_2 = io_in_2[2]; // @[FDIV.scala 669:58]
  wire  a_xor_b_2 = a_2 ^ b_2; // @[FDIV.scala 670:21]
  wire  a_and_b_2 = a_2 & b_2; // @[FDIV.scala 671:21]
  wire  sum_2 = a_xor_b_2 ^ cin_2; // @[FDIV.scala 672:23]
  wire  cout_2 = a_and_b_2 | a_xor_b_2 & cin_2; // @[FDIV.scala 673:24]
  wire [1:0] temp_2 = {cout_2,sum_2}; // @[Cat.scala 31:58]
  wire  a_3 = io_in_0[3]; // @[FDIV.scala 669:32]
  wire  b_3 = io_in_1[3]; // @[FDIV.scala 669:45]
  wire  cin_3 = io_in_2[3]; // @[FDIV.scala 669:58]
  wire  a_xor_b_3 = a_3 ^ b_3; // @[FDIV.scala 670:21]
  wire  a_and_b_3 = a_3 & b_3; // @[FDIV.scala 671:21]
  wire  sum_3 = a_xor_b_3 ^ cin_3; // @[FDIV.scala 672:23]
  wire  cout_3 = a_and_b_3 | a_xor_b_3 & cin_3; // @[FDIV.scala 673:24]
  wire [1:0] temp_3 = {cout_3,sum_3}; // @[Cat.scala 31:58]
  wire  a_4 = io_in_0[4]; // @[FDIV.scala 669:32]
  wire  b_4 = io_in_1[4]; // @[FDIV.scala 669:45]
  wire  cin_4 = io_in_2[4]; // @[FDIV.scala 669:58]
  wire  a_xor_b_4 = a_4 ^ b_4; // @[FDIV.scala 670:21]
  wire  a_and_b_4 = a_4 & b_4; // @[FDIV.scala 671:21]
  wire  sum_4 = a_xor_b_4 ^ cin_4; // @[FDIV.scala 672:23]
  wire  cout_4 = a_and_b_4 | a_xor_b_4 & cin_4; // @[FDIV.scala 673:24]
  wire [1:0] temp_4 = {cout_4,sum_4}; // @[Cat.scala 31:58]
  wire  a_5 = io_in_0[5]; // @[FDIV.scala 669:32]
  wire  b_5 = io_in_1[5]; // @[FDIV.scala 669:45]
  wire  cin_5 = io_in_2[5]; // @[FDIV.scala 669:58]
  wire  a_xor_b_5 = a_5 ^ b_5; // @[FDIV.scala 670:21]
  wire  a_and_b_5 = a_5 & b_5; // @[FDIV.scala 671:21]
  wire  sum_5 = a_xor_b_5 ^ cin_5; // @[FDIV.scala 672:23]
  wire  cout_5 = a_and_b_5 | a_xor_b_5 & cin_5; // @[FDIV.scala 673:24]
  wire [1:0] temp_5 = {cout_5,sum_5}; // @[Cat.scala 31:58]
  wire  a_6 = io_in_0[6]; // @[FDIV.scala 669:32]
  wire  b_6 = io_in_1[6]; // @[FDIV.scala 669:45]
  wire  cin_6 = io_in_2[6]; // @[FDIV.scala 669:58]
  wire  a_xor_b_6 = a_6 ^ b_6; // @[FDIV.scala 670:21]
  wire  a_and_b_6 = a_6 & b_6; // @[FDIV.scala 671:21]
  wire  sum_6 = a_xor_b_6 ^ cin_6; // @[FDIV.scala 672:23]
  wire  cout_6 = a_and_b_6 | a_xor_b_6 & cin_6; // @[FDIV.scala 673:24]
  wire [1:0] temp_6 = {cout_6,sum_6}; // @[Cat.scala 31:58]
  wire  a_7 = io_in_0[7]; // @[FDIV.scala 669:32]
  wire  b_7 = io_in_1[7]; // @[FDIV.scala 669:45]
  wire  cin_7 = io_in_2[7]; // @[FDIV.scala 669:58]
  wire  a_xor_b_7 = a_7 ^ b_7; // @[FDIV.scala 670:21]
  wire  a_and_b_7 = a_7 & b_7; // @[FDIV.scala 671:21]
  wire  sum_7 = a_xor_b_7 ^ cin_7; // @[FDIV.scala 672:23]
  wire  cout_7 = a_and_b_7 | a_xor_b_7 & cin_7; // @[FDIV.scala 673:24]
  wire [1:0] temp_7 = {cout_7,sum_7}; // @[Cat.scala 31:58]
  wire  a_8 = io_in_0[8]; // @[FDIV.scala 669:32]
  wire  b_8 = io_in_1[8]; // @[FDIV.scala 669:45]
  wire  cin_8 = io_in_2[8]; // @[FDIV.scala 669:58]
  wire  a_xor_b_8 = a_8 ^ b_8; // @[FDIV.scala 670:21]
  wire  a_and_b_8 = a_8 & b_8; // @[FDIV.scala 671:21]
  wire  sum_8 = a_xor_b_8 ^ cin_8; // @[FDIV.scala 672:23]
  wire  cout_8 = a_and_b_8 | a_xor_b_8 & cin_8; // @[FDIV.scala 673:24]
  wire [1:0] temp_8 = {cout_8,sum_8}; // @[Cat.scala 31:58]
  wire  a_9 = io_in_0[9]; // @[FDIV.scala 669:32]
  wire  b_9 = io_in_1[9]; // @[FDIV.scala 669:45]
  wire  cin_9 = io_in_2[9]; // @[FDIV.scala 669:58]
  wire  a_xor_b_9 = a_9 ^ b_9; // @[FDIV.scala 670:21]
  wire  a_and_b_9 = a_9 & b_9; // @[FDIV.scala 671:21]
  wire  sum_9 = a_xor_b_9 ^ cin_9; // @[FDIV.scala 672:23]
  wire  cout_9 = a_and_b_9 | a_xor_b_9 & cin_9; // @[FDIV.scala 673:24]
  wire [1:0] temp_9 = {cout_9,sum_9}; // @[Cat.scala 31:58]
  wire  a_10 = io_in_0[10]; // @[FDIV.scala 669:32]
  wire  b_10 = io_in_1[10]; // @[FDIV.scala 669:45]
  wire  cin_10 = io_in_2[10]; // @[FDIV.scala 669:58]
  wire  a_xor_b_10 = a_10 ^ b_10; // @[FDIV.scala 670:21]
  wire  a_and_b_10 = a_10 & b_10; // @[FDIV.scala 671:21]
  wire  sum_10 = a_xor_b_10 ^ cin_10; // @[FDIV.scala 672:23]
  wire  cout_10 = a_and_b_10 | a_xor_b_10 & cin_10; // @[FDIV.scala 673:24]
  wire [1:0] temp_10 = {cout_10,sum_10}; // @[Cat.scala 31:58]
  wire  a_11 = io_in_0[11]; // @[FDIV.scala 669:32]
  wire  b_11 = io_in_1[11]; // @[FDIV.scala 669:45]
  wire  cin_11 = io_in_2[11]; // @[FDIV.scala 669:58]
  wire  a_xor_b_11 = a_11 ^ b_11; // @[FDIV.scala 670:21]
  wire  a_and_b_11 = a_11 & b_11; // @[FDIV.scala 671:21]
  wire  sum_11 = a_xor_b_11 ^ cin_11; // @[FDIV.scala 672:23]
  wire  cout_11 = a_and_b_11 | a_xor_b_11 & cin_11; // @[FDIV.scala 673:24]
  wire [1:0] temp_11 = {cout_11,sum_11}; // @[Cat.scala 31:58]
  wire  a_12 = io_in_0[12]; // @[FDIV.scala 669:32]
  wire  b_12 = io_in_1[12]; // @[FDIV.scala 669:45]
  wire  cin_12 = io_in_2[12]; // @[FDIV.scala 669:58]
  wire  a_xor_b_12 = a_12 ^ b_12; // @[FDIV.scala 670:21]
  wire  a_and_b_12 = a_12 & b_12; // @[FDIV.scala 671:21]
  wire  sum_12 = a_xor_b_12 ^ cin_12; // @[FDIV.scala 672:23]
  wire  cout_12 = a_and_b_12 | a_xor_b_12 & cin_12; // @[FDIV.scala 673:24]
  wire [1:0] temp_12 = {cout_12,sum_12}; // @[Cat.scala 31:58]
  wire  a_13 = io_in_0[13]; // @[FDIV.scala 669:32]
  wire  b_13 = io_in_1[13]; // @[FDIV.scala 669:45]
  wire  cin_13 = io_in_2[13]; // @[FDIV.scala 669:58]
  wire  a_xor_b_13 = a_13 ^ b_13; // @[FDIV.scala 670:21]
  wire  a_and_b_13 = a_13 & b_13; // @[FDIV.scala 671:21]
  wire  sum_13 = a_xor_b_13 ^ cin_13; // @[FDIV.scala 672:23]
  wire  cout_13 = a_and_b_13 | a_xor_b_13 & cin_13; // @[FDIV.scala 673:24]
  wire [1:0] temp_13 = {cout_13,sum_13}; // @[Cat.scala 31:58]
  wire  a_14 = io_in_0[14]; // @[FDIV.scala 669:32]
  wire  b_14 = io_in_1[14]; // @[FDIV.scala 669:45]
  wire  cin_14 = io_in_2[14]; // @[FDIV.scala 669:58]
  wire  a_xor_b_14 = a_14 ^ b_14; // @[FDIV.scala 670:21]
  wire  a_and_b_14 = a_14 & b_14; // @[FDIV.scala 671:21]
  wire  sum_14 = a_xor_b_14 ^ cin_14; // @[FDIV.scala 672:23]
  wire  cout_14 = a_and_b_14 | a_xor_b_14 & cin_14; // @[FDIV.scala 673:24]
  wire [1:0] temp_14 = {cout_14,sum_14}; // @[Cat.scala 31:58]
  wire  a_15 = io_in_0[15]; // @[FDIV.scala 669:32]
  wire  b_15 = io_in_1[15]; // @[FDIV.scala 669:45]
  wire  cin_15 = io_in_2[15]; // @[FDIV.scala 669:58]
  wire  a_xor_b_15 = a_15 ^ b_15; // @[FDIV.scala 670:21]
  wire  a_and_b_15 = a_15 & b_15; // @[FDIV.scala 671:21]
  wire  sum_15 = a_xor_b_15 ^ cin_15; // @[FDIV.scala 672:23]
  wire  cout_15 = a_and_b_15 | a_xor_b_15 & cin_15; // @[FDIV.scala 673:24]
  wire [1:0] temp_15 = {cout_15,sum_15}; // @[Cat.scala 31:58]
  wire  a_16 = io_in_0[16]; // @[FDIV.scala 669:32]
  wire  b_16 = io_in_1[16]; // @[FDIV.scala 669:45]
  wire  cin_16 = io_in_2[16]; // @[FDIV.scala 669:58]
  wire  a_xor_b_16 = a_16 ^ b_16; // @[FDIV.scala 670:21]
  wire  a_and_b_16 = a_16 & b_16; // @[FDIV.scala 671:21]
  wire  sum_16 = a_xor_b_16 ^ cin_16; // @[FDIV.scala 672:23]
  wire  cout_16 = a_and_b_16 | a_xor_b_16 & cin_16; // @[FDIV.scala 673:24]
  wire [1:0] temp_16 = {cout_16,sum_16}; // @[Cat.scala 31:58]
  wire  a_17 = io_in_0[17]; // @[FDIV.scala 669:32]
  wire  b_17 = io_in_1[17]; // @[FDIV.scala 669:45]
  wire  cin_17 = io_in_2[17]; // @[FDIV.scala 669:58]
  wire  a_xor_b_17 = a_17 ^ b_17; // @[FDIV.scala 670:21]
  wire  a_and_b_17 = a_17 & b_17; // @[FDIV.scala 671:21]
  wire  sum_17 = a_xor_b_17 ^ cin_17; // @[FDIV.scala 672:23]
  wire  cout_17 = a_and_b_17 | a_xor_b_17 & cin_17; // @[FDIV.scala 673:24]
  wire [1:0] temp_17 = {cout_17,sum_17}; // @[Cat.scala 31:58]
  wire  a_18 = io_in_0[18]; // @[FDIV.scala 669:32]
  wire  b_18 = io_in_1[18]; // @[FDIV.scala 669:45]
  wire  cin_18 = io_in_2[18]; // @[FDIV.scala 669:58]
  wire  a_xor_b_18 = a_18 ^ b_18; // @[FDIV.scala 670:21]
  wire  a_and_b_18 = a_18 & b_18; // @[FDIV.scala 671:21]
  wire  sum_18 = a_xor_b_18 ^ cin_18; // @[FDIV.scala 672:23]
  wire  cout_18 = a_and_b_18 | a_xor_b_18 & cin_18; // @[FDIV.scala 673:24]
  wire [1:0] temp_18 = {cout_18,sum_18}; // @[Cat.scala 31:58]
  wire  a_19 = io_in_0[19]; // @[FDIV.scala 669:32]
  wire  b_19 = io_in_1[19]; // @[FDIV.scala 669:45]
  wire  cin_19 = io_in_2[19]; // @[FDIV.scala 669:58]
  wire  a_xor_b_19 = a_19 ^ b_19; // @[FDIV.scala 670:21]
  wire  a_and_b_19 = a_19 & b_19; // @[FDIV.scala 671:21]
  wire  sum_19 = a_xor_b_19 ^ cin_19; // @[FDIV.scala 672:23]
  wire  cout_19 = a_and_b_19 | a_xor_b_19 & cin_19; // @[FDIV.scala 673:24]
  wire [1:0] temp_19 = {cout_19,sum_19}; // @[Cat.scala 31:58]
  wire  a_20 = io_in_0[20]; // @[FDIV.scala 669:32]
  wire  b_20 = io_in_1[20]; // @[FDIV.scala 669:45]
  wire  cin_20 = io_in_2[20]; // @[FDIV.scala 669:58]
  wire  a_xor_b_20 = a_20 ^ b_20; // @[FDIV.scala 670:21]
  wire  a_and_b_20 = a_20 & b_20; // @[FDIV.scala 671:21]
  wire  sum_20 = a_xor_b_20 ^ cin_20; // @[FDIV.scala 672:23]
  wire  cout_20 = a_and_b_20 | a_xor_b_20 & cin_20; // @[FDIV.scala 673:24]
  wire [1:0] temp_20 = {cout_20,sum_20}; // @[Cat.scala 31:58]
  wire  a_21 = io_in_0[21]; // @[FDIV.scala 669:32]
  wire  b_21 = io_in_1[21]; // @[FDIV.scala 669:45]
  wire  cin_21 = io_in_2[21]; // @[FDIV.scala 669:58]
  wire  a_xor_b_21 = a_21 ^ b_21; // @[FDIV.scala 670:21]
  wire  a_and_b_21 = a_21 & b_21; // @[FDIV.scala 671:21]
  wire  sum_21 = a_xor_b_21 ^ cin_21; // @[FDIV.scala 672:23]
  wire  cout_21 = a_and_b_21 | a_xor_b_21 & cin_21; // @[FDIV.scala 673:24]
  wire [1:0] temp_21 = {cout_21,sum_21}; // @[Cat.scala 31:58]
  wire  a_22 = io_in_0[22]; // @[FDIV.scala 669:32]
  wire  b_22 = io_in_1[22]; // @[FDIV.scala 669:45]
  wire  cin_22 = io_in_2[22]; // @[FDIV.scala 669:58]
  wire  a_xor_b_22 = a_22 ^ b_22; // @[FDIV.scala 670:21]
  wire  a_and_b_22 = a_22 & b_22; // @[FDIV.scala 671:21]
  wire  sum_22 = a_xor_b_22 ^ cin_22; // @[FDIV.scala 672:23]
  wire  cout_22 = a_and_b_22 | a_xor_b_22 & cin_22; // @[FDIV.scala 673:24]
  wire [1:0] temp_22 = {cout_22,sum_22}; // @[Cat.scala 31:58]
  wire  a_23 = io_in_0[23]; // @[FDIV.scala 669:32]
  wire  b_23 = io_in_1[23]; // @[FDIV.scala 669:45]
  wire  cin_23 = io_in_2[23]; // @[FDIV.scala 669:58]
  wire  a_xor_b_23 = a_23 ^ b_23; // @[FDIV.scala 670:21]
  wire  a_and_b_23 = a_23 & b_23; // @[FDIV.scala 671:21]
  wire  sum_23 = a_xor_b_23 ^ cin_23; // @[FDIV.scala 672:23]
  wire  cout_23 = a_and_b_23 | a_xor_b_23 & cin_23; // @[FDIV.scala 673:24]
  wire [1:0] temp_23 = {cout_23,sum_23}; // @[Cat.scala 31:58]
  wire  a_24 = io_in_0[24]; // @[FDIV.scala 669:32]
  wire  b_24 = io_in_1[24]; // @[FDIV.scala 669:45]
  wire  cin_24 = io_in_2[24]; // @[FDIV.scala 669:58]
  wire  a_xor_b_24 = a_24 ^ b_24; // @[FDIV.scala 670:21]
  wire  a_and_b_24 = a_24 & b_24; // @[FDIV.scala 671:21]
  wire  sum_24 = a_xor_b_24 ^ cin_24; // @[FDIV.scala 672:23]
  wire  cout_24 = a_and_b_24 | a_xor_b_24 & cin_24; // @[FDIV.scala 673:24]
  wire [1:0] temp_24 = {cout_24,sum_24}; // @[Cat.scala 31:58]
  wire  a_25 = io_in_0[25]; // @[FDIV.scala 669:32]
  wire  b_25 = io_in_1[25]; // @[FDIV.scala 669:45]
  wire  cin_25 = io_in_2[25]; // @[FDIV.scala 669:58]
  wire  a_xor_b_25 = a_25 ^ b_25; // @[FDIV.scala 670:21]
  wire  a_and_b_25 = a_25 & b_25; // @[FDIV.scala 671:21]
  wire  sum_25 = a_xor_b_25 ^ cin_25; // @[FDIV.scala 672:23]
  wire  cout_25 = a_and_b_25 | a_xor_b_25 & cin_25; // @[FDIV.scala 673:24]
  wire [1:0] temp_25 = {cout_25,sum_25}; // @[Cat.scala 31:58]
  wire  a_26 = io_in_0[26]; // @[FDIV.scala 669:32]
  wire  b_26 = io_in_1[26]; // @[FDIV.scala 669:45]
  wire  cin_26 = io_in_2[26]; // @[FDIV.scala 669:58]
  wire  a_xor_b_26 = a_26 ^ b_26; // @[FDIV.scala 670:21]
  wire  a_and_b_26 = a_26 & b_26; // @[FDIV.scala 671:21]
  wire  sum_26 = a_xor_b_26 ^ cin_26; // @[FDIV.scala 672:23]
  wire  cout_26 = a_and_b_26 | a_xor_b_26 & cin_26; // @[FDIV.scala 673:24]
  wire [1:0] temp_26 = {cout_26,sum_26}; // @[Cat.scala 31:58]
  wire [5:0] io_out_0_lo_lo = {temp_5[0],temp_4[0],temp_3[0],temp_2[0],temp_1[0],temp_0[0]}; // @[Cat.scala 31:58]
  wire [12:0] io_out_0_lo = {temp_12[0],temp_11[0],temp_10[0],temp_9[0],temp_8[0],temp_7[0],temp_6[0],io_out_0_lo_lo}; // @[Cat.scala 31:58]
  wire [6:0] io_out_0_hi_lo = {temp_19[0],temp_18[0],temp_17[0],temp_16[0],temp_15[0],temp_14[0],temp_13[0]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_0_hi = {temp_26[0],temp_25[0],temp_24[0],temp_23[0],temp_22[0],temp_21[0],temp_20[0],io_out_0_hi_lo
    }; // @[Cat.scala 31:58]
  wire [5:0] io_out_1_lo_lo = {temp_5[1],temp_4[1],temp_3[1],temp_2[1],temp_1[1],temp_0[1]}; // @[Cat.scala 31:58]
  wire [12:0] io_out_1_lo = {temp_12[1],temp_11[1],temp_10[1],temp_9[1],temp_8[1],temp_7[1],temp_6[1],io_out_1_lo_lo}; // @[Cat.scala 31:58]
  wire [6:0] io_out_1_hi_lo = {temp_19[1],temp_18[1],temp_17[1],temp_16[1],temp_15[1],temp_14[1],temp_13[1]}; // @[Cat.scala 31:58]
  wire [13:0] io_out_1_hi = {temp_26[1],temp_25[1],temp_24[1],temp_23[1],temp_22[1],temp_21[1],temp_20[1],io_out_1_hi_lo
    }; // @[Cat.scala 31:58]
  assign io_out_0 = {io_out_0_hi,io_out_0_lo}; // @[Cat.scala 31:58]
  assign io_out_1 = {io_out_1_hi,io_out_1_lo}; // @[Cat.scala 31:58]
endmodule
module SqrtIterModule(
  input         clock,
  input  [24:0] io_a,
  input  [1:0]  io_state,
  output [26:0] io_rem,
  output [25:0] io_res,
  output [25:0] io_resM1
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
`endif // RANDOMIZE_REG_INIT
  wire [7:0] signs_csa_sel_0_io_in_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_0_io_in_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_0_io_in_2; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_0_io_out_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_0_io_out_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_1_io_in_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_1_io_in_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_1_io_in_2; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_1_io_out_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_1_io_out_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_2_io_in_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_2_io_in_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_2_io_in_2; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_2_io_out_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_2_io_out_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_3_io_in_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_3_io_in_1; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_3_io_in_2; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_3_io_out_0; // @[FDIV.scala 568:21]
  wire [7:0] signs_csa_sel_3_io_out_1; // @[FDIV.scala 568:21]
  wire [26:0] csa_iter_io_in_0; // @[FDIV.scala 598:23]
  wire [26:0] csa_iter_io_in_1; // @[FDIV.scala 598:23]
  wire [26:0] csa_iter_io_in_2; // @[FDIV.scala 598:23]
  wire [26:0] csa_iter_io_out_0; // @[FDIV.scala 598:23]
  wire [26:0] csa_iter_io_out_1; // @[FDIV.scala 598:23]
  wire [26:0] wsInit = {2'h3,io_a}; // @[Cat.scala 31:58]
  wire [7:0] _signs_T_16 = {signs_csa_sel_3_io_out_1[6:0], 1'h0}; // @[FDIV.scala 587:43]
  wire [7:0] _signs_T_18 = signs_csa_sel_3_io_out_0 + _signs_T_16; // @[FDIV.scala 587:20]
  wire  signs_3 = _signs_T_18[7]; // @[FDIV.scala 587:49]
  wire [7:0] _signs_T_11 = {signs_csa_sel_2_io_out_1[6:0], 1'h0}; // @[FDIV.scala 587:43]
  wire [7:0] _signs_T_13 = signs_csa_sel_2_io_out_0 + _signs_T_11; // @[FDIV.scala 587:20]
  wire  signs_2 = _signs_T_13[7]; // @[FDIV.scala 587:49]
  wire [7:0] _signs_T_6 = {signs_csa_sel_1_io_out_1[6:0], 1'h0}; // @[FDIV.scala 587:43]
  wire [7:0] _signs_T_8 = signs_csa_sel_1_io_out_0 + _signs_T_6; // @[FDIV.scala 587:20]
  wire  signs_1 = _signs_T_8[7]; // @[FDIV.scala 587:49]
  wire [7:0] _signs_T_1 = {signs_csa_sel_0_io_out_1[6:0], 1'h0}; // @[FDIV.scala 587:43]
  wire [7:0] _signs_T_3 = signs_csa_sel_0_io_out_0 + _signs_T_1; // @[FDIV.scala 587:20]
  wire  signs_0 = _signs_T_3[7]; // @[FDIV.scala 587:49]
  wire [3:0] _s_T = {signs_3,signs_2,signs_1,signs_0}; // @[FDIV.scala 589:25]
  wire  _s_sel_q_4_T_3 = ~_s_T[2]; // @[FDIV.scala 634:32]
  wire  _s_sel_q_4_T_6 = ~_s_T[1]; // @[FDIV.scala 634:47]
  wire  s_sel_q_4 = ~_s_T[3] & ~_s_T[2] & ~_s_T[1]; // @[FDIV.scala 634:44]
  wire  s_sel_q_3 = _s_T[3] & _s_sel_q_4_T_3 & _s_sel_q_4_T_6; // @[FDIV.scala 633:42]
  wire  s_sel_q_2 = _s_T[2] & _s_sel_q_4_T_6; // @[FDIV.scala 632:27]
  wire  s_sel_q_1 = ~_s_T[0] & _s_T[1] & _s_T[2]; // @[FDIV.scala 631:42]
  wire  s_sel_q_0 = _s_T[0] & _s_T[1] & _s_T[2]; // @[FDIV.scala 630:40]
  wire [4:0] s = {s_sel_q_4,s_sel_q_3,s_sel_q_2,s_sel_q_1,s_sel_q_0}; // @[FDIV.scala 635:10]
  reg [25:0] aReg; // @[Reg.scala 16:16]
  wire [27:0] _aIter_quotNext_T_1 = {aReg, 2'h0}; // @[FDIV.scala 642:21]
  wire [27:0] _aIter_quotNext_T_2 = _aIter_quotNext_T_1 | 28'h2; // @[FDIV.scala 642:26]
  wire [27:0] _aIter_quotNext_T_15 = s[4] ? _aIter_quotNext_T_2 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_5 = _aIter_quotNext_T_1 | 28'h1; // @[FDIV.scala 643:26]
  wire [27:0] _aIter_quotNext_T_16 = s[3] ? _aIter_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_20 = _aIter_quotNext_T_15 | _aIter_quotNext_T_16; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_17 = s[2] ? _aIter_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_21 = _aIter_quotNext_T_20 | _aIter_quotNext_T_17; // @[Mux.scala 27:73]
  reg [25:0] bReg; // @[Reg.scala 16:16]
  wire [27:0] _aIter_quotNext_T_10 = {bReg, 2'h0}; // @[FDIV.scala 645:23]
  wire [27:0] _aIter_quotNext_T_11 = _aIter_quotNext_T_10 | 28'h3; // @[FDIV.scala 645:28]
  wire [27:0] _aIter_quotNext_T_18 = s[1] ? _aIter_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_22 = _aIter_quotNext_T_21 | _aIter_quotNext_T_18; // @[Mux.scala 27:73]
  wire [27:0] _aIter_quotNext_T_14 = _aIter_quotNext_T_10 | 28'h2; // @[FDIV.scala 646:28]
  wire [27:0] _aIter_quotNext_T_19 = s[0] ? _aIter_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] aIter_quotNext = _aIter_quotNext_T_22 | _aIter_quotNext_T_19; // @[Mux.scala 27:73]
  wire [25:0] aIter = aIter_quotNext[25:0]; // @[FDIV.scala 655:14]
  wire  _aReg_T_4 = io_state[0] | io_state[1]; // @[FDIV.scala 531:74]
  wire [27:0] _bIter_quotM1Next_T_15 = s[4] ? _aIter_quotNext_T_5 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_16 = s[3] ? _aIter_quotNext_T_1 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_20 = _bIter_quotM1Next_T_15 | _bIter_quotM1Next_T_16; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_17 = s[2] ? _aIter_quotNext_T_11 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_21 = _bIter_quotM1Next_T_20 | _bIter_quotM1Next_T_17; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_18 = s[1] ? _aIter_quotNext_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_22 = _bIter_quotM1Next_T_21 | _bIter_quotM1Next_T_18; // @[Mux.scala 27:73]
  wire [27:0] _bIter_quotM1Next_T_14 = _aIter_quotNext_T_10 | 28'h1; // @[FDIV.scala 653:28]
  wire [27:0] _bIter_quotM1Next_T_19 = s[0] ? _bIter_quotM1Next_T_14 : 28'h0; // @[Mux.scala 27:73]
  wire [27:0] bIter_quotM1Next = _bIter_quotM1Next_T_22 | _bIter_quotM1Next_T_19; // @[Mux.scala 27:73]
  wire [25:0] bIter = bIter_quotM1Next[25:0]; // @[FDIV.scala 655:38]
  wire [26:0] wsIter = csa_iter_io_out_0; // @[FDIV.scala 602:30]
  reg [26:0] wsReg; // @[Reg.scala 16:16]
  wire [26:0] wcIter = {csa_iter_io_out_1[25:0], 1'h0}; // @[FDIV.scala 603:41]
  reg [26:0] wcReg; // @[Reg.scala 16:16]
  reg [31:0] jReg; // @[Reg.scala 16:16]
  wire [31:0] _jReg_T_2 = jReg + 32'h1; // @[FDIV.scala 536:50]
  wire [31:0] _lookup_T_1 = jReg - 32'h1; // @[FDIV.scala 540:47]
  wire [32:0] _lookup_T_2 = {_lookup_T_1, 1'h0}; // @[FDIV.scala 540:54]
  wire [25:0] _lookup_T_3 = aReg >> _lookup_T_2; // @[FDIV.scala 540:40]
  wire [31:0] _lookup_T_6 = jReg - 32'h3; // @[FDIV.scala 540:91]
  wire [32:0] _lookup_T_7 = {_lookup_T_6, 1'h0}; // @[FDIV.scala 540:98]
  wire [25:0] _lookup_T_8 = aReg >> _lookup_T_7; // @[FDIV.scala 540:81]
  wire [25:0] _lookup_T_9 = _lookup_T_3[0] ? 26'h7 : _lookup_T_8; // @[FDIV.scala 540:35]
  wire [2:0] _lookup_T_13 = {aReg[0],2'h0}; // @[Cat.scala 31:58]
  wire [2:0] _lookup_T_14 = ~aReg[2] ? _lookup_T_13 : 3'h7; // @[FDIV.scala 542:17]
  wire [2:0] _lookup_T_18 = ~aReg[4] ? aReg[2:0] : 3'h7; // @[FDIV.scala 543:17]
  wire [2:0] _lookup_T_26 = ~_lookup_T_3[0] ? aReg[4:2] : 3'h7; // @[FDIV.scala 544:17]
  wire [25:0] _lookup_T_28 = 32'h1 == jReg ? 26'h5 : _lookup_T_9; // @[Mux.scala 81:58]
  wire [25:0] _lookup_T_30 = 32'h2 == jReg ? {{23'd0}, _lookup_T_14} : _lookup_T_28; // @[Mux.scala 81:58]
  wire [25:0] _lookup_T_32 = 32'h3 == jReg ? {{23'd0}, _lookup_T_18} : _lookup_T_30; // @[Mux.scala 81:58]
  wire [25:0] lookup = 32'h4 == jReg ? {{23'd0}, _lookup_T_26} : _lookup_T_32; // @[Mux.scala 81:58]
  wire [2:0] mNeg_plaInput = lookup[2:0]; // @[decoder.scala 38:16 pla.scala 77:22]
  wire [2:0] mNeg_invInputs = ~mNeg_plaInput; // @[pla.scala 78:21]
  wire  mNeg_andMatrixInput_0 = mNeg_invInputs[1]; // @[pla.scala 91:29]
  wire  mNeg_andMatrixInput_1 = mNeg_invInputs[2]; // @[pla.scala 91:29]
  wire [1:0] _mNeg_T = {mNeg_andMatrixInput_0,mNeg_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _mNeg_T_1 = &_mNeg_T; // @[pla.scala 98:74]
  wire  mNeg_andMatrixInput_0_1 = mNeg_invInputs[0]; // @[pla.scala 91:29]
  wire [1:0] mNeg_hi = {mNeg_andMatrixInput_0_1,mNeg_andMatrixInput_0}; // @[Cat.scala 31:58]
  wire [2:0] _mNeg_T_2 = {mNeg_andMatrixInput_0_1,mNeg_andMatrixInput_0,mNeg_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _mNeg_T_3 = &_mNeg_T_2; // @[pla.scala 98:74]
  wire  mNeg_andMatrixInput_0_2 = mNeg_plaInput[0]; // @[pla.scala 90:45]
  wire [1:0] _mNeg_T_4 = {mNeg_andMatrixInput_0_2,mNeg_andMatrixInput_0}; // @[Cat.scala 31:58]
  wire  _mNeg_T_5 = &_mNeg_T_4; // @[pla.scala 98:74]
  wire [2:0] _mNeg_T_6 = {mNeg_andMatrixInput_0_2,mNeg_andMatrixInput_0,mNeg_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _mNeg_T_7 = &_mNeg_T_6; // @[pla.scala 98:74]
  wire  mNeg_andMatrixInput_0_4 = mNeg_plaInput[1]; // @[pla.scala 90:45]
  wire  _mNeg_T_8 = &mNeg_andMatrixInput_0_4; // @[pla.scala 98:74]
  wire [1:0] _mNeg_T_9 = {mNeg_andMatrixInput_0_2,mNeg_andMatrixInput_0_4}; // @[Cat.scala 31:58]
  wire  _mNeg_T_10 = &_mNeg_T_9; // @[pla.scala 98:74]
  wire  mNeg_andMatrixInput_0_6 = mNeg_plaInput[2]; // @[pla.scala 90:45]
  wire  _mNeg_T_11 = &mNeg_andMatrixInput_0_6; // @[pla.scala 98:74]
  wire [1:0] _mNeg_T_12 = {mNeg_andMatrixInput_0_1,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_13 = &_mNeg_T_12; // @[pla.scala 98:74]
  wire [1:0] _mNeg_T_14 = {mNeg_andMatrixInput_0_4,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_15 = &_mNeg_T_14; // @[pla.scala 98:74]
  wire [1:0] _mNeg_orMatrixOutputs_T = {_mNeg_T_3,_mNeg_T_10}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_1 = |_mNeg_orMatrixOutputs_T; // @[pla.scala 114:39]
  wire [2:0] _mNeg_orMatrixOutputs_T_2 = {_mNeg_T_7,_mNeg_T_13,_mNeg_T_15}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_3 = |_mNeg_orMatrixOutputs_T_2; // @[pla.scala 114:39]
  wire [2:0] _mNeg_orMatrixOutputs_T_4 = {_mNeg_T_1,_mNeg_T_5,_mNeg_T_15}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_5 = |_mNeg_orMatrixOutputs_T_4; // @[pla.scala 114:39]
  wire  _mNeg_orMatrixOutputs_T_6 = |_mNeg_T_1; // @[pla.scala 114:39]
  wire [1:0] _mNeg_orMatrixOutputs_T_7 = {_mNeg_T_8,_mNeg_T_11}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_8 = |_mNeg_orMatrixOutputs_T_7; // @[pla.scala 114:39]
  wire [5:0] mNeg_orMatrixOutputs = {1'h0,_mNeg_orMatrixOutputs_T_8,_mNeg_orMatrixOutputs_T_6,_mNeg_orMatrixOutputs_T_5,
    _mNeg_orMatrixOutputs_T_3,_mNeg_orMatrixOutputs_T_1}; // @[Cat.scala 31:58]
  wire [5:0] mNeg_invMatrixOutputs = {mNeg_orMatrixOutputs[5],mNeg_orMatrixOutputs[4],mNeg_orMatrixOutputs[3],
    mNeg_orMatrixOutputs[2],mNeg_orMatrixOutputs[1],mNeg_orMatrixOutputs[0]}; // @[Cat.scala 31:58]
  wire  mNeg_signBit = mNeg_invMatrixOutputs[5]; // @[FDIV.scala 132:20]
  wire [6:0] _mNeg_T_16 = {mNeg_signBit,mNeg_orMatrixOutputs[5],mNeg_orMatrixOutputs[4],mNeg_orMatrixOutputs[3],
    mNeg_orMatrixOutputs[2],mNeg_orMatrixOutputs[1],mNeg_orMatrixOutputs[0]}; // @[Cat.scala 31:58]
  wire  _mNeg_T_19 = &mNeg_hi; // @[pla.scala 98:74]
  wire  _mNeg_T_20 = &mNeg_andMatrixInput_1; // @[pla.scala 98:74]
  wire [1:0] _mNeg_T_23 = {mNeg_andMatrixInput_0_4,mNeg_andMatrixInput_1}; // @[Cat.scala 31:58]
  wire  _mNeg_T_24 = &_mNeg_T_23; // @[pla.scala 98:74]
  wire [2:0] _mNeg_T_25 = {mNeg_andMatrixInput_0_1,mNeg_andMatrixInput_0,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_26 = &_mNeg_T_25; // @[pla.scala 98:74]
  wire [1:0] _mNeg_T_27 = {mNeg_andMatrixInput_0_2,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_28 = &_mNeg_T_27; // @[pla.scala 98:74]
  wire  _mNeg_orMatrixOutputs_T_9 = |_mNeg_T_7; // @[pla.scala 114:39]
  wire [1:0] _mNeg_orMatrixOutputs_T_10 = {_mNeg_T_24,_mNeg_T_26}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_11 = |_mNeg_orMatrixOutputs_T_10; // @[pla.scala 114:39]
  wire [1:0] _mNeg_orMatrixOutputs_T_12 = {_mNeg_T_19,_mNeg_T_20}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_13 = |_mNeg_orMatrixOutputs_T_12; // @[pla.scala 114:39]
  wire [1:0] _mNeg_orMatrixOutputs_T_14 = {_mNeg_T_28,_mNeg_T_15}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_15 = |_mNeg_orMatrixOutputs_T_14; // @[pla.scala 114:39]
  wire [5:0] mNeg_orMatrixOutputs_1 = {2'h0,_mNeg_orMatrixOutputs_T_15,_mNeg_orMatrixOutputs_T_13,
    _mNeg_orMatrixOutputs_T_11,_mNeg_orMatrixOutputs_T_9}; // @[Cat.scala 31:58]
  wire [5:0] mNeg_invMatrixOutputs_1 = {mNeg_orMatrixOutputs_1[5],mNeg_orMatrixOutputs_1[4],mNeg_orMatrixOutputs_1[3],
    mNeg_orMatrixOutputs_1[2],mNeg_orMatrixOutputs_1[1],mNeg_orMatrixOutputs_1[0]}; // @[Cat.scala 31:58]
  wire  mNeg_signBit_1 = mNeg_invMatrixOutputs_1[5]; // @[FDIV.scala 132:20]
  wire [6:0] _mNeg_T_31 = {mNeg_signBit_1,mNeg_orMatrixOutputs_1[5],mNeg_orMatrixOutputs_1[4],mNeg_orMatrixOutputs_1[3],
    mNeg_orMatrixOutputs_1[2],mNeg_orMatrixOutputs_1[1],mNeg_orMatrixOutputs_1[0]}; // @[Cat.scala 31:58]
  wire [1:0] _mNeg_T_34 = {mNeg_andMatrixInput_0,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_35 = &_mNeg_T_34; // @[pla.scala 98:74]
  wire  _mNeg_orMatrixOutputs_T_16 = |_mNeg_T_35; // @[pla.scala 114:39]
  wire  _mNeg_orMatrixOutputs_T_17 = |_mNeg_T_20; // @[pla.scala 114:39]
  wire [5:0] mNeg_orMatrixOutputs_2 = {3'h7,_mNeg_orMatrixOutputs_T_17,_mNeg_orMatrixOutputs_T_16,1'h0}; // @[Cat.scala 31:58]
  wire [5:0] mNeg_invMatrixOutputs_2 = {mNeg_orMatrixOutputs_2[5],mNeg_orMatrixOutputs_2[4],mNeg_orMatrixOutputs_2[3],
    mNeg_orMatrixOutputs_2[2],mNeg_orMatrixOutputs_2[1],mNeg_orMatrixOutputs_2[0]}; // @[Cat.scala 31:58]
  wire  mNeg_signBit_2 = mNeg_invMatrixOutputs_2[5]; // @[FDIV.scala 132:20]
  wire [6:0] _mNeg_T_36 = {mNeg_signBit_2,mNeg_orMatrixOutputs_2[5],mNeg_orMatrixOutputs_2[4],mNeg_orMatrixOutputs_2[3],
    mNeg_orMatrixOutputs_2[2],mNeg_orMatrixOutputs_2[1],mNeg_orMatrixOutputs_2[0]}; // @[Cat.scala 31:58]
  wire [2:0] _mNeg_T_50 = {mNeg_andMatrixInput_0_2,mNeg_andMatrixInput_0_4,mNeg_andMatrixInput_0_6}; // @[Cat.scala 31:58]
  wire  _mNeg_T_51 = &_mNeg_T_50; // @[pla.scala 98:74]
  wire [2:0] _mNeg_orMatrixOutputs_T_21 = {_mNeg_T_7,_mNeg_T_26,_mNeg_T_51}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_22 = |_mNeg_orMatrixOutputs_T_21; // @[pla.scala 114:39]
  wire [2:0] _mNeg_orMatrixOutputs_T_23 = {_mNeg_T_19,_mNeg_T_13,_mNeg_T_35}; // @[Cat.scala 31:58]
  wire  _mNeg_orMatrixOutputs_T_24 = |_mNeg_orMatrixOutputs_T_23; // @[pla.scala 114:39]
  wire  _mNeg_orMatrixOutputs_T_25 = |_mNeg_T_11; // @[pla.scala 114:39]
  wire [5:0] mNeg_orMatrixOutputs_3 = {1'h1,_mNeg_orMatrixOutputs_T_17,_mNeg_orMatrixOutputs_T_25,
    _mNeg_orMatrixOutputs_T_24,_mNeg_orMatrixOutputs_T_22,1'h0}; // @[Cat.scala 31:58]
  wire [5:0] mNeg_invMatrixOutputs_3 = {mNeg_orMatrixOutputs_3[5],mNeg_orMatrixOutputs_3[4],mNeg_orMatrixOutputs_3[3],
    mNeg_orMatrixOutputs_3[2],mNeg_orMatrixOutputs_3[1],mNeg_orMatrixOutputs_3[0]}; // @[Cat.scala 31:58]
  wire  mNeg_signBit_3 = mNeg_invMatrixOutputs_3[5]; // @[FDIV.scala 132:20]
  wire [6:0] _mNeg_T_52 = {mNeg_signBit_3,mNeg_orMatrixOutputs_3[5],mNeg_orMatrixOutputs_3[4],mNeg_orMatrixOutputs_3[3],
    mNeg_orMatrixOutputs_3[2],mNeg_orMatrixOutputs_3[1],mNeg_orMatrixOutputs_3[0]}; // @[Cat.scala 31:58]
  wire [29:0] _f_T = {bReg, 4'h0}; // @[FDIV.scala 591:11]
  wire [29:0] _f_T_1 = _f_T | 30'hc; // @[FDIV.scala 591:16]
  wire [28:0] _f_T_2 = {bReg, 3'h0}; // @[FDIV.scala 592:11]
  wire [28:0] _f_T_3 = _f_T_2 | 29'h7; // @[FDIV.scala 592:16]
  wire [25:0] _f_T_4 = ~aReg; // @[FDIV.scala 594:6]
  wire [28:0] _f_T_5 = {_f_T_4, 3'h0}; // @[FDIV.scala 594:12]
  wire [28:0] _f_T_6 = _f_T_5 | 29'h7; // @[FDIV.scala 594:17]
  wire [29:0] _f_T_8 = {_f_T_4, 4'h0}; // @[FDIV.scala 595:12]
  wire [29:0] _f_T_9 = _f_T_8 | 30'hc; // @[FDIV.scala 595:17]
  wire [29:0] _f_T_15 = s[0] ? _f_T_1 : 30'h0; // @[Mux.scala 27:73]
  wire [28:0] _f_T_16 = s[1] ? _f_T_3 : 29'h0; // @[Mux.scala 27:73]
  wire [28:0] _f_T_18 = s[3] ? _f_T_6 : 29'h0; // @[Mux.scala 27:73]
  wire [29:0] _f_T_19 = s[4] ? _f_T_9 : 30'h0; // @[Mux.scala 27:73]
  wire [29:0] _GEN_5 = {{1'd0}, _f_T_16}; // @[Mux.scala 27:73]
  wire [29:0] _f_T_20 = _f_T_15 | _GEN_5; // @[Mux.scala 27:73]
  wire [29:0] _GEN_6 = {{1'd0}, _f_T_18}; // @[Mux.scala 27:73]
  wire [29:0] _f_T_22 = _f_T_20 | _GEN_6; // @[Mux.scala 27:73]
  wire [29:0] _f_T_23 = _f_T_22 | _f_T_19; // @[Mux.scala 27:73]
  wire [28:0] _csa_iter_io_in_0_T = {wsReg, 2'h0}; // @[FDIV.scala 599:29]
  wire [28:0] _csa_iter_io_in_1_T = {wcReg, 2'h0}; // @[FDIV.scala 600:29]
  wire [28:0] f = _f_T_23[28:0]; // @[FDIV.scala 528:15 590:5]
  wire [53:0] _GEN_7 = {f, 25'h0}; // @[FDIV.scala 601:26]
  wire [59:0] _csa_iter_io_in_2_T = {{6'd0}, _GEN_7}; // @[FDIV.scala 601:26]
  wire [32:0] _csa_iter_io_in_2_T_1 = {jReg, 1'h0}; // @[FDIV.scala 601:46]
  wire [59:0] _csa_iter_io_in_2_T_2 = _csa_iter_io_in_2_T >> _csa_iter_io_in_2_T_1; // @[FDIV.scala 601:40]
  CSA3_2_31 signs_csa_sel_0 ( // @[FDIV.scala 568:21]
    .io_in_0(signs_csa_sel_0_io_in_0),
    .io_in_1(signs_csa_sel_0_io_in_1),
    .io_in_2(signs_csa_sel_0_io_in_2),
    .io_out_0(signs_csa_sel_0_io_out_0),
    .io_out_1(signs_csa_sel_0_io_out_1)
  );
  CSA3_2_31 signs_csa_sel_1 ( // @[FDIV.scala 568:21]
    .io_in_0(signs_csa_sel_1_io_in_0),
    .io_in_1(signs_csa_sel_1_io_in_1),
    .io_in_2(signs_csa_sel_1_io_in_2),
    .io_out_0(signs_csa_sel_1_io_out_0),
    .io_out_1(signs_csa_sel_1_io_out_1)
  );
  CSA3_2_31 signs_csa_sel_2 ( // @[FDIV.scala 568:21]
    .io_in_0(signs_csa_sel_2_io_in_0),
    .io_in_1(signs_csa_sel_2_io_in_1),
    .io_in_2(signs_csa_sel_2_io_in_2),
    .io_out_0(signs_csa_sel_2_io_out_0),
    .io_out_1(signs_csa_sel_2_io_out_1)
  );
  CSA3_2_31 signs_csa_sel_3 ( // @[FDIV.scala 568:21]
    .io_in_0(signs_csa_sel_3_io_in_0),
    .io_in_1(signs_csa_sel_3_io_in_1),
    .io_in_2(signs_csa_sel_3_io_in_2),
    .io_out_0(signs_csa_sel_3_io_out_0),
    .io_out_1(signs_csa_sel_3_io_out_1)
  );
  CSA3_2_35 csa_iter ( // @[FDIV.scala 598:23]
    .io_in_0(csa_iter_io_in_0),
    .io_in_1(csa_iter_io_in_1),
    .io_in_2(csa_iter_io_in_2),
    .io_out_0(csa_iter_io_out_0),
    .io_out_1(csa_iter_io_out_1)
  );
  assign io_rem = wsReg + wcReg; // @[FDIV.scala 622:19]
  assign io_res = aReg; // @[FDIV.scala 623:10]
  assign io_resM1 = bReg; // @[FDIV.scala 624:12]
  assign signs_csa_sel_0_io_in_0 = wsReg[26:19]; // @[FDIV.scala 569:31]
  assign signs_csa_sel_0_io_in_1 = wcReg[26:19]; // @[FDIV.scala 570:31]
  assign signs_csa_sel_0_io_in_2 = {_mNeg_T_16,1'h0}; // @[Cat.scala 31:58]
  assign signs_csa_sel_1_io_in_0 = wsReg[26:19]; // @[FDIV.scala 569:31]
  assign signs_csa_sel_1_io_in_1 = wcReg[26:19]; // @[FDIV.scala 570:31]
  assign signs_csa_sel_1_io_in_2 = {_mNeg_T_31,1'h0}; // @[Cat.scala 31:58]
  assign signs_csa_sel_2_io_in_0 = wsReg[26:19]; // @[FDIV.scala 569:31]
  assign signs_csa_sel_2_io_in_1 = wcReg[26:19]; // @[FDIV.scala 570:31]
  assign signs_csa_sel_2_io_in_2 = {_mNeg_T_36,1'h0}; // @[Cat.scala 31:58]
  assign signs_csa_sel_3_io_in_0 = wsReg[26:19]; // @[FDIV.scala 569:31]
  assign signs_csa_sel_3_io_in_1 = wcReg[26:19]; // @[FDIV.scala 570:31]
  assign signs_csa_sel_3_io_in_2 = {_mNeg_T_52,1'h0}; // @[Cat.scala 31:58]
  assign csa_iter_io_in_0 = _csa_iter_io_in_0_T[26:0]; // @[FDIV.scala 599:20]
  assign csa_iter_io_in_1 = _csa_iter_io_in_1_T[26:0]; // @[FDIV.scala 600:20]
  assign csa_iter_io_in_2 = _csa_iter_io_in_2_T_2[26:0]; // @[FDIV.scala 601:20]
  always @(posedge clock) begin
    if (_aReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 531:27]
        aReg <= 26'h1;
      end else begin
        aReg <= aIter;
      end
    end
    if (_aReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 532:27]
        bReg <= 26'h0;
      end else begin
        bReg <= bIter;
      end
    end
    if (_aReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 533:28]
        wsReg <= wsInit;
      end else begin
        wsReg <= wsIter;
      end
    end
    if (_aReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 534:28]
        wcReg <= 27'h0;
      end else begin
        wcReg <= wcIter;
      end
    end
    if (_aReg_T_4) begin // @[Reg.scala 17:18]
      if (io_state[0]) begin // @[FDIV.scala 536:27]
        jReg <= 32'h1;
      end else begin
        jReg <= _jReg_T_2;
      end
    end
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  aReg = _RAND_0[25:0];
  _RAND_1 = {1{`RANDOM}};
  bReg = _RAND_1[25:0];
  _RAND_2 = {1{`RANDOM}};
  wsReg = _RAND_2[26:0];
  _RAND_3 = {1{`RANDOM}};
  wcReg = _RAND_3[26:0];
  _RAND_4 = {1{`RANDOM}};
  jReg = _RAND_4[31:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module RoundingUnitD(
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
module FDIV(
  input         clock,
  input         reset,
  input  [31:0] io_a,
  input  [31:0] io_b,
  input  [2:0]  io_rm,
  output [31:0] io_result,
  output [4:0]  io_fflags,
  input         io_specialIO_in_valid,
  input         io_specialIO_out_ready,
  output        io_specialIO_in_ready,
  output        io_specialIO_out_valid,
  input         io_specialIO_isSqrt,
  input         io_specialIO_kill
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
  reg [31:0] _RAND_5;
  reg [31:0] _RAND_6;
  reg [31:0] _RAND_7;
  reg [31:0] _RAND_8;
  reg [31:0] _RAND_9;
  reg [31:0] _RAND_10;
  reg [31:0] _RAND_11;
  reg [31:0] _RAND_12;
  reg [31:0] _RAND_13;
  reg [31:0] _RAND_14;
  reg [31:0] _RAND_15;
  reg [31:0] _RAND_16;
  reg [31:0] _RAND_17;
  reg [31:0] _RAND_18;
  reg [31:0] _RAND_19;
  reg [31:0] _RAND_20;
  reg [31:0] _RAND_21;
  reg [31:0] _RAND_22;
  reg [31:0] _RAND_23;
  reg [31:0] _RAND_24;
  reg [31:0] _RAND_25;
  reg [31:0] _RAND_26;
  reg [31:0] _RAND_27;
`endif // RANDOMIZE_REG_INIT
  wire [23:0] aLZC_clz_io_in; // @[CLZ.scala 22:21]
  wire [4:0] aLZC_clz_io_out; // @[CLZ.scala 22:21]
  wire [23:0] dLZC_clz_io_in; // @[CLZ.scala 22:21]
  wire [4:0] dLZC_clz_io_out; // @[CLZ.scala 22:21]
  wire  divModule_clock; // @[FDIV.scala 294:25]
  wire [23:0] divModule_io_a; // @[FDIV.scala 294:25]
  wire [23:0] divModule_io_d; // @[FDIV.scala 294:25]
  wire [1:0] divModule_io_state; // @[FDIV.scala 294:25]
  wire  divModule_io_lastIterDoHalf; // @[FDIV.scala 294:25]
  wire  divModule_io_sigCmp; // @[FDIV.scala 294:25]
  wire [27:0] divModule_io_rem; // @[FDIV.scala 294:25]
  wire [25:0] divModule_io_quot; // @[FDIV.scala 294:25]
  wire [25:0] divModule_io_quotM1; // @[FDIV.scala 294:25]
  wire  sqrtModule_clock; // @[FDIV.scala 301:26]
  wire [24:0] sqrtModule_io_a; // @[FDIV.scala 301:26]
  wire [1:0] sqrtModule_io_state; // @[FDIV.scala 301:26]
  wire [26:0] sqrtModule_io_rem; // @[FDIV.scala 301:26]
  wire [25:0] sqrtModule_io_res; // @[FDIV.scala 301:26]
  wire [25:0] sqrtModule_io_resM1; // @[FDIV.scala 301:26]
  wire [22:0] rounder_io_in; // @[FDIV.scala 314:23]
  wire  rounder_io_roundIn; // @[FDIV.scala 314:23]
  wire  rounder_io_stickyIn; // @[FDIV.scala 314:23]
  wire  rounder_io_signIn; // @[FDIV.scala 314:23]
  wire [2:0] rounder_io_rm; // @[FDIV.scala 314:23]
  wire [22:0] rounder_io_out; // @[FDIV.scala 314:23]
  wire  rounder_io_inexact; // @[FDIV.scala 314:23]
  wire  rounder_io_cout; // @[FDIV.scala 314:23]
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
  wire  decode_a_isSubnormal = decode_a_expIsZero & sigNotZero; // @[package.scala 41:46]
  wire  decode_a_isInf = expIsOnes & decode_a_sigIsZero; // @[package.scala 42:40]
  wire  decode_a_isZero = decode_a_expIsZero & decode_a_sigIsZero; // @[package.scala 43:41]
  wire  decode_a_isNaN = expIsOnes & sigNotZero; // @[package.scala 44:40]
  wire  decode_a_isSNaN = decode_a_isNaN & ~fp_a_sig[22]; // @[package.scala 45:37]
  wire  decode_a_isQNaN = decode_a_isNaN & fp_a_sig[22]; // @[package.scala 46:37]
  wire  expNotZero_1 = |fp_b_exp; // @[package.scala 32:31]
  wire  expIsOnes_1 = &fp_b_exp; // @[package.scala 33:31]
  wire  sigNotZero_1 = |fp_b_sig; // @[package.scala 34:31]
  wire  decode_b_expIsZero = ~expNotZero_1; // @[package.scala 37:27]
  wire  decode_b_sigIsZero = ~sigNotZero_1; // @[package.scala 40:27]
  wire  decode_b_isSubnormal = decode_b_expIsZero & sigNotZero_1; // @[package.scala 41:46]
  wire  decode_b_isInf = expIsOnes_1 & decode_b_sigIsZero; // @[package.scala 42:40]
  wire  decode_b_isZero = decode_b_expIsZero & decode_b_sigIsZero; // @[package.scala 43:41]
  wire  decode_b_isNaN = expIsOnes_1 & sigNotZero_1; // @[package.scala 44:40]
  wire  decode_b_isSNaN = decode_b_isNaN & ~fp_b_sig[22]; // @[package.scala 45:37]
  wire [7:0] _GEN_36 = {{7'd0}, decode_a_expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_a_exp = fp_a_exp | _GEN_36; // @[package.scala 83:27]
  wire [23:0] raw_a_sig = {expNotZero,fp_a_sig}; // @[Cat.scala 31:58]
  wire [7:0] _GEN_37 = {{7'd0}, decode_b_expIsZero}; // @[package.scala 83:27]
  wire [7:0] raw_b_exp = fp_b_exp | _GEN_37; // @[package.scala 83:27]
  wire [23:0] raw_b_sig = {expNotZero_1,fp_b_sig}; // @[Cat.scala 31:58]
  reg [5:0] state; // @[FDIV.scala 173:22]
  wire  in_fire = io_specialIO_in_ready & io_specialIO_in_valid; // @[FDIV.scala 175:26]
  wire [5:0] _io_specialIO_out_valid_T = {{5'd0}, state[5]}; // @[FDIV.scala 177:21]
  reg  aSubReg; // @[Reg.scala 16:16]
  reg  dSubReg; // @[Reg.scala 16:16]
  wire  _hasSubnormal_T = ~io_specialIO_isSqrt; // @[FDIV.scala 198:31]
  wire  hasSubnormal = decode_a_isSubnormal | ~io_specialIO_isSqrt & decode_b_isSubnormal; // @[FDIV.scala 198:27]
  reg  sqrtReg; // @[Reg.scala 16:16]
  reg [2:0] rmReg; // @[Reg.scala 16:16]
  reg  resSignReg; // @[Reg.scala 16:16]
  wire [5:0] _T_3 = {{1'd0}, state[5:1]}; // @[FDIV.scala 208:20]
  wire [5:0] _T_5 = {{2'd0}, state[5:2]}; // @[FDIV.scala 210:20]
  reg [8:0] aExp; // @[Reg.scala 16:16]
  wire [8:0] _sqrtFinalExp_T_1 = aExp - 9'h7f; // @[FDIV.scala 257:36]
  wire  sqrtFinalExp_signBit = _sqrtFinalExp_T_1[8]; // @[FDIV.scala 132:20]
  wire [1:0] _sqrtFinalExp_T_4 = sqrtFinalExp_signBit ? 2'h3 : 2'h0; // @[Bitwise.scala 74:12]
  wire [9:0] sqrtFinalExp = {_sqrtFinalExp_T_4,_sqrtFinalExp_T_1[8:1]}; // @[Cat.scala 31:58]
  wire  divFinalExp_signBit = aExp[8]; // @[FDIV.scala 132:20]
  wire [9:0] _divFinalExp_T = {divFinalExp_signBit,aExp}; // @[Cat.scala 31:58]
  reg [8:0] dExp; // @[Reg.scala 16:16]
  wire  divFinalExp_signBit_1 = dExp[8]; // @[FDIV.scala 132:20]
  wire [9:0] _divFinalExp_T_1 = {divFinalExp_signBit_1,dExp}; // @[Cat.scala 31:58]
  wire [9:0] _divFinalExp_T_2 = ~_divFinalExp_T_1; // @[FDIV.scala 255:50]
  wire [9:0] _divFinalExp_T_4 = _divFinalExp_T + _divFinalExp_T_2; // @[FDIV.scala 255:48]
  reg [23:0] aSigReg; // @[Reg.scala 16:16]
  reg [23:0] dSigReg; // @[Reg.scala 16:16]
  wire  sigCmp = aSigReg < dSigReg; // @[FDIV.scala 253:24]
  wire  _divFinalExp_T_5 = ~sigCmp; // @[FDIV.scala 255:81]
  wire [9:0] _GEN_41 = {{9'd0}, _divFinalExp_T_5}; // @[FDIV.scala 255:78]
  wire [9:0] divFinalExp = _divFinalExp_T_4 + _GEN_41; // @[FDIV.scala 255:78]
  wire [9:0] finalExp = sqrtReg ? sqrtFinalExp : divFinalExp; // @[FDIV.scala 258:21]
  wire [9:0] _infRes_T = ~finalExp; // @[FDIV.scala 261:17]
  wire [9:0] _infRes_T_2 = _infRes_T + 10'h80; // @[FDIV.scala 261:27]
  wire  infRes = _infRes_T_2[9]; // @[FDIV.scala 261:105]
  reg  inv; // @[Reg.scala 16:16]
  wire  _overflow_T = ~inv; // @[FDIV.scala 266:28]
  reg  dz; // @[Reg.scala 16:16]
  wire  _overflow_T_2 = ~dz; // @[FDIV.scala 266:36]
  reg  inf_div; // @[Reg.scala 16:16]
  wire  _overflow_T_4 = ~inf_div; // @[FDIV.scala 266:43]
  wire  overflow = infRes & ~inv & ~dz & ~inf_div; // @[FDIV.scala 266:40]
  wire [9:0] _zeroRes_T_1 = finalExp + 10'h97; // @[FDIV.scala 260:27]
  wire  zeroRes = _zeroRes_T_1[9]; // @[FDIV.scala 260:77]
  reg  zero_div; // @[Reg.scala 16:16]
  wire  _underflow_pre_T_4 = ~zero_div; // @[FDIV.scala 267:49]
  reg  div_inf; // @[Reg.scala 16:16]
  wire  _underflow_pre_T_6 = ~div_inf; // @[FDIV.scala 267:62]
  wire  underflow_pre = zeroRes & _overflow_T & _overflow_T_2 & ~zero_div & ~div_inf; // @[FDIV.scala 267:59]
  wire  skipIter = overflow | underflow_pre | inv | dz | zeroRes | zero_div | div_inf | inf_div; // @[FDIV.scala 276:88]
  wire [5:0] _state_T_15 = skipIter ? 6'h10 : 6'h8; // @[FDIV.scala 211:17]
  wire [5:0] _T_7 = {{3'd0}, state[5:3]}; // @[FDIV.scala 212:33]
  reg [3:0] iterNumReg; // @[Reg.scala 16:16]
  wire  finalIter = iterNumReg == 4'h0; // @[FDIV.scala 291:27]
  wire [5:0] _T_10 = {{4'd0}, state[5:4]}; // @[FDIV.scala 214:20]
  wire [5:0] _GEN_5 = _io_specialIO_out_valid_T[0] & io_specialIO_out_ready ? 6'h1 : state; // @[FDIV.scala 216:45 217:11 219:11]
  wire [5:0] _GEN_6 = _T_10[0] ? 6'h20 : _GEN_5; // @[FDIV.scala 214:32 215:11]
  wire [5:0] _GEN_7 = finalIter & _T_7[0] ? 6'h10 : _GEN_6; // @[FDIV.scala 212:43 213:11]
  wire [5:0] _GEN_8 = _T_5[0] ? _state_T_15 : _GEN_7; // @[FDIV.scala 210:31 211:11]
  wire [54:0] _GEN_0 = {{31'd0}, aSigReg}; // @[FDIV.scala 233:23]
  wire [54:0] _aSigNorm_T = _GEN_0 << aLZC_clz_io_out; // @[FDIV.scala 233:23]
  wire [23:0] aSigNorm = _aSigNorm_T[23:0]; // @[FDIV.scala 222:22 233:12]
  wire  _aSigReg_T_7 = state[0] | _T_3[0]; // @[FDIV.scala 223:82]
  wire [54:0] _GEN_1 = {{31'd0}, dSigReg}; // @[FDIV.scala 235:23]
  wire [54:0] _dSigNorm_T = _GEN_1 << dLZC_clz_io_out; // @[FDIV.scala 235:23]
  wire [23:0] dSigNorm = _dSigNorm_T[23:0]; // @[FDIV.scala 224:22 235:12]
  wire [8:0] _GEN_47 = {{4'd0}, aLZC_clz_io_out}; // @[FDIV.scala 236:19]
  wire [8:0] aExpFix = aExp - _GEN_47; // @[FDIV.scala 236:19]
  wire  _aExp_T_8 = state[0] | _T_3[0] & aSubReg; // @[FDIV.scala 229:78]
  wire [8:0] _GEN_49 = {{4'd0}, dLZC_clz_io_out}; // @[FDIV.scala 237:19]
  wire [8:0] dExpFix = dExp - _GEN_49; // @[FDIV.scala 237:19]
  wire  _dExp_T_8 = state[0] | _T_3[0] & dSubReg; // @[FDIV.scala 231:78]
  wire  _inv_T = ~decode_a_isZero; // @[FDIV.scala 243:49]
  wire  _inv_T_5 = decode_a_isInf & decode_b_isInf | decode_b_isZero & decode_a_isZero; // @[FDIV.scala 243:121]
  reg  inv_flag; // @[Reg.scala 16:16]
  wire  _dz_T_3 = decode_b_isZero & _inv_T & _hasSubnormal_T; // @[FDIV.scala 245:58]
  wire  _zero_div_T_2 = decode_a_isZero & (~decode_b_isZero | io_specialIO_isSqrt); // @[FDIV.scala 246:44]
  wire  _div_inf_T_3 = _hasSubnormal_T & decode_b_isInf & ~decode_a_isInf; // @[FDIV.scala 247:53]
  wire  sqrtShift = ~aExp[0]; // @[FDIV.scala 256:19]
  wire [9:0] _subRes_T_1 = finalExp + 10'h7e; // @[FDIV.scala 262:26]
  wire  subRes = _subRes_T_1[9] & ~zeroRes; // @[FDIV.scala 262:79]
  reg  subResReg; // @[Reg.scala 16:16]
  wire  inexact_pre = _underflow_pre_T_4 & _overflow_T & _overflow_T_2 & _underflow_pre_T_6 & _overflow_T_4; // @[FDIV.scala 268:58]
  wire  _special_fflags_T_3 = dz & _overflow_T & _overflow_T_4; // @[FDIV.scala 270:59]
  wire [4:0] _special_fflags_T_4 = {inv_flag,_special_fflags_T_3,overflow,underflow_pre,inexact_pre}; // @[Cat.scala 31:58]
  reg [4:0] special_fflags; // @[Reg.scala 16:16]
  reg [7:0] special_exp; // @[Reg.scala 16:16]
  reg [22:0] special_sig; // @[Reg.scala 16:16]
  reg  skipIterReg; // @[Reg.scala 16:16]
  wire [9:0] resultSigBits = subRes ? _zeroRes_T_1 : 10'h19; // @[FDIV.scala 282:26]
  reg  needShiftReg; // @[Reg.scala 16:16]
  wire  _oddIterReg_T_1 = ~resultSigBits[1]; // @[FDIV.scala 286:30]
  reg  oddIterReg; // @[Reg.scala 16:16]
  wire [9:0] _iterNumInit_T_1 = resultSigBits - 10'h1; // @[FDIV.scala 288:49]
  wire [8:0] iterNumInit = sqrtReg ? _iterNumInit_T_1[9:1] : {{1'd0}, resultSigBits[9:2]}; // @[FDIV.scala 288:24]
  wire  _iterNumReg_T_4 = _T_5[0] | _T_7[0]; // @[FDIV.scala 289:54]
  wire [3:0] _iterNum_T_3 = iterNumReg - 4'h1; // @[FDIV.scala 290:58]
  wire [8:0] _iterNum_T_4 = _T_5[0] ? iterNumInit : {{5'd0}, _iterNum_T_3}; // @[FDIV.scala 290:17]
  wire [3:0] iterNum = _iterNum_T_4[3:0]; // @[FDIV.scala 287:21 290:11]
  wire [24:0] _sqrtModule_io_a_T_1 = {1'h0,aSigReg}; // @[Cat.scala 31:58]
  wire [24:0] _sqrtModule_io_a_T_2 = {aSigReg,1'h0}; // @[Cat.scala 31:58]
  wire [25:0] quotIter = sqrtReg ? sqrtModule_io_res : divModule_io_quot; // @[FDIV.scala 306:18]
  wire [25:0] quotM1Iter = sqrtReg ? sqrtModule_io_resM1 : divModule_io_quotM1; // @[FDIV.scala 307:20]
  wire  r_signBit = sqrtModule_io_rem[26]; // @[FDIV.scala 132:20]
  wire [1:0] _r_T_1 = r_signBit ? 2'h3 : 2'h0; // @[Bitwise.scala 74:12]
  wire [28:0] _r_T_2 = {_r_T_1,sqrtModule_io_rem}; // @[Cat.scala 31:58]
  wire  r_signBit_1 = divModule_io_rem[27]; // @[FDIV.scala 132:20]
  wire [28:0] _r_T_3 = {r_signBit_1,divModule_io_rem}; // @[Cat.scala 31:58]
  wire [28:0] r = sqrtReg ? _r_T_2 : _r_T_3; // @[FDIV.scala 310:14]
  wire [25:0] qFinal = r[28] ? quotM1Iter : quotIter; // @[FDIV.scala 311:19]
  wire [9:0] _resExp_T_1 = finalExp + 10'h7f; // @[FDIV.scala 321:45]
  wire [9:0] _resExp_T_2 = subResReg ? 10'h0 : _resExp_T_1; // @[FDIV.scala 321:19]
  wire [7:0] resExp = _resExp_T_2[7:0]; // @[FDIV.scala 321:78]
  wire  uf = ~(|resExp) & rounder_io_inexact; // @[FDIV.scala 323:24]
  wire  of = &resExp; // @[FDIV.scala 324:19]
  wire [4:0] normal_fflags = {2'h0,of,uf,rounder_io_inexact}; // @[Cat.scala 31:58]
  wire  _noInf_T_1 = rmReg == 3'h2; // @[FDIV.scala 329:40]
  wire  _noInf_T_2 = ~resSignReg; // @[FDIV.scala 329:51]
  wire  _noInf_T_5 = rmReg == 3'h3; // @[FDIV.scala 329:74]
  wire  noInf = (rmReg == 3'h1 | rmReg == 3'h2 & ~resSignReg | rmReg == 3'h3 & resSignReg) & special_fflags[2]; // @[FDIV.scala 329:98]
  wire  noZero = (_noInf_T_1 & resSignReg | _noInf_T_5 & _noInf_T_2) & special_fflags[1]; // @[FDIV.scala 330:82]
  wire  _combinedExp_T = noInf | noZero; // @[FDIV.scala 334:31]
  wire [7:0] _combinedExp_T_4 = resExp + 8'h1; // @[FDIV.scala 336:85]
  wire [22:0] _combinedSig_T_3 = rounder_io_in; // @[FDIV.scala 339:91]
  wire [22:0] _combinedSig_T_4 = rounder_io_out; // @[FDIV.scala 339:125]
  reg [7:0] combinedExpReg; // @[Reg.scala 16:16]
  reg [22:0] combinedSigReg; // @[Reg.scala 16:16]
  reg [4:0] combinedFFlagsReg; // @[Reg.scala 16:16]
  reg  combinedSignReg; // @[Reg.scala 16:16]
  wire [8:0] io_result_hi = {combinedSignReg,combinedExpReg}; // @[Cat.scala 31:58]
  CLZD aLZC_clz ( // @[CLZ.scala 22:21]
    .io_in(aLZC_clz_io_in),
    .io_out(aLZC_clz_io_out)
  );
  CLZD dLZC_clz ( // @[CLZ.scala 22:21]
    .io_in(dLZC_clz_io_in),
    .io_out(dLZC_clz_io_out)
  );
  DivIterModule divModule ( // @[FDIV.scala 294:25]
    .clock(divModule_clock),
    .io_a(divModule_io_a),
    .io_d(divModule_io_d),
    .io_state(divModule_io_state),
    .io_lastIterDoHalf(divModule_io_lastIterDoHalf),
    .io_sigCmp(divModule_io_sigCmp),
    .io_rem(divModule_io_rem),
    .io_quot(divModule_io_quot),
    .io_quotM1(divModule_io_quotM1)
  );
  SqrtIterModule sqrtModule ( // @[FDIV.scala 301:26]
    .clock(sqrtModule_clock),
    .io_a(sqrtModule_io_a),
    .io_state(sqrtModule_io_state),
    .io_rem(sqrtModule_io_rem),
    .io_res(sqrtModule_io_res),
    .io_resM1(sqrtModule_io_resM1)
  );
  RoundingUnitD rounder ( // @[FDIV.scala 314:23]
    .io_in(rounder_io_in),
    .io_roundIn(rounder_io_roundIn),
    .io_stickyIn(rounder_io_stickyIn),
    .io_signIn(rounder_io_signIn),
    .io_rm(rounder_io_rm),
    .io_out(rounder_io_out),
    .io_inexact(rounder_io_inexact),
    .io_cout(rounder_io_cout)
  );
  assign io_result = {io_result_hi,combinedSigReg}; // @[Cat.scala 31:58]
  assign io_fflags = combinedFFlagsReg; // @[FDIV.scala 349:13]
  assign io_specialIO_in_ready = state[0]; // @[FDIV.scala 176:20]
  assign io_specialIO_out_valid = _io_specialIO_out_valid_T[0]; // @[FDIV.scala 177:21]
  assign aLZC_clz_io_in = aSigReg; // @[CLZ.scala 23:15]
  assign dLZC_clz_io_in = dSigReg; // @[CLZ.scala 23:15]
  assign divModule_clock = clock;
  assign divModule_io_a = aSigReg; // @[FDIV.scala 295:18]
  assign divModule_io_d = dSigReg; // @[FDIV.scala 296:18]
  assign divModule_io_state = {_T_7[0],_T_5[0]}; // @[Cat.scala 31:58]
  assign divModule_io_lastIterDoHalf = oddIterReg & finalIter; // @[FDIV.scala 298:45]
  assign divModule_io_sigCmp = aSigReg < dSigReg; // @[FDIV.scala 253:24]
  assign sqrtModule_clock = clock;
  assign sqrtModule_io_a = ~sqrtShift ? _sqrtModule_io_a_T_1 : _sqrtModule_io_a_T_2; // @[FDIV.scala 302:25]
  assign sqrtModule_io_state = {_T_7[0],_T_5[0]}; // @[Cat.scala 31:58]
  assign rounder_io_in = needShiftReg ? qFinal[24:2] : qFinal[23:1]; // @[FDIV.scala 315:23]
  assign rounder_io_roundIn = needShiftReg ? qFinal[1] : qFinal[0]; // @[FDIV.scala 313:18]
  assign rounder_io_stickyIn = |r | needShiftReg & qFinal[0]; // @[FDIV.scala 312:26]
  assign rounder_io_signIn = resSignReg; // @[FDIV.scala 318:21]
  assign rounder_io_rm = rmReg; // @[FDIV.scala 319:17]
  always @(posedge clock) begin
    if (reset) begin // @[FDIV.scala 173:22]
      state <= 6'h1; // @[FDIV.scala 173:22]
    end else if (io_specialIO_kill) begin // @[FDIV.scala 204:27]
      state <= 6'h1; // @[FDIV.scala 205:11]
    end else if (state[0] & in_fire) begin // @[FDIV.scala 206:41]
      if (hasSubnormal) begin // @[FDIV.scala 207:17]
        state <= 6'h2;
      end else begin
        state <= 6'h4;
      end
    end else if (_T_3[0]) begin // @[FDIV.scala 208:31]
      state <= 6'h4; // @[FDIV.scala 209:11]
    end else begin
      state <= _GEN_8;
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      aSubReg <= decode_a_isSubnormal; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      dSubReg <= decode_b_isSubnormal; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      sqrtReg <= io_specialIO_isSqrt; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      rmReg <= io_rm; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      if (io_specialIO_isSqrt) begin // @[FDIV.scala 201:33]
        resSignReg <= decode_a_isZero & fp_a_sign;
      end else begin
        resSignReg <= fp_a_sign ^ fp_b_sign;
      end
    end
    if (_aExp_T_8) begin // @[Reg.scala 17:18]
      if (state[0]) begin // @[FDIV.scala 229:27]
        aExp <= {{1'd0}, raw_a_exp};
      end else begin
        aExp <= aExpFix;
      end
    end
    if (_dExp_T_8) begin // @[Reg.scala 17:18]
      if (state[0]) begin // @[FDIV.scala 231:27]
        dExp <= {{1'd0}, raw_b_exp};
      end else begin
        dExp <= dExpFix;
      end
    end
    if (_aSigReg_T_7) begin // @[Reg.scala 17:18]
      if (state[0]) begin // @[FDIV.scala 223:30]
        aSigReg <= raw_a_sig;
      end else begin
        aSigReg <= aSigNorm;
      end
    end
    if (_aSigReg_T_7) begin // @[Reg.scala 17:18]
      if (state[0]) begin // @[FDIV.scala 225:30]
        dSigReg <= raw_b_sig;
      end else begin
        dSigReg <= dSigNorm;
      end
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      if (io_specialIO_isSqrt) begin // @[FDIV.scala 243:26]
        inv <= fp_a_sign & ~decode_a_isZero | decode_a_isNaN;
      end else begin
        inv <= decode_a_isInf & decode_b_isInf | decode_b_isZero & decode_a_isZero | decode_a_isNaN | decode_b_isNaN;
      end
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      dz <= _dz_T_3; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      if (io_specialIO_isSqrt) begin // @[FDIV.scala 248:30]
        inf_div <= decode_a_isInf;
      end else begin
        inf_div <= decode_a_isInf & ~decode_b_isInf;
      end
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      zero_div <= _zero_div_T_2; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      div_inf <= _div_inf_T_3; // @[Reg.scala 17:22]
    end
    if (_iterNumReg_T_4) begin // @[Reg.scala 17:18]
      iterNumReg <= iterNum; // @[Reg.scala 17:22]
    end
    if (state[0]) begin // @[Reg.scala 17:18]
      if (io_specialIO_isSqrt) begin // @[FDIV.scala 244:31]
        inv_flag <= fp_a_sign & ~decode_a_isQNaN & _inv_T | decode_a_isSNaN;
      end else begin
        inv_flag <= _inv_T_5 | decode_a_isSNaN | decode_b_isSNaN;
      end
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      subResReg <= subRes; // @[Reg.scala 17:22]
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      special_fflags <= _special_fflags_T_4; // @[Reg.scala 17:22]
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      if (inv | overflow | dz | inf_div) begin // @[FDIV.scala 272:34]
        special_exp <= 8'hff;
      end else begin
        special_exp <= 8'h0;
      end
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      if (inv) begin // @[FDIV.scala 273:34]
        special_sig <= 23'h400000;
      end else begin
        special_sig <= 23'h0;
      end
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      skipIterReg <= skipIter; // @[Reg.scala 17:22]
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      if (sqrtReg) begin // @[FDIV.scala 284:22]
        needShiftReg <= resultSigBits[0];
      end else begin
        needShiftReg <= ~resultSigBits[0];
      end
    end
    if (_T_5[0]) begin // @[Reg.scala 17:18]
      oddIterReg <= _oddIterReg_T_1; // @[Reg.scala 17:22]
    end
    if (_T_10[0]) begin // @[Reg.scala 17:18]
      if (noInf | noZero) begin // @[FDIV.scala 334:24]
        if (noInf) begin // @[FDIV.scala 332:25]
          combinedExpReg <= 8'hfe;
        end else begin
          combinedExpReg <= 8'h0;
        end
      end else if (skipIterReg) begin // @[FDIV.scala 335:8]
        combinedExpReg <= special_exp;
      end else if (rounder_io_cout & resExp != 8'hfe) begin // @[FDIV.scala 336:10]
        combinedExpReg <= _combinedExp_T_4;
      end else begin
        combinedExpReg <= resExp;
      end
    end
    if (_T_10[0]) begin // @[Reg.scala 17:18]
      if (_combinedExp_T) begin // @[FDIV.scala 337:24]
        if (noInf) begin // @[FDIV.scala 331:25]
          combinedSigReg <= 23'h7fffff;
        end else begin
          combinedSigReg <= 23'h1;
        end
      end else if (skipIterReg) begin // @[FDIV.scala 338:8]
        combinedSigReg <= special_sig;
      end else if (rounder_io_cout & resExp == 8'hfe) begin // @[FDIV.scala 339:10]
        combinedSigReg <= _combinedSig_T_3;
      end else begin
        combinedSigReg <= _combinedSig_T_4;
      end
    end
    if (_T_10[0]) begin // @[Reg.scala 17:18]
      if (skipIterReg) begin // @[FDIV.scala 326:27]
        combinedFFlagsReg <= special_fflags;
      end else begin
        combinedFFlagsReg <= normal_fflags;
      end
    end
    if (_T_10[0]) begin // @[Reg.scala 17:18]
      if (inv) begin // @[FDIV.scala 271:35]
        combinedSignReg <= 1'h0;
      end else begin
        combinedSignReg <= resSignReg;
      end
    end
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  state = _RAND_0[5:0];
  _RAND_1 = {1{`RANDOM}};
  aSubReg = _RAND_1[0:0];
  _RAND_2 = {1{`RANDOM}};
  dSubReg = _RAND_2[0:0];
  _RAND_3 = {1{`RANDOM}};
  sqrtReg = _RAND_3[0:0];
  _RAND_4 = {1{`RANDOM}};
  rmReg = _RAND_4[2:0];
  _RAND_5 = {1{`RANDOM}};
  resSignReg = _RAND_5[0:0];
  _RAND_6 = {1{`RANDOM}};
  aExp = _RAND_6[8:0];
  _RAND_7 = {1{`RANDOM}};
  dExp = _RAND_7[8:0];
  _RAND_8 = {1{`RANDOM}};
  aSigReg = _RAND_8[23:0];
  _RAND_9 = {1{`RANDOM}};
  dSigReg = _RAND_9[23:0];
  _RAND_10 = {1{`RANDOM}};
  inv = _RAND_10[0:0];
  _RAND_11 = {1{`RANDOM}};
  dz = _RAND_11[0:0];
  _RAND_12 = {1{`RANDOM}};
  inf_div = _RAND_12[0:0];
  _RAND_13 = {1{`RANDOM}};
  zero_div = _RAND_13[0:0];
  _RAND_14 = {1{`RANDOM}};
  div_inf = _RAND_14[0:0];
  _RAND_15 = {1{`RANDOM}};
  iterNumReg = _RAND_15[3:0];
  _RAND_16 = {1{`RANDOM}};
  inv_flag = _RAND_16[0:0];
  _RAND_17 = {1{`RANDOM}};
  subResReg = _RAND_17[0:0];
  _RAND_18 = {1{`RANDOM}};
  special_fflags = _RAND_18[4:0];
  _RAND_19 = {1{`RANDOM}};
  special_exp = _RAND_19[7:0];
  _RAND_20 = {1{`RANDOM}};
  special_sig = _RAND_20[22:0];
  _RAND_21 = {1{`RANDOM}};
  skipIterReg = _RAND_21[0:0];
  _RAND_22 = {1{`RANDOM}};
  needShiftReg = _RAND_22[0:0];
  _RAND_23 = {1{`RANDOM}};
  oddIterReg = _RAND_23[0:0];
  _RAND_24 = {1{`RANDOM}};
  combinedExpReg = _RAND_24[7:0];
  _RAND_25 = {1{`RANDOM}};
  combinedSigReg = _RAND_25[22:0];
  _RAND_26 = {1{`RANDOM}};
  combinedFFlagsReg = _RAND_26[4:0];
  _RAND_27 = {1{`RANDOM}};
  combinedSignReg = _RAND_27[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
