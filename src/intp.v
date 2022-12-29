/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu interpreter module
File name	: intp.v
Module name	: intp
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the interpreter module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module intp	(
		//----| system interface
		input	wire		rstn,		// reset
		input	wire		clk,		// clock

		//----| slave interface
		input	wire		slv_stt,	// slave start
		output	reg		slv_fin,	// slave finish
		input	wire	[31:0]	slv_ofs,	// slave offset
		input	wire	[31:0]	slv_siz,	// slave size
		output	wire		slv_bsy,	// slave busy

		//----| npc interface
		output	reg		npc_req,	// npc request
		input	wire		npc_gnt,	// npc grant
		output	reg		npc_rwn,	// npc read/write negative
		output	reg	[31:0]	npc_adr,	// npc address
		output	reg	[31:0]	npc_len,	// npc length
		output	wire	[63:0]	npc_wdt,	// npc write data
		input	wire	[63:0]	npc_rdt,	// npc read data
		input	wire		npc_ack,	// npc acknowledge
		
		//----| npc interface
		output	reg	[1:0]	fpu_opc,	// fpu opcode
		output	reg	[31:0]	fpu_a,		// fpu a
		output	reg	[31:0]	fpu_b,		// fpu b
		input	wire	[31:0]	fpu_y,		// fpu result
		output	reg		fpu_iv,		// fpu in valid
		output	reg		fpu_or,		// fpu out ready
		input	wire		fpu_ir,		// fpu in ready
		input	wire		fpu_ov,		// fpu out valid

		//----| sram  interface
		output	wire		sram_ena,	// sram a enable
		output	wire		sram_wea,	// sram a write enable
		output	wire	[13:0]	sram_addra,	// sram a write address
		output	wire	[63:0]	sram_dina,	// sram a data in
		output	wire		sram_enb,	// sram b read enable
		output	wire	[13:0]	sram_addrb,	// sram b read address
		input	wire	[63:0]	sram_doutb	// sram b read data
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------
//---- state
localparam S_IDLE	= 1 << 0;
localparam S_COPY_REQ	= 1 << 1;
localparam S_COPY_DATA	= 1 << 2;
localparam S_OPC_READ	= 1 << 3;
localparam S_EXEC	= 1 << 4;
localparam S_LOAD_REQ	= 1 << 5;
localparam S_LOAD_DATA	= 1 << 6;
localparam S_STORE_PRE	= 1 << 7;
localparam S_STORE_REQ	= 1 << 8;
localparam S_STORE_DATA	= 1 << 9;
localparam S_FPU1	= 1 << 10;
localparam S_FPU2	= 1 << 11;
localparam S_FOP	= 1 << 12;
localparam S_FIN	= 1 << 13;
localparam S_RETURN	= 1 << 14;

//---- opcode
localparam OPC_NOP	= 4'd0;
localparam OPC_SET_HIGH	= 4'd1;
localparam OPC_SET_LOW	= 4'd2;
localparam OPC_LOAD	= 4'd3;
localparam OPC_STORE	= 4'd4;
localparam OPC_ADD	= 4'd5;
localparam OPC_SUB	= 4'd6;
localparam OPC_MUL	= 4'd7;
localparam OPC_DIV	= 4'd8;
localparam OPC_RETURN	= 4'd9;

//----| state machine |---------------------------------------------------------
(* mark_debug = "true" *)reg	[14:0]	state;
(* mark_debug = "true" *)reg	[15:0]	scnt;
reg	[31:0]	ra, rb, rc, rd;
reg	[31:0]	ra_radr, rb_radr, rc_wadr;
reg	[31:0]	opc_radr;	
reg	[7:0]	opc_cmd;
wire		opc_div		= opc_cmd == OPC_DIV;
reg		lf_wren;
reg	[14:0]	lf_wadr;
reg	[63:0]	lf_wdat;
reg		lf_rden;
reg	[14:0]	lf_radr;
wire	[63:0]	lf_rdat		= sram_doutb;

reg		lh_wren;
reg	[14:0]	lh_wadr;
reg	[31:0]	lh_wdat;
reg		lh_rden;
reg	[14:0]	lh_radr;
wire	[31:0]	lh_rdat;

reg	[15:0]	fpu_cnt;
reg		fpu_alat;
reg		fpu_blat;
reg		fpu_ylat;
wire	[31:0]	opcode		= lh_rdat;
wire	[7:0]	opc		= opcode[00+:8];
wire	[7:0]	rno		= opcode[08+:8];
wire	[15:0]	rval		= opcode[16+:16];
wire	[15:0]	cnt		= opcode[08+:16];
always @(negedge rstn or posedge clk)
begin
	if(!rstn)
	begin
		state		<= S_IDLE;
		scnt		<= 0;
		npc_req		<= 0;
		npc_rwn		<= 0;
		npc_adr		<= 0;
		npc_len		<= 0;
		lf_wren		<= 0;
		lf_wadr		<= 0;
		lf_wdat		<= 0;
		lf_rden		<= 0;
		lf_radr		<= 0;
		lh_wren		<= 0;
		lh_wadr		<= 0;
		lh_wdat		<= 0;
		lh_rden		<= 0;
		lh_radr		<= 0;
		opc_cmd		<= 0;
		fpu_opc		<= 0;
		fpu_cnt		<= 0;
		fpu_a		<= 0;
		fpu_iv		<= 0;
		fpu_or		<= 1;
		slv_fin		<= 0;
		ra		<= 0;
		rb		<= 0;
		rc		<= 0;
		rd		<= 0;
		ra_radr		<= 0;
		rb_radr		<= 0;
		rc_wadr		<= 0;
		opc_radr	<= 0;
		fpu_alat	<= 0;
		fpu_blat	<= 0;
		fpu_ylat	<= 0;
	end
	else
	begin
		case (state)

		S_IDLE:
		begin
			state		<= slv_stt ? S_COPY_REQ : state;
			scnt		<= 0;
			npc_adr		<= slv_ofs;
			npc_len		<= slv_siz[31:3] + |slv_siz[2:0];
			npc_rwn		<= 1;
		end

		S_COPY_REQ:
		begin
			state		<= npc_gnt ? S_COPY_DATA : state;
			npc_req		<= npc_gnt ? 0 : 1;
			lf_wadr		<= 0;
		end

		S_COPY_DATA:
		begin
			state		<= npc_ack && scnt == npc_len - 1 ? S_OPC_READ : state;
			scnt		<= npc_ack ? (scnt == npc_len - 1 ? 0 : scnt + 1) : scnt;
			opc_radr	<= 0;
			lf_wren		<= npc_ack;
			lf_wadr		<= lf_wren ? lf_wadr + 2 : lf_wadr;
			lf_wdat		<= npc_rdt;
			lh_rden		<= npc_ack && scnt == npc_len - 1;
			lh_radr		<= opc_radr;
		end

		S_OPC_READ:
		begin
			state		<= S_EXEC;
			opc_radr	<= lh_rden ? opc_radr + 1 : opc_radr;
			lf_wren		<= 0;
			lh_wren		<= 0;
			lh_rden		<= 0;
			lh_radr		<= lh_rden ? lh_radr + 1 : lh_radr;
		end

		S_EXEC:
		begin
			state		<= opc == 'h03 ? S_LOAD_REQ : opc == 'h04 ? S_STORE_PRE : opc >= 'h05 && opc <= 'h08 ? S_FPU1 : opc == 'h09 ? S_RETURN : S_OPC_READ;
			ra		<= opc == 'h01 && rno == 1 ? {rval, ra[00+:16]} : opc == 'h02 && rno == 1 ? {ra[16+:16], rval} : ra;
			rb		<= opc == 'h01 && rno == 2 ? {rval, rb[00+:16]} : opc == 'h02 && rno == 2 ? {rb[16+:16], rval} : rb;
			rc		<= opc == 'h01 && rno == 3 ? {rval, rc[00+:16]} : opc == 'h02 && rno == 3 ? {rc[16+:16], rval} : rc;
			rd		<= opc == 'h01 && rno == 4 ? {rval, rd[00+:16]} : opc == 'h02 && rno == 4 ? {rd[16+:16], rval} : rd;

			ra_radr		<= ra / 4;
			rb_radr		<= rb / 4;
			rc_wadr		<= rc / 4;

			npc_req		<= opc == 'h03 ? 1 : 0;
			npc_adr		<= ra;
			npc_rwn		<= opc == 'h03;
			npc_len		<= cnt[15:1] + cnt[0];

			opc_cmd		<= opc;
			fpu_opc		<= opc >= 'h05 && opc <= 'h08 ?  opc - 'h05 : fpu_opc;
			fpu_cnt		<= cnt;

			lh_rden		<= opc >= 'h00 && opc <= 'h02 || opc >= 'h05 && opc <= 'h08;
			lf_rden		<= opc == 'h04;
			lh_radr		<= opc >= 'h05 && opc <= 'h08 ? ra / 4 : lh_radr;
			lf_radr		<= opc == 'h04 ? rb / 4 : lf_radr;
			lf_wadr		<= rb / 4;
		end

		S_LOAD_REQ:
		begin
			state		<= npc_gnt ? S_LOAD_DATA : state;
			npc_req		<= npc_gnt ? 0 : 1;
			lf_rden		<= 0;
		end

		S_LOAD_DATA:
		begin
			state		<= npc_ack && scnt == npc_len - 1 ? S_OPC_READ : state;
			scnt		<= npc_ack ? (scnt == npc_len - 1 ? 0 : scnt + 1) : scnt;
			lf_wren		<= npc_ack;
			lf_wadr		<= lf_wren ? lf_wadr + 2 : lf_wadr;
			lf_wdat		<= npc_rdt;
			lh_rden		<= npc_ack && scnt == npc_len - 1;
		end

		S_STORE_PRE:
		begin
			state		<= scnt == 3 ? S_STORE_REQ : state;
			scnt		<= scnt == 3 ? 0 : scnt + 1;
			npc_req		<= scnt == 3;
			lf_rden		<= scnt < 2;
			lf_radr		<= lf_rden ? lf_radr + 2 : lf_radr;
		end

		S_STORE_REQ:
		begin
			state		<= npc_gnt ? S_STORE_DATA : state;
			npc_req		<= npc_gnt ? 0 : 1;
		end

		S_STORE_DATA:
		begin
			state		<= npc_ack && scnt == npc_len - 1 ? S_OPC_READ : state;
			scnt		<= npc_ack ? (scnt == npc_len - 1 ? 0 : scnt + 1) : scnt;
			lf_rden		<= npc_ack && scnt != npc_len - 1;
			lf_radr		<= npc_ack && scnt == npc_len - 1 ? opc_radr : lf_rden ? lf_radr + 2 : lf_radr;

			lh_rden		<= npc_ack && scnt == npc_len - 1;
			lh_radr		<= npc_ack && scnt == npc_len - 1 ? opc_radr : lh_radr;
		end

		S_FPU1:
		begin
			state		<= S_FPU2;
			lh_rden		<= 1;
			lh_radr		<= rb_radr;
			lh_wadr		<= rc / 4;
			ra_radr		<= ra_radr + 1;
			fpu_alat	<= 1;
		end

		S_FPU2:
		begin
			state		<= !opc_div || fpu_ir ? S_FOP : state;
			lh_rden		<= fpu_cnt > 1;
			lh_radr		<= ra_radr;
			lh_wren		<= fpu_ylat;
			lh_wdat		<= fpu_ylat ? fpu_y : lh_wdat;
			rb_radr		<= !opc_div || fpu_ir ? rb_radr + 1 : rb_radr;
			fpu_a		<= fpu_alat ? lh_rdat : fpu_a;
			fpu_alat	<= 0;
			fpu_blat	<= 1;
			fpu_ylat	<= 0;
		end

		S_FOP:
		begin
			state		<= (opc_div ? fpu_ov : scnt == 1) ? (fpu_cnt == 1 ? S_FIN : S_FPU2) : state;
			scnt		<= (opc_div ? fpu_ov : scnt == 1) ? 0 : scnt + 1;
			fpu_cnt		<= (opc_div ? fpu_ov : scnt == 1) ? fpu_cnt - 1 : fpu_cnt;
			lh_rden		<= (opc_div ? fpu_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			lh_wadr		<= lh_wren ? lh_wadr + 1 : lh_wadr;
			lh_radr		<= (opc_div ? fpu_ov : scnt == 1) ? rb_radr : lh_radr;
			lh_wren		<= 0;
			ra_radr		<= (opc_div ? fpu_ov : scnt == 1) ? ra_radr + 1 : ra_radr;
			fpu_iv		<= opc_div && scnt == 1;
			fpu_or		<= 1;
			fpu_alat	<= (opc_div ? fpu_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			fpu_blat	<= 0;
			fpu_b		<= fpu_blat ? lh_rdat : fpu_b;
			fpu_ylat	<= (opc_div ? fpu_ov : scnt == 1);
		end

		S_FIN:
		begin
			state		<= S_OPC_READ;
			fpu_ylat	<= 0;
			lh_wren		<= fpu_ylat;
			lh_wdat		<= fpu_ylat ? fpu_y : lh_wdat;
			lh_rden		<= 1;
			lh_radr		<= opc_radr;
			fpu_iv		<= 0;
		end

		S_RETURN:
		begin
			state		<= scnt == 1 ? S_IDLE : state;
			scnt		<= scnt == 1 ? 0 : scnt + 1;
			slv_fin		<= scnt == 0 ? 1 : 0;
		end

		default:
		begin
			state		<= S_IDLE;
		end
		endcase		
	end
end

//----| npc wdat control |------------------------------------------------------
reg		qi;
always @(negedge rstn or posedge clk) qi <= !rstn ? 0 : (state == S_STORE_PRE || state == S_STORE_DATA) && lf_rden;
wire		qo		= state == S_STORE_DATA && npc_ack; 
reg	[1:0]	qc;
always @(negedge rstn or posedge clk) qc <= !rstn ? 0 : state == S_STORE_PRE && !scnt ? 0 : qi && !qo ? qc + 1 : !qi && qo ? qc - 1 : qc;
reg [3*64-1:0]	q;
always @(negedge rstn or posedge clk) q <= !rstn ? 0 : qi && !qo ? (qc == 0 ? {lf_rdat, q[0+:2*64]} : qc == 1 ? {q[2*64+:64], lf_rdat, q[0+:64]} : {q[1*64+:2*64], lf_rdat}) :
						      !qi &&  qo ? (qc == 1 ? {3*64'h0} : {q[0+:2*64], 64'h0}) : 
						       qi &&  qo ? (qc == 1 ? {lf_rdat, 64'h0, 64'h0} : qc == 2 ? {q[1*64+:64], lf_rdat, 64'h0} : {q[0+:2*64], lf_rdat}) : q;
assign		npc_wdt	= q[2*64+:64];

//----| sram 32bit write part |-------------------------------------------------
reg	[31:0]	lh_wlsb;
always @(negedge rstn or posedge clk) lh_wlsb <= !rstn ? 0 : lh_wren ? lh_wdat : lh_wlsb;

//----| sram 32bit read part |--------------------------------------------------
reg		lh_rden_dly;
always @(negedge rstn or posedge clk) lh_rden_dly <= !rstn ? 0 : lh_rden;
reg		lh_radr0;
always @(negedge rstn or posedge clk) lh_radr0 <= !rstn ? 0 : lh_rden ? lh_radr[0] : lh_radr0;
reg	[31:0]	lh_rmsb;
always @(negedge rstn or posedge clk) lh_rmsb <= !rstn ? 0 : lh_rden_dly ? sram_doutb[32+:32] : lh_rmsb;
//assign		lh_rdat		= !lh_radr0 ? sram_doutb[0+:32] : lh_rmsb;
assign		lh_rdat		= !lh_radr0 ? sram_doutb[0+:32] : sram_doutb[32+:32];


//----| sram signal mapping |---------------------------------------------------
assign		sram_ena	= lf_wren | lh_wren;
assign		sram_wea	= lf_wren | lh_wren;
assign		sram_addra	= lf_wren ? lf_wadr / 2 : lh_wadr / 2;
assign		sram_dina	= lf_wren ? lf_wdat : !lh_wadr[0] ? lh_wdat : {lh_wdat, lh_wlsb};
assign		sram_enb	= lf_rden | lh_rden;
assign		sram_addrb	= lf_rden ? lf_radr / 2 : lh_radr / 2;

//----| output mapping |--------------------------------------------------------
assign		slv_bsy		= state != S_IDLE;

endmodule
//==============================================================================
