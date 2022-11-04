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
		output	wire	[31:0]	npc_wdt,	// npc write data
		input	wire	[31:0]	npc_rdt,	// npc read data
		input	wire		npc_ack,	// npc acknowledge
		
		//----| npc interface
		output	reg	[1:0]	fpu_opc,	// fpu opcode
		output	reg	[31:0]	fpu_a,		// fpu a
		output	wire	[31:0]	fpu_b,		// fpu b
		input	wire	[31:0]	fpu_y,		// fpu result
		output	reg		fpu_iv,		// fpu in valid
		output	reg		fpu_or,		// fpu out ready
		input	wire		fpu_ir,		// fpu in ready
		input	wire		fpu_ov,		// fpu out valid

		//----| sram  interface
		output	wire		sram_ena,	// sram a enable
		output	wire		sram_wea,	// sram a write enable
		output	wire	[14:0]	sram_addra,	// sram a write address
		output	wire	[31:0]	sram_dina,	// sram a data in
		output	wire		sram_enb,	// sram b read enable
		output	wire	[14:0]	sram_addrb,	// sram b read address
		input	wire	[31:0]	sram_doutb	// sram b read data
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
localparam S_RETURN	= 1 << 13;

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
reg	[13:0]	state;
reg	[15:0]	scnt;
reg	[31:0]	ra, rb, rc, rd;
reg	[31:0]	ra_radr, rb_radr, rc_wadr;
reg	[31:0]	opc_radr;	
reg	[7:0]	opc_cmd;
wire		opc_div		= opc_cmd == OPC_DIV;
reg		lm_wren;
reg	[14:0]	lm_wadr;
reg	[31:0]	lm_wdat;
reg		lm_rden;
reg	[14:0]	lm_radr;
wire	[31:0]	lm_rdat		= sram_doutb;
reg	[15:0]	fpu_cnt;
assign		fpu_b		= lm_rdat;
wire	[31:0]	opcode		= lm_rdat;
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
		lm_wren		<= 0;
		lm_wadr		<= 0;
		lm_wdat		<= 0;
		lm_rden		<= 0;
		lm_radr		<= 0;
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
	end
	else
	begin
		case (state)

		S_IDLE:
		begin
			state		<= slv_stt ? S_COPY_REQ : state;
			scnt		<= 0;
			npc_adr		<= slv_ofs;
			npc_len		<= slv_siz / 4;
			npc_rwn		<= 1;
		end

		S_COPY_REQ:
		begin
			state		<= npc_gnt ? S_COPY_DATA : state;
			npc_req		<= npc_gnt ? 0 : 1;
			lm_wadr		<= 0;
		end

		S_COPY_DATA:
		begin
			state		<= npc_ack && scnt == npc_len - 1 ? S_OPC_READ : state;
			scnt		<= npc_ack ? (scnt == npc_len - 1 ? 0 : scnt + 1) : scnt;
			lm_wren		<= npc_ack;
			lm_wadr		<= lm_wren ? lm_wadr + 1 : lm_wadr;
			lm_wdat		<= npc_rdt;
			lm_rden		<= npc_ack && scnt == npc_len - 1;
			opc_radr	<= 0;
			lm_radr		<= opc_radr;
		end

		S_OPC_READ:
		begin
			state		<= S_EXEC;
			lm_wren		<= 0;
			lm_rden		<= 0;
			lm_radr		<= lm_rden ? lm_radr + 1 : lm_radr;
			opc_radr	<= lm_rden ? opc_radr + 1 : opc_radr;
			lm_wren		<= 0;
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
			npc_len		<= cnt;

			opc_cmd		<= opc;
			fpu_opc		<= opc >= 'h05 && opc <= 'h08 ?  opc - 'h05 : fpu_opc;
			fpu_cnt		<= cnt;

			lm_rden		<= opc >= 'h00 && opc <= 'h02 || opc == 'h04 ||	opc >= 'h05 && opc <= 'h08;
			lm_radr		<= opc >= 'h05 && opc <= 'h08 ? ra / 4 : opc == 'h04 ? rb / 4 : lm_radr;
			lm_wadr		<= rb / 4;
		end

		S_LOAD_REQ:
		begin
			state		<= npc_gnt ? S_LOAD_DATA : state;
			npc_req		<= npc_gnt ? 0 : 1;
			lm_rden		<= 0;
		end

		S_LOAD_DATA:
		begin
			state		<= npc_ack && scnt == npc_len - 1 ? S_OPC_READ : state;
			scnt		<= npc_ack ? (scnt == npc_len - 1 ? 0 : scnt + 1) : scnt;
			lm_wren		<= npc_ack;
			lm_wadr		<= lm_wren ? lm_wadr + 1 : lm_wadr;
			lm_wdat		<= npc_rdt;
			lm_rden		<= npc_ack && scnt == npc_len - 1;
		end

		S_STORE_PRE:
		begin
			state		<= scnt == 3 ? S_STORE_REQ : state;
			scnt		<= scnt == 3 ? 0 : scnt + 1;
			npc_req		<= scnt == 3;
			lm_rden		<= scnt < 2;
			lm_radr		<= lm_rden ? lm_radr + 1 : lm_radr;
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
			lm_rden		<= npc_ack;
			lm_radr		<= npc_ack && scnt == npc_len - 1 ? opc_radr : lm_rden ? lm_radr + 1 : lm_radr;
		end

		S_FPU1:
		begin
			state		<= S_FPU2;
			lm_rden		<= 1;
			lm_radr		<= rb_radr;
			ra_radr		<= ra_radr + 1;
			lm_wadr		<= rc / 4;
		end

		S_FPU2:
		begin
			state		<= !opc_div || fpu_ir ? S_FOP : state;
			lm_rden		<= fpu_cnt > 1;
			fpu_a		<= lm_rdat;
			lm_radr		<= ra_radr;
			rb_radr		<= !opc_div || fpu_ir ? rb_radr + 1 : rb_radr;
			lm_wren		<= 0;
			lm_wadr		<= lm_wren ? lm_wadr + 1 : lm_wadr;
			fpu_iv		<= opc_div & fpu_ir;
		end

		S_FOP:
		begin
			state		<= (opc_div ? fpu_ov : !scnt) ? (fpu_cnt == 1 ? S_OPC_READ : S_FPU2) : state;
			scnt		<= (opc_div ? fpu_ov : !scnt) ? 0 : scnt + 1;
			fpu_cnt		<= (opc_div ? fpu_ov : !scnt) ? fpu_cnt - 1 : fpu_cnt;
			lm_rden		<= (opc_div ? fpu_ov : !scnt);

			lm_radr		<= (opc_div ? fpu_ov : !scnt) ? (fpu_cnt == 1 ? opc_radr : rb_radr) : lm_radr;
			lm_wren		<= (opc_div ? fpu_ov : !scnt);
			lm_wdat		<= fpu_y;
			ra_radr		<= (opc_div ? fpu_ov : !scnt) ? ra_radr + 1 : ra_radr;
			fpu_iv		<= 0;
			fpu_or		<= 1; // opc_div && fpu_ov;
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
always @(negedge rstn or posedge clk) qi <= !rstn ? 0 : (state == S_STORE_PRE || state == S_STORE_DATA) && lm_rden;
wire		qo		= state == S_STORE_DATA && npc_ack; 
reg	[1:0]	qc;
always @(negedge rstn or posedge clk) qc <= !rstn ? 0 : state == S_STORE_PRE && !scnt ? 0 : qi && !qo ? qc + 1 : !qi && qo ? qc - 1 : qc;
reg [3*32-1:0]	q;
always @(negedge rstn or posedge clk) q <= !rstn ? 0 : qi && !qo ? (qc == 0 ? {lm_rdat, q[63:0]} : qc == 1 ? {q[95:64], lm_rdat, q[31:0]} : {q[95:32], lm_rdat}) :
						      !qi &&  qo ? (qc == 1 ? {96'h0} : {q[63:0], 32'h0}) : 
						       qi &&  qo ? (qc == 1 ? {lm_rdat, 64'h0} : qc == 2 ? {q[63:32], lm_rdat, 32'h0} : {q[63:0], lm_rdat}) : q;
assign		npc_wdt	= q[95:64];

//----| sram signal mapping |---------------------------------------------------
assign		sram_ena	= lm_wren;
assign		sram_wea	= lm_wren;
assign		sram_addra	= lm_wadr;
assign		sram_dina	= lm_wdat;
assign		sram_enb	= lm_rden;
assign		sram_addrb	= lm_radr;

//----| output mapping |--------------------------------------------------------
assign		slv_bsy		= state != S_IDLE;

endmodule
//==============================================================================
