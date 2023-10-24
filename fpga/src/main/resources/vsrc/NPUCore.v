module NPUCore
(
    input clk,
    input rstn,

    input [39:0] rocc_if_host_mem_offset,
    input [15:0] rocc_if_size,
    input [15:0] rocc_if_local_mem_offset,
    input [6:0] rocc_if_funct,
    input rocc_if_cmd_vld,
    output reg rocc_if_fin,
    output rocc_if_busy,

    output reg [1:0] bf16_opc,
    output reg [15:0] bf16_a, 
    output reg [15:0] bf16_b,
    input [15:0] bf16_y,
    output reg bf16_iv,
    output reg bf16_or,
    input bf16_ov,
    input bf16_ir,

    // output int32_opc,
    // output [31:0] int32_a,
    // output [31:0] int32_b,
    // input [31:0] int32_y,
    output  reg         dma_req,
    input               dma_ready,
    output  reg         dma_rwn,
    output  reg [39:0]  dma_hostAddr,
    output  reg [13:0]  dma_localAddr,
    output  reg [15:0]  dma_tansferLength,
    output  reg [127:0] dma_writeData,
    input       [127:0] dma_readData,
    input               dma_ack,

    //----| write signals
    output	sram_ena,		// write enable
    output	sram_wea,		// write werite enable
    output	[13:0]	sram_addra,		// write address
    output	[127:0]	sram_dina,		// write data in

    //----| read signals
    output	sram_enb,		// read enable
    output	[13:0]	sram_addrb,		// read address
    input	[127:0]	sram_doutb		// read data out
);
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
localparam OPC_NOP	        = 5'd0;
localparam OPC_SET	        = 5'd1;
localparam OPC_SETI	        = 5'd2;
localparam OPC_SETI_LOW	    = 5'd3;
localparam OPC_SETI_HIGH    = 5'd4;
localparam OPC_GET	        = 5'd5;
localparam OPC_MOVE	        = 5'd6;
localparam OPC_LOAD	        = 5'd7;
localparam OPC_STORE	    = 5'd8;
localparam OPC_VADD_BF16	= 5'd9;
localparam OPC_VSUB_BF16	= 5'd10;
localparam OPC_VMUL_BF16	= 5'd11;
localparam OPC_VDIV_BF16	= 5'd12;
localparam OPC_ADD_INT32	= 5'd13;
localparam OPC_SUB_INT32	= 5'd14;
localparam OPC_IFZ	        = 5'd15;
localparam OPC_IFEQ	        = 5'd16;
localparam OPC_IFNEQ	    = 5'd17;
localparam OPC_JMP	        = 5'd18;
localparam OPC_RETURN	    = 5'd19;

reg	[14:0]	state;
reg	[15:0]	scnt;
reg	[31:0]	ra, rb, rc, rd;
reg	[31:0]	ra_radr, rb_radr, rc_wadr;
reg	[31:0]	opc_radr;	
reg	[7:0]	opc_cmd;
wire		opc_div		= opc_cmd == OPC_VDIV_BF16;
reg		lf_wren;
reg	[14:0]	lf_wadr;
reg	[127:0]	lf_wdat;
reg		lf_rden;
reg	[14:0]	lf_radr;
wire	[127:0]	lf_rdat		= sram_doutb;

reg		lh_wren;
reg	[14:0]	lh_wadr;
reg	[31:0]	lh_wdat;
reg		lh_rden;
reg	[14:0]	lh_radr;
wire	[31:0]	lh_rdat;

reg	[15:0]	fpu_cnt;
reg		bf16_alat;
reg		bf16_blat;
reg		bf16_ylat;
wire	[31:0]	opcode		= lh_rdat;
wire	[7:0]	opc		= opcode[00+:8];
wire	[7:0]	rno		= opcode[08+:8];
wire	[15:0]	rval	= opcode[16+:16];
wire	[15:0]	cnt		= opcode[08+:16];

always @(negedge rstn or posedge clk) begin
	if(!rstn) begin
		state		<= S_IDLE;
		scnt		<= 0;
		dma_req		<= 0;
		dma_rwn		<= 0;
		dma_hostAddr    <= 0;
		dma_tansferLength   <= 0;
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
		bf16_opc		<= 0;
		fpu_cnt		<= 0;
		bf16_a		<= 0;
		bf16_iv		<= 0;
		bf16_or		<= 1;
		rocc_if_fin	<= 0;
		ra		<= 0;
		rb		<= 0;
		rc		<= 0;
		rd		<= 0;
		ra_radr		<= 0;
		rb_radr		<= 0;
		rc_wadr		<= 0;
		opc_radr	<= 0;
		bf16_alat	<= 0;
		bf16_blat	<= 0;
		bf16_ylat	<= 0;
    end else begin
		case (state)

		S_IDLE:
		begin
            if (rocc_if_cmd_vld) begin
                if (rocc_if_funct == 7'd2) begin
                    state <= S_OPC_READ;
                    dma_rwn		<= 0;
                    opc_radr	<= 0;
                end else if(rocc_if_funct == 7'd3) begin
                    state <= S_LOAD_REQ;
                    dma_rwn		<= 1;
                end else if(rocc_if_funct == 7'd4) begin
                    state <= S_STORE_PRE;
                    dma_rwn		<= 0;
                end else begin
                    state <= state;
                    dma_rwn		<= 0;
                end
            end
			scnt		<= 0;
			dma_hostAddr        <= rocc_if_host_mem_offset;
			dma_tansferLength	<= rocc_if_size[15:3] + |rocc_if_size[2:0];
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
			state		<= opc == 'h03 ? S_LOAD_REQ : opc == 'h04 ? S_STORE_PRE : opc >= 'h09 && opc <= 'h0c ? S_FPU1 : opc == 'h13 ? S_RETURN : S_OPC_READ;
			ra		<= opc == 'h01 && rno == 1 ? {rval, ra[00+:16]} : opc == 'h02 && rno == 1 ? {ra[16+:16], rval} : ra;
			rb		<= opc == 'h01 && rno == 2 ? {rval, rb[00+:16]} : opc == 'h02 && rno == 2 ? {rb[16+:16], rval} : rb;
			rc		<= opc == 'h01 && rno == 3 ? {rval, rc[00+:16]} : opc == 'h02 && rno == 3 ? {rc[16+:16], rval} : rc;
			rd		<= opc == 'h01 && rno == 4 ? {rval, rd[00+:16]} : opc == 'h02 && rno == 4 ? {rd[16+:16], rval} : rd;

			ra_radr		<= ra / 4;
			rb_radr		<= rb / 4;
			rc_wadr		<= rc / 4;

			dma_req		<= opc == 'h03 ? 1 : 0;
			dma_hostAddr		<= ra;
			dma_rwn		<= opc == 'h03;
			dma_tansferLength		<= cnt[15:1] + cnt[0];

			opc_cmd		<= opc;
			bf16_opc		<= opc >= 'h05 && opc <= 'h08 ?  opc - 'h05 : bf16_opc;
			fpu_cnt		<= cnt;

			lh_rden		<= opc >= 'h00 && opc <= 'h02 || opc >= 'h05 && opc <= 'h08;
			lf_rden		<= opc == 'h04;
			lh_radr		<= opc >= 'h05 && opc <= 'h08 ? ra / 4 : lh_radr;
			lf_radr		<= opc == 'h04 ? rb / 4 : lf_radr;
			lf_wadr		<= rb / 4;
		end

		S_LOAD_REQ:
		begin
			state		<= dma_ready ? S_LOAD_DATA : state;
			dma_req		<= dma_ready ? 0 : 1;
			lf_rden		<= 0;
		end

		S_LOAD_DATA:
		begin
			state		<= dma_ack && scnt == dma_tansferLength - 1 ? S_OPC_READ : state;
			scnt		<= dma_ack ? (scnt == dma_tansferLength - 1 ? 0 : scnt + 1) : scnt;
			lf_wren		<= dma_ack;
			lf_wadr		<= lf_wren ? lf_wadr + 2 : lf_wadr;
			lf_wdat		<= dma_readData;
			lh_rden		<= dma_ack && scnt == dma_tansferLength - 1;
			lh_radr		<= rocc_if_local_mem_offset;
		end

		S_STORE_PRE:
		begin
			state		<= scnt == 3 ? S_STORE_REQ : state;
			scnt		<= scnt == 3 ? 0 : scnt + 1;
			dma_req		<= scnt == 3;
			lf_rden		<= scnt < 2;
			lf_radr		<= lf_rden ? lf_radr + 2 : lf_radr;
		end

		S_STORE_REQ:
		begin
			state		<= dma_ready ? S_STORE_DATA : state;
			dma_req		<= dma_ready ? 0 : 1;
		end

		S_STORE_DATA:
		begin
			state		<= dma_ack && scnt == dma_tansferLength - 1 ? S_OPC_READ : state;
			scnt		<= dma_ack ? (scnt == dma_tansferLength - 1 ? 0 : scnt + 1) : scnt;
			lf_rden		<= dma_ack && scnt != dma_tansferLength - 1;
			lf_radr		<= dma_ack && scnt == dma_tansferLength - 1 ? opc_radr : lf_rden ? lf_radr + 2 : lf_radr;

			lh_rden		<= dma_ack && scnt == dma_tansferLength - 1;
			lh_radr		<= dma_ack && scnt == dma_tansferLength - 1 ? opc_radr : lh_radr;
		end

		S_FPU1:
		begin
			state		<= S_FPU2;
			lh_rden		<= 1;
			lh_radr		<= rb_radr;
			lh_wadr		<= rc / 4;
			ra_radr		<= ra_radr + 1;
			bf16_alat	<= 1;
		end

		S_FPU2:
		begin
			state		<= !opc_div || bf16_ir ? S_FOP : state;
			lh_rden		<= fpu_cnt > 1;
			lh_radr		<= ra_radr;
			lh_wren		<= bf16_ylat;
			lh_wdat		<= bf16_ylat ? bf16_y : lh_wdat;
			rb_radr		<= !opc_div || bf16_ir ? rb_radr + 1 : rb_radr;
			bf16_a		<= bf16_alat ? lh_rdat : bf16_a;
			bf16_alat	<= 0;
			bf16_blat	<= 1;
			bf16_ylat	<= 0;
		end

		S_FOP:
		begin
			state		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? S_FIN : S_FPU2) : state;
			scnt		<= (opc_div ? bf16_ov : scnt == 1) ? 0 : scnt + 1;
			fpu_cnt		<= (opc_div ? bf16_ov : scnt == 1) ? fpu_cnt - 1 : fpu_cnt;
			lh_rden		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			lh_wadr		<= lh_wren ? lh_wadr + 1 : lh_wadr;
			lh_radr		<= (opc_div ? bf16_ov : scnt == 1) ? rb_radr : lh_radr;
			lh_wren		<= 0;
			ra_radr		<= (opc_div ? bf16_ov : scnt == 1) ? ra_radr + 1 : ra_radr;
			bf16_iv		<= opc_div && scnt == 1;
			bf16_or		<= 1;
			bf16_alat	<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			bf16_blat	<= 0;
			bf16_b		<= bf16_blat ? lh_rdat : bf16_b;
			bf16_ylat	<= (opc_div ? bf16_ov : scnt == 1);
		end

		S_FIN:
		begin
			state		<= S_OPC_READ;
			bf16_ylat	<= 0;
			lh_wren		<= bf16_ylat;
			lh_wdat		<= bf16_ylat ? bf16_y : lh_wdat;
			lh_rden		<= 1;
			lh_radr		<= opc_radr;
			bf16_iv		<= 0;
		end

		S_RETURN:
		begin
			state		<= scnt == 1 ? S_IDLE : state;
			scnt		<= scnt == 1 ? 0 : scnt + 1;
			rocc_if_fin		<= scnt == 0 ? 1 : 0;
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
wire		qo		= state == S_STORE_DATA && dma_ack; 
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
assign		rocc_if_busy		= state != S_IDLE;
    
endmodule