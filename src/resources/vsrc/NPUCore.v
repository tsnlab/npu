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
    output reg bf16_isSqrt,
    output reg bf16_kill,

    output  reg         dma_req,
    input               dma_ready,
    output  reg         dma_rwn,
    output  reg [39:0]  dma_hostAddr,
    output  reg [15:0]  dma_localAddr,
    output  reg [15:0]  dma_transferLength,
    output      [127:0] dma_writeData,
    input       [127:0] dma_readData,
    input               dma_ack,

    //----| write signals
    output	sram_ena,		// write enable
    output	sram_wea,		// write werite enable
    output	[11:0]	sram_addra,		// write address
    output	[127:0]	sram_dina,		// write data in

    //----| read signals
    output	sram_enb,		// read enable
    output	[11:0]	sram_addrb,		// read address
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
localparam S_BF16_1	= 1 << 10;
localparam S_BF16_2	= 1 << 11;
localparam S_BF16_OP = 1 << 12;
localparam S_FIN	= 1 << 13;
localparam S_RETURN	= 1 << 14;
//---- opcode
localparam OPC_NOP	        = 8'h00;
localparam OPC_SET	        = 8'h01;
localparam OPC_SETI	        = 8'h02;
localparam OPC_SETI_LOW	    = 8'h03;
localparam OPC_SETI_HIGH    = 8'h04;
localparam OPC_GET	        = 8'h05;
localparam OPC_MOVE	        = 8'h06;
localparam OPC_LOAD	        = 8'h07;
localparam OPC_STORE	    = 8'h08;
localparam OPC_VADD_BF16	= 8'h09;
localparam OPC_VSUB_BF16	= 8'h0a;
localparam OPC_VMUL_BF16	= 8'h0b;
localparam OPC_VDIV_BF16	= 8'h0c;
localparam OPC_ADD_INT32	= 8'h0d;
localparam OPC_SUB_INT32	= 8'h0e;
localparam OPC_IFZ	        = 8'h0f;
localparam OPC_IFEQ	        = 8'h10;
localparam OPC_IFNEQ	    = 8'h11;
localparam OPC_JMP	        = 8'h12;
localparam OPC_RETURN	    = 8'hff;

reg	[14:0]	state;
reg	[15:0]	scnt;
reg [1:0]   opc_cnt;
reg	[31:0]	rf[0:15];
reg [31:0]  bf16_a_addr, bf16_b_addr, bf16_y_addr;
reg	[7:0]	opc_cmd;
wire		opc_div		= opc_cmd == OPC_VDIV_BF16;

reg [11:0]  opc_radr;
reg     localmem_rden;
reg [11:0]  localmem_radr;
reg     localmem_wren;
reg [11:0]  localmem_wadr;


wire [15:0] sram_doutb_7 = sram_doutb[127:112];
wire [15:0] sram_doutb_6 = sram_doutb[111:96];
wire [15:0] sram_doutb_5 = sram_doutb[95:80];
wire [15:0] sram_doutb_4 = sram_doutb[79:64];
wire [15:0] sram_doutb_3 = sram_doutb[63:48];
wire [15:0] sram_doutb_2 = sram_doutb[47:32];
wire [15:0] sram_doutb_1 = sram_doutb[31:16];
wire [15:0] sram_doutb_0 = sram_doutb[15:0];

reg [127:0] sram_dina_reg;
reg [127:0] sram_b_dina_reg;
reg bf16_alat, bf16_blat, bf16_ylat;
reg	[31:0]	fpu_cnt;
wire    [31:0]  opcode;
wire	[7:0]	opc		= opcode[00+:8];
wire	[19:0]	rval_u20    = {opcode[08+:4], opcode[16+:8], opcode[24+:8]};
wire	[15:0]	rval	= opcode[16+:16];
wire    [3:0]   arg_ano = opcode[12+:4];
wire    [3:0]   arg_bno = opcode[08+:4];
wire    [3:0]   arg_cno = opcode[20+:4];
wire    [3:0]   arg_dno = opcode[16+:4];

assign  opcode = opc_cnt == 2'b01 ? sram_doutb[31:0] :
                opc_cnt == 2'b10 ? sram_doutb[63:32] :
                opc_cnt == 2'b11 ? sram_doutb[95:64] : sram_doutb[127:96];

always @(negedge rstn or posedge clk) begin
	if(!rstn) begin
		state		<= S_IDLE;
		scnt		<= 0;
        opc_cnt     <= 0;
		dma_req		<= 0;
		dma_rwn		<= 0;
		dma_hostAddr    <= 0;
		dma_localAddr    <= 0;
		dma_transferLength   <= 0;
        opc_radr        <= 0;
        localmem_rden   <= 0;
        localmem_radr   <= 0;
        localmem_wren   <= 0;
        localmem_wadr   <= 0;
		opc_cmd		<= 0;
		bf16_opc		<= 0;
		fpu_cnt		<= 0;
		bf16_a		<= 0;
		bf16_b		<= 0;
        sram_dina_reg <= 0;
        sram_b_dina_reg <= 0;
		bf16_iv		<= 0;
		bf16_or		<= 1;
		rocc_if_fin	<= 0;
        bf16_a_addr <= 0;
        bf16_b_addr <= 0;
        bf16_y_addr <= 0;
        bf16_alat   <= 0;
        bf16_blat   <= 0;
        bf16_ylat   <= 0;
		rf[0]		<= 0;
		rf[1]		<= 0;
		rf[2]		<= 0;
		rf[3]		<= 0;
		rf[4]		<= 0;
		rf[5]		<= 0;
		rf[6]		<= 0;
		rf[7]		<= 0;
		rf[8]		<= 0;
		rf[9]		<= 0;
		rf[10]		<= 0;
		rf[11]		<= 0;
		rf[12]		<= 0;
		rf[13]		<= 0;
		rf[14]		<= 0;
		rf[15]		<= 0;
		bf16_isSqrt	<= 0;
		bf16_kill	<= 0;
//		opcode      <= 0;
    end else begin
		case (state)

		S_IDLE:
		begin
            if (rocc_if_cmd_vld) begin
                if (rocc_if_funct == 7'd2) begin
                    state <= S_OPC_READ;
                    dma_rwn		<= 0;
                    opc_radr	<= 0;
                    localmem_rden   <= 1;
                    localmem_radr   <= 0;
                end else if(rocc_if_funct == 7'd3) begin
                    state <= S_LOAD_REQ;
                    dma_req     <= 1;
                    dma_rwn		<= 1;
                    dma_localAddr   <= rocc_if_local_mem_offset;
                    dma_hostAddr    <= rocc_if_host_mem_offset;
                    dma_transferLength   <= rocc_if_size;
                end else if(rocc_if_funct == 7'd4) begin
                    state <= S_STORE_PRE;
                    dma_req     <= 0;
                    dma_rwn		<= 0;
                    dma_localAddr   <= rocc_if_local_mem_offset;
                    dma_hostAddr    <= rocc_if_host_mem_offset;
                    dma_transferLength   <= rocc_if_size;
                end else begin
                    state <= state;
                    dma_rwn		<= 0;
                end
            end
			scnt		<= 0;
            opc_cnt     <= 0;
		end

		S_OPC_READ:
		begin
			state		<= S_EXEC;
            opc_cnt <= opc_cnt + 1;
            localmem_wren   <= 0;
            localmem_rden   <= 0;
            opc_radr    <= localmem_rden ? (opc_cnt == 2'b11 ? opc_radr + 1 : opc_radr) : opc_radr;
            localmem_radr   <= localmem_rden ? (opc_cnt == 2'b11 ? localmem_radr + 8 : localmem_radr) : localmem_radr;
		end

		S_EXEC:
		begin
			state	    <= opc == OPC_LOAD ? S_LOAD_REQ : opc == OPC_STORE ? S_STORE_PRE : opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? S_BF16_1 : opc == OPC_RETURN ? S_RETURN : S_OPC_READ;
			rf[1]		<= opc == OPC_SETI && arg_ano == 1 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 1 ? {rval, rf[1][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 1 ? {rf[1][16+:16], rval} : rf[1];
			rf[2]		<= opc == OPC_SETI && arg_ano == 2 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 2 ? {rval, rf[2][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 2 ? {rf[2][16+:16], rval} : rf[2];
			rf[3]		<= opc == OPC_SETI && arg_ano == 3 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 3 ? {rval, rf[3][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 3 ? {rf[3][16+:16], rval} : rf[3];
			rf[4]		<= opc == OPC_SETI && arg_ano == 4 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 4 ? {rval, rf[4][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 4 ? {rf[4][16+:16], rval} : rf[4];
			rf[5]		<= opc == OPC_SETI && arg_ano == 5 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 5 ? {rval, rf[5][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 5 ? {rf[5][16+:16], rval} : rf[5];
			rf[6]		<= opc == OPC_SETI && arg_ano == 6 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 6 ? {rval, rf[6][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 6 ? {rf[6][16+:16], rval} : rf[6];
			rf[7]		<= opc == OPC_SETI && arg_ano == 7 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 7 ? {rval, rf[7][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 7 ? {rf[7][16+:16], rval} : rf[7];
			// rf[14]		<= opc == OPC_SETI_HIGH && arg_ano == 14 ? {rval, rf[14][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 14 ? {rf[14][16+:16], rval} : rf[14];
			// rf[15]	    <= opc == OPC_SETI_HIGH && arg_ano == 15 ? {rval, rf[15][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 15 ? {rf[15][16+:16], rval} : rf[15];

            bf16_a_addr <= arg_bno == 1 ? rf[1] / 2 : 
                arg_bno == 2 ? rf[2] / 2 :
                arg_bno == 3 ? rf[3] / 2 :
                arg_bno == 4 ? rf[4] / 2 :
                arg_bno == 5 ? rf[5] / 2 :
                arg_bno == 6 ? rf[6] / 2 :
                arg_bno == 7 ? rf[7] / 2 : localmem_wadr;
            bf16_b_addr <= arg_cno == 1 ? rf[1] / 2 : 
                arg_cno == 2 ? rf[2] / 2 :
                arg_cno == 3 ? rf[3] / 2 :
                arg_cno == 4 ? rf[4] / 2 :
                arg_cno == 5 ? rf[5] / 2 :
                arg_cno == 6 ? rf[6] / 2 :
                arg_cno == 7 ? rf[7] / 2 : localmem_wadr;
            bf16_y_addr   <= arg_ano == 1 ? rf[1] / 2 : 
                arg_ano == 2 ? rf[2] / 2 :
                arg_ano == 3 ? rf[3] / 2 :
                arg_ano == 4 ? rf[4] / 2 :
                arg_ano == 5 ? rf[5] / 2 :
                arg_ano == 6 ? rf[6] / 2 :
                arg_ano == 7 ? rf[7] / 2 : localmem_wadr;

			dma_req		        <= opc == OPC_LOAD ? 1 : 0;
			dma_rwn		        <= opc == OPC_LOAD;
			dma_localAddr		<= opc == OPC_LOAD || opc == OPC_STORE ? 
                opc == OPC_LOAD ? 
                arg_ano == 1 ? rf[1] : 
                arg_ano == 2 ? rf[2] :
                arg_ano == 3 ? rf[3] :
                arg_ano == 4 ? rf[4] :
                arg_ano == 5 ? rf[5] :
                arg_ano == 6 ? rf[6] :
                arg_ano == 7 ? rf[7] : dma_localAddr :

                arg_bno == 1 ? rf[1] : 
                arg_bno == 2 ? rf[2] :
                arg_bno == 3 ? rf[3] :
                arg_bno == 4 ? rf[4] :
                arg_bno == 5 ? rf[5] :
                arg_bno == 6 ? rf[6] :
                arg_bno == 7 ? rf[7] : dma_localAddr : dma_localAddr;

			dma_hostAddr		<= opc == OPC_LOAD || opc == OPC_STORE ? 
                opc == OPC_LOAD ? 
                arg_bno == 1 ? rf[1] : 
                arg_bno == 2 ? rf[2] :
                arg_bno == 3 ? rf[3] :
                arg_bno == 4 ? rf[4] :
                arg_bno == 5 ? rf[5] :
                arg_bno == 6 ? rf[6] :
                arg_bno == 7 ? rf[7] : dma_hostAddr :
                
                arg_ano == 1 ? rf[1] : 
                arg_ano == 2 ? rf[2] :
                arg_ano == 3 ? rf[3] :
                arg_ano == 4 ? rf[4] :
                arg_ano == 5 ? rf[5] :
                arg_ano == 6 ? rf[6] :
                arg_ano == 7 ? rf[7] : dma_hostAddr : dma_hostAddr;

			dma_transferLength	<= opc == OPC_LOAD || opc == OPC_STORE ? 
                arg_cno == 1 ? rf[1][15:0]/16 : 
                arg_cno == 2 ? rf[2][15:0]/16 :
                arg_cno == 3 ? rf[3][15:0]/16 :
                arg_cno == 4 ? rf[4][15:0]/16 :
                arg_cno == 5 ? rf[5][15:0]/16 :
                arg_cno == 6 ? rf[6][15:0]/16 :
                arg_cno == 7 ? rf[7][15:0]/16 : dma_transferLength : dma_transferLength;

			opc_cmd		<= opc;
			bf16_opc	<= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? opc - OPC_VADD_BF16 : bf16_opc;
			fpu_cnt		<= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? 
                arg_dno == 1 ? rf[1] : 
                arg_dno == 2 ? rf[2] :
                arg_dno == 3 ? rf[3] :
                arg_dno == 4 ? rf[4] :
                arg_dno == 5 ? rf[5] :
                arg_dno == 6 ? rf[6] :
                arg_dno == 7 ? rf[7] : fpu_cnt : fpu_cnt ;
            localmem_rden   <= (opc >= OPC_NOP && opc <= OPC_SETI_HIGH) || (opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16) || (opc == OPC_STORE);
            

            localmem_radr	<= (opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16) || (opc == OPC_STORE) ? 
                arg_bno == 1 ? rf[1] / 2 : 
                arg_bno == 2 ? rf[2] / 2 :
                arg_bno == 3 ? rf[3] / 2 :
                arg_bno == 4 ? rf[4] / 2 :
                arg_bno == 5 ? rf[5] / 2 :
                arg_bno == 6 ? rf[6] / 2 :
                arg_bno == 7 ? rf[7] / 2 : localmem_radr : localmem_radr;
		end

		S_LOAD_REQ:
		begin
			state		<= dma_ready ? S_LOAD_DATA : state;
			dma_req		<= dma_ready ? 0 : 1;
            localmem_rden <= 0;
		end

		S_LOAD_DATA:
		begin
			state		<= dma_ack && scnt == dma_transferLength - 1 ? S_OPC_READ : state;
			scnt		<= dma_ack ? (scnt == dma_transferLength - 1 ? 0 : scnt + 1) : scnt;
            
            localmem_wren <= dma_ack;
            localmem_wadr <= localmem_wren ? localmem_wadr + 8 : localmem_wadr;
            localmem_rden <= dma_ack && scnt == dma_transferLength - 1;

            sram_dina_reg <= dma_readData;
		end

		S_STORE_PRE:
		begin
			state   <= S_STORE_REQ;
			dma_req <= 1;
            localmem_rden   <= 0;
            localmem_radr   <= localmem_rden ? localmem_radr + 8 : localmem_radr;
		end

		S_STORE_REQ:
		begin
			state		<= dma_ready ? S_STORE_DATA : state;
			dma_req		<= dma_ready ? 0 : 1;
		end

		S_STORE_DATA:
		begin
			state		<= dma_ack && scnt == dma_transferLength - 1 ? S_OPC_READ : state;
			scnt		<= dma_ack ? (scnt == dma_transferLength - 1 ? 0 : scnt + 1) : scnt;
            localmem_rden   <= 1;
            localmem_radr   <= dma_ack && scnt == dma_transferLength - 1 ? opc_radr * 8 : (localmem_rden ? localmem_radr + 8 : localmem_radr);
		end

        S_BF16_1:
        begin
            state   <= S_BF16_2;
            localmem_rden   <= 1;
            localmem_radr   <= bf16_b_addr;
            localmem_wadr <= bf16_y_addr;
            bf16_a_addr <= bf16_a_addr + 1;
            bf16_alat <= 1;
        end

        S_BF16_2:
        begin
			state           <= !opc_div || bf16_ir ? S_BF16_OP : state;
			localmem_rden   <= fpu_cnt > 1;
			localmem_radr   <= bf16_a_addr;
			localmem_wren   <= bf16_ylat;
            
            sram_dina_reg[15:0] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b000 ? bf16_y : sram_dina_reg[15:0]) : sram_dina_reg[15:0];
            sram_dina_reg[31:16] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b001 ? bf16_y : sram_dina_reg[31:16]) : sram_dina_reg[31:16];
            sram_dina_reg[47:32] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b010 ? bf16_y : sram_dina_reg[47:32]) : sram_dina_reg[47:32];
            sram_dina_reg[63:48] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b011 ? bf16_y : sram_dina_reg[63:48]) : sram_dina_reg[63:48];
            sram_dina_reg[79:64] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b100 ? bf16_y : sram_dina_reg[79:64]) : sram_dina_reg[79:64];
            sram_dina_reg[95:80] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b101 ? bf16_y : sram_dina_reg[95:80]) : sram_dina_reg[95:80];
            sram_dina_reg[111:96] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b110 ? bf16_y : sram_dina_reg[111:96]) : sram_dina_reg[111:96];
            sram_dina_reg[127:112] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b111 ? bf16_y : sram_dina_reg[127:112]) : sram_dina_reg[127:112];
			
                // lh_wdat		<= bf16_ylat ? fpu_y : lh_wdat;
			bf16_b_addr		<= !opc_div || bf16_ir ? bf16_b_addr + 1 : bf16_b_addr;
            
            bf16_a  <= bf16_alat ? ( 
                bf16_a_addr[2:0] == 3'b001 ? sram_doutb_0 :
                bf16_a_addr[2:0] == 3'b010 ? sram_doutb_1 :
                bf16_a_addr[2:0] == 3'b011 ? sram_doutb_2 :
                bf16_a_addr[2:0] == 3'b100 ? sram_doutb_3 :
                bf16_a_addr[2:0] == 3'b101 ? sram_doutb_4 :
                bf16_a_addr[2:0] == 3'b110 ? sram_doutb_5 :
                bf16_a_addr[2:0] == 3'b111 ? sram_doutb_6 : sram_doutb_7) : bf16_a;

            bf16_alat   <= 0;
            bf16_blat   <= 1;
			bf16_ylat	<= 0;
        end

        S_BF16_OP:
        begin
			state		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? S_FIN : S_BF16_2) : state;
			scnt		<= (opc_div ? bf16_ov : scnt == 1) ? 0 : scnt + 1;
			fpu_cnt		<= (opc_div ? bf16_ov : scnt == 1) ? fpu_cnt - 1 : fpu_cnt;

			localmem_rden   <= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			localmem_wadr   <= localmem_wren ? localmem_wadr + 1 : localmem_wadr;
			localmem_radr   <= (opc_div ? bf16_ov : scnt == 1) ? bf16_b_addr : localmem_radr;
			localmem_wren   <= 0;

			bf16_a_addr <= (opc_div ? bf16_ov : scnt == 1) ? bf16_a_addr + 1 : bf16_a_addr;
			bf16_iv		<= opc_div && scnt == 1;
			bf16_or		<= 1;
			bf16_alat	<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			bf16_blat	<= 0;
            bf16_b      <= bf16_blat ? ( 
                bf16_b_addr[2:0] == 3'b001 ? sram_doutb_0 :
                bf16_b_addr[2:0] == 3'b010 ? sram_doutb_1 :
                bf16_b_addr[2:0] == 3'b011 ? sram_doutb_2 :
                bf16_b_addr[2:0] == 3'b100 ? sram_doutb_3 :
                bf16_b_addr[2:0] == 3'b101 ? sram_doutb_4 :
                bf16_b_addr[2:0] == 3'b110 ? sram_doutb_5 :
                bf16_b_addr[2:0] == 3'b111 ? sram_doutb_6 : sram_doutb_7) : bf16_b;
			bf16_ylat	<= (opc_div ? bf16_ov : scnt == 1);
        end

        S_FIN:
        begin
			state		<= S_OPC_READ;
			bf16_ylat	<= 0;

			localmem_wren   <= bf16_ylat;
			// lh_wdat		<= fpu_ylat ? fpu_y : lh_wdat;
            
            sram_dina_reg[15:0] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b000 ? bf16_y : sram_dina_reg[15:0]) : sram_dina_reg[15:0];
            sram_dina_reg[31:16] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b001 ? bf16_y : sram_dina_reg[31:16]) : sram_dina_reg[31:16];
            sram_dina_reg[47:32] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b010 ? bf16_y : sram_dina_reg[47:32]) : sram_dina_reg[47:32];
            sram_dina_reg[63:48] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b011 ? bf16_y : sram_dina_reg[63:48]) : sram_dina_reg[63:48];
            sram_dina_reg[79:64] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b100 ? bf16_y : sram_dina_reg[79:64]) : sram_dina_reg[79:64];
            sram_dina_reg[95:80] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b101 ? bf16_y : sram_dina_reg[95:80]) : sram_dina_reg[95:80];
            sram_dina_reg[111:96] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b110 ? bf16_y : sram_dina_reg[111:96]) : sram_dina_reg[111:96];
            sram_dina_reg[127:112] <= bf16_ylat ? (localmem_wadr[2:0] == 3'b111 ? bf16_y : sram_dina_reg[127:112]) : sram_dina_reg[127:112];
            
			localmem_rden   <= 1;
			localmem_radr   <= opc_radr * 8;
			bf16_iv         <= 0;
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

assign		dma_writeData	= sram_doutb;


assign		sram_ena	= localmem_wren;
assign		sram_wea	= localmem_wren;
assign		sram_addra	= localmem_wadr / 8;
assign		sram_dina	= sram_dina_reg;
assign		sram_enb	= localmem_rden;
assign		sram_addrb	= localmem_radr / 8;

//----| output mapping |--------------------------------------------------------
assign		rocc_if_busy		= state != S_IDLE;
    
endmodule
