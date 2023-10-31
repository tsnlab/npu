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

    // output int32_opc,
    // output [31:0] int32_a,
    // output [31:0] int32_b,
    // input [31:0] int32_y,
    output  reg         dma_req,
    input               dma_ready,
    output  reg         dma_rwn,
    output  reg [39:0]  dma_hostAddr,
    output  reg [11:0]  dma_localAddr,
    output  reg [15:0]  dma_tansferLength,
    output      [127:0] dma_writeData,
    input       [127:0] dma_readData,
    input               dma_ack,

    //----| write signals
    output	sram_a_ena,		// write enable
    output	sram_a_wea,		// write werite enable
    output	[11:0]	sram_a_addra,		// write address
    output	[127:0]	sram_a_dina,		// write data in

    //----| read signals
    output	sram_a_enb,		// read enable
    output	[11:0]	sram_a_addrb,		// read address
    input	[127:0]	sram_a_doutb,		// read data out
    //----| write signals
    output	sram_b_ena,		// write enable
    output	sram_b_wea,		// write werite enable
    output	[11:0]	sram_b_addra,		// write address
    output	[127:0]	sram_b_dina,		// write data in

    //----| read signals
    output	sram_b_enb,		// read enable
    output	[11:0]	sram_b_addrb,		// read address
    input	[127:0]	sram_b_doutb		// read data out
);
//---- state
localparam A_KERNEL	= 13'b0000000000000;
localparam A_SRC_A	= 13'b0100000000000;
localparam A_SRC_B	= 13'b1000000000000;
localparam A_DST_Y	= 13'b1100000000000;

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
localparam S_BF16	= 1 << 10;
localparam S_FOP	= 1 << 11;
localparam S_FIN	= 1 << 12;
localparam S_RETURN	= 1 << 13;
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
reg	[31:0]	rf[0:15];
//reg	[31:0]	ra_radr, rb_radr, rc_wadr;
//reg	[31:0]	opc_radr;	
reg	[7:0]	opc_cmd;
wire		opc_div		= opc_cmd == OPC_VDIV_BF16;

reg     kernel_wren;
reg     operanda_wren;
reg     operandb_wren;
reg     resultc_wren;

reg	[11:0]  kernel_wadr;
reg	[11:0]  operanda_wadr;
reg	[11:0]  operandb_wadr;
reg	[11:0]  resultc_wadr;

reg	[11:0]	kernel_radr;
reg	[11:0]	operanda_radr;
reg	[11:0]	operandb_radr;
reg	[11:0]	resultc_radr;

reg		kernel_rden;
reg     operanda_rden;
reg     operandb_rden;
reg     resultc_rden;
//wire	[127:0]	lf_rdat		= sram_doutb;

reg sram_a_wea_reg;
reg sram_b_wea_reg;
reg sram_a_ena_reg;
reg sram_b_ena_reg;
reg [11:0]  sram_a_addra_reg;
reg [11:0]  sram_b_addra_reg;
reg [127:0] sram_a_dina_reg;
reg [127:0] sram_b_dina_reg;
reg sram_a_enb_reg;
reg sram_b_enb_reg;
reg [11:0]  sram_a_addrb_reg;
reg [11:0]  sram_b_addrb_reg;
reg [127:0] sram_a_doutb_reg;
reg [127:0] sram_b_doutb_reg;
reg		lh_wren;
reg	[14:0]	lh_wadr;
reg	[31:0]	lh_wdat;
// reg		lh_rden;
//reg		sram_rden;
//reg	[14:0]	lh_radr;
//wire	[31:0]	lh_rdat;

reg	[31:0]	fpu_cnt;
//reg		bf16_alat;
//reg		bf16_blat;
//reg		bf16_ylat;
wire	[31:0]	opcode		= sram_a_doutb;
wire	[7:0]	opc		= opcode[00+:8];
//wire	[3:0]	rno		= opcode[08+:4];
wire	[15:0]	rval_u20    = opcode[12+:20];
wire	[15:0]	rval	= opcode[16+:16];
// wire	[15:0]	cnt		= opcode[08+:16];
//wire	[15:0]	lscnt_no    = opcode[16+:4];
// wire	[15:0]	bf16_exec_cnt_no    = opcode[20+:4];
wire    [3:0]   arg_ano = opcode[08+:4];
wire    [3:0]   arg_bno = opcode[12+:4];
wire    [3:0]   arg_cno = opcode[16+:4];
wire    [3:0]   arg_dno = opcode[20+:4];

always @(negedge rstn or posedge clk) begin
	if(!rstn) begin
		state		<= S_IDLE;
		scnt		<= 0;
		dma_req		<= 0;
		dma_rwn		<= 0;
		dma_hostAddr    <= 0;
		dma_tansferLength   <= 0;
		kernel_wren		<= 0;
		kernel_wadr		<= 0;
		kernel_rden		<= 0;
		kernel_radr		<= 0;
		operanda_wren	<= 0;
		operanda_wadr	<= 0;
		operanda_rden	<= 0;
		operanda_radr	<= 0;
		operandb_wren	<= 0;
		operandb_wadr	<= 0;
		operandb_rden	<= 0;
		operandb_radr	<= 0;
		resultc_wren	<= 0;
		resultc_wadr	<= 0;
		resultc_rden	<= 0;
		resultc_radr	<= 0;
		opc_cmd		<= 0;
		bf16_opc		<= 0;
		fpu_cnt		<= 0;
		bf16_a		<= 0;
		bf16_iv		<= 0;
		bf16_or		<= 1;
		rocc_if_fin	<= 0;
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
    end else begin
		case (state)

		S_IDLE:
		begin
            if (rocc_if_cmd_vld) begin
                if (rocc_if_funct == 7'd2) begin
                    state <= S_OPC_READ;
                    dma_rwn		<= 0;
                    // opc_radr	<= 0;
                    kernel_rden		<= 1;
                end else if(rocc_if_funct == 7'd3) begin
                    state <= S_LOAD_REQ;
                    dma_req     <= 1;
                    dma_rwn		<= 1;
                    dma_localAddr   <= rocc_if_local_mem_offset;
                    dma_hostAddr    <= rocc_if_host_mem_offset;
                    dma_tansferLength   <= rocc_if_size;
                end else if(rocc_if_funct == 7'd4) begin
                    state <= S_STORE_PRE;
                    dma_req     <= 0;
                    dma_rwn		<= 0;
                    dma_localAddr   <= rocc_if_local_mem_offset;
                    dma_hostAddr    <= rocc_if_host_mem_offset;
                    dma_tansferLength   <= rocc_if_size;
                end else begin
                    state <= state;
                    dma_rwn		<= 0;
                end
            end
			scnt		<= 0;
		end

		S_OPC_READ:
		begin
			state		<= S_EXEC;
			kernel_wren		<= 0;
			kernel_rden		<= 0;
			kernel_radr		<= kernel_rden ? kernel_radr + 1 : kernel_radr;
		end

		S_EXEC:
		begin
			state	    <= opc == OPC_LOAD ? S_LOAD_REQ : opc == OPC_STORE ? S_STORE_PRE : opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? S_BF16 : opc == OPC_RETURN ? S_RETURN : S_OPC_READ;
			rf[1]		<= opc == OPC_SETI && arg_ano == 1 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 1 ? {rval, rf[1][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 1 ? {rf[1][16+:16], rval} : rf[1];
			rf[2]		<= opc == OPC_SETI && arg_ano == 2 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 2 ? {rval, rf[2][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 2 ? {rf[2][16+:16], rval} : rf[2];
			rf[3]		<= opc == OPC_SETI && arg_ano == 3 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 3 ? {rval, rf[3][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 3 ? {rf[3][16+:16], rval} : rf[3];
			rf[4]		<= opc == OPC_SETI && arg_ano == 4 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 4 ? {rval, rf[4][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 4 ? {rf[4][16+:16], rval} : rf[4];
			rf[5]		<= opc == OPC_SETI && arg_ano == 5 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 5 ? {rval, rf[5][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 5 ? {rf[5][16+:16], rval} : rf[5];
			rf[6]		<= opc == OPC_SETI && arg_ano == 6 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 6 ? {rval, rf[6][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 6 ? {rf[6][16+:16], rval} : rf[6];
			rf[7]		<= opc == OPC_SETI && arg_ano == 7 ? {12'h000, rval_u20} : opc == OPC_SETI_HIGH && arg_ano == 7 ? {rval, rf[7][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 7 ? {rf[7][16+:16], rval} : rf[7];
			// rf[14]		<= opc == OPC_SETI_HIGH && arg_ano == 14 ? {rval, rf[14][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 14 ? {rf[14][16+:16], rval} : rf[14];
			// rf[15]	    <= opc == OPC_SETI_HIGH && arg_ano == 15 ? {rval, rf[15][00+:16]} : opc == OPC_SETI_LOW && arg_ano == 15 ? {rf[15][16+:16], rval} : rf[15];


			// rc_wadr		<=  arg_ano == 1 ? rf[1]/4 : 
            //     arg_ano == 2 ? rf[2]/4 :
            //     arg_ano == 3 ? rf[3]/4 :
            //     arg_ano == 4 ? rf[4]/4 :
            //     arg_ano == 5 ? rf[5]/4 :
            //     arg_ano == 6 ? rf[6]/4 :
            //     arg_ano == 7 ? rf[7]/4 : rc_wadr ;
                
			// ra_radr		<=  arg_bno == 1 ? rf[1]/4 : 
            //     arg_bno == 2 ? rf[2]/4 :
            //     arg_bno == 3 ? rf[3]/4 :
            //     arg_bno == 4 ? rf[4]/4 :
            //     arg_bno == 5 ? rf[5]/4 :
            //     arg_bno == 6 ? rf[6]/4 :
            //     arg_bno == 7 ? rf[7]/4 : ra_radr ;

			// rb_radr		<=  arg_cno == 1 ? rf[1]/4 : 
            //     arg_cno == 2 ? rf[2]/4 :
            //     arg_cno == 3 ? rf[3]/4 :
            //     arg_cno == 4 ? rf[4]/4 :
            //     arg_cno == 5 ? rf[5]/4 :
            //     arg_cno == 6 ? rf[6]/4 :
            //     arg_cno == 7 ? rf[7]/4 : rb_radr ;

			dma_req		        <= opc == OPC_LOAD ? 1 : 0;
			dma_rwn		        <= opc == OPC_LOAD;
			dma_localAddr		<= opc == OPC_LOAD || opc == OPC_STORE ? 
                opc == OPC_LOAD ? //if
                arg_ano == 1 ? rf[1][15:0] : 
                arg_ano == 2 ? rf[2][15:0] :
                arg_ano == 3 ? rf[3][15:0] :
                arg_ano == 4 ? rf[4][15:0] :
                arg_ano == 5 ? rf[5][15:0] :
                arg_ano == 6 ? rf[6][15:0] :
                arg_ano == 7 ? rf[7][15:0] : dma_localAddr :
                //else
                arg_bno == 1 ? rf[1][15:0] : 
                arg_bno == 2 ? rf[2][15:0] :
                arg_bno == 3 ? rf[3][15:0] :
                arg_bno == 4 ? rf[4][15:0] :
                arg_bno == 5 ? rf[5][15:0] :
                arg_bno == 6 ? rf[6][15:0] :
                arg_bno == 7 ? rf[7][15:0] : dma_localAddr : dma_localAddr;

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

			dma_tansferLength	<= opc == OPC_LOAD || opc == OPC_STORE ? 
                arg_cno == 1 ? rf[1][15:0] : 
                arg_cno == 2 ? rf[2][15:0] :
                arg_cno == 3 ? rf[3][15:0] :
                arg_cno == 4 ? rf[4][15:0] :
                arg_cno == 5 ? rf[5][15:0] :
                arg_cno == 6 ? rf[6][15:0] :
                arg_cno == 7 ? rf[7][15:0] : dma_tansferLength : dma_tansferLength;

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
			// fpu_cnt		<= cnt;
            
            // sram_a_enb_reg  <= (opc >= OPC_NOP && opc <= OPC_SETI_HIGH) || (opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16);
            kernel_rden <= opc >= OPC_NOP && opc <= OPC_SETI_HIGH;
            operanda_rden <= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16;
            // sram_b_enb_reg  <= (opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16) || (opc == OPC_STORE);
            operandb_rden <= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16;
            resultc_rden <= opc == OPC_STORE;
			// lh_rden		<= opc >= OPC_NOP && opc <= OPC_MOVE || opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16;
			// lf_rden		<= opc == OPC_STORE;
            operanda_radr   <= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ?
                arg_bno == 1 ? rf[1]/4 : 
                arg_bno == 2 ? rf[2]/4 :
                arg_bno == 3 ? rf[3]/4 :
                arg_bno == 4 ? rf[4]/4 :
                arg_bno == 5 ? rf[5]/4 :
                arg_bno == 6 ? rf[6]/4 :
                arg_bno == 7 ? rf[7]/4 : operanda_radr : operanda_radr ;

            operandb_radr   <= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? 
                arg_cno == 1 ? rf[1]/4 : 
                arg_cno == 2 ? rf[2]/4 :
                arg_cno == 3 ? rf[3]/4 :
                arg_cno == 4 ? rf[4]/4 :
                arg_cno == 5 ? rf[5]/4 :
                arg_cno == 6 ? rf[6]/4 :
                arg_cno == 7 ? rf[7]/4 : operandb_radr : operandb_radr ;
			// lh_radr		<= opc >= OPC_VADD_BF16 && opc <= OPC_VDIV_BF16 ? rf[1] / 4 : lh_radr;

			resultc_radr	<= opc == OPC_STORE ? 
                arg_bno == 1 ? rf[1]/4 : 
                arg_bno == 2 ? rf[2]/4 :
                arg_bno == 3 ? rf[3]/4 :
                arg_bno == 4 ? rf[4]/4 :
                arg_bno == 5 ? rf[5]/4 :
                arg_bno == 6 ? rf[6]/4 :
                arg_bno == 7 ? rf[7]/4 : resultc_radr : resultc_radr;


			// lf_wadr		<= rb / 4;
		end

		S_LOAD_REQ:
		begin
			state		<= dma_ready ? S_LOAD_DATA : state;
			dma_req		<= dma_ready ? 0 : 1;
            // sram_a_enb_reg  <= 0;
            // sram_b_enb_reg  <= 0;
            kernel_rden     <= 0;
            operanda_rden   <= 0;
            operandb_rden   <= 0;
            resultc_rden    <= 0;
		end

		S_LOAD_DATA:
		begin
			state		<= dma_ack && scnt == dma_tansferLength - 1 ? S_OPC_READ : state;
			scnt		<= dma_ack ? (scnt == dma_tansferLength - 1 ? 0 : scnt + 1) : scnt;
       
            // sram_a_wea_reg  <= dma_localAddr[12] == 1'b0;
            // sram_b_wea_reg  <= dma_localAddr[12] == 1'b1;

            // sram_a_ena  <= dma_localAddr[12] == 1'b0;
            // sram_b_ena  <= dma_localAddr[12] == 1'b1;

            // sram_a_addra_reg    <= sram_a_wea_reg && !dma_localAddr[12]? sram_a_addra_reg + 4 : dma_localAddr[11:0];
            // sram_b_addra_reg    <= sram_b_wea_reg && dma_localAddr[12]? sram_a_addra_reg + 4 : dma_localAddr[11:0];
            kernel_wren     <= dma_localAddr[12:11] == 2'b00;
            kernel_wadr     <= kernel_wren ? kernel_wadr + 4 : dma_localAddr[15:0];
            operanda_wren   <= dma_localAddr[12:11] == 2'b01;
            operanda_wadr   <= operanda_wren ? operanda_wadr + 4 : dma_localAddr[15:0];
            operandb_wren   <= dma_localAddr[12:11] == 2'b10;
            operandb_wadr   <= operandb_wren ? operandb_wadr + 4 : dma_localAddr[15:0];
            resultc_wren    <= dma_localAddr[12:11] == 2'b11;
            resultc_wadr    <= resultc_wren ? resultc_wadr + 4 : dma_localAddr[15:0];

			// lf_wren		<= dma_ack;
			// lf_wadr		<= lf_wren ? lf_wadr + 2 : lf_wadr;
            sram_a_dina_reg <= dma_readData;
            sram_b_dina_reg <= dma_readData;
			// lf_wdat		<= dma_readData;
			kernel_rden		<= dma_ack && scnt == dma_tansferLength - 1;
		end

		S_STORE_PRE:
		begin
			state		<= scnt == 3 ? S_STORE_REQ : state;
			scnt		<= scnt == 3 ? 0 : scnt + 1;
			dma_req		<= scnt == 3;
			// lf_rden		<= scnt < 2;
            resultc_rden    <= scnt < 2;
            resultc_radr    <= resultc_radr ? resultc_radr + 4 : resultc_radr;
			// lf_radr		<= lf_rden ? lf_radr + 2 : lf_radr;
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
			resultc_rden    <= dma_ack && scnt != dma_tansferLength - 1;
			resultc_radr	<= dma_ack && scnt == dma_tansferLength - 1 ? 0 : resultc_rden ? resultc_radr + 2 : resultc_radr;

			kernel_rden		<= dma_ack && scnt == dma_tansferLength - 1;
			// kernel_radr		<= dma_ack && scnt == dma_tansferLength - 1 ? opc_radr : lh_radr;
		end

        S_BF16:
        begin
			state		<= !opc_div || bf16_ir ? S_FOP : state;

            bf16_a  <= sram_a_doutb;
            bf16_b  <= sram_b_doutb;

            resultc_wadr   <= arg_ano == 1 ? rf[1]/4 : 
                arg_ano == 2 ? rf[2]/4 :
                arg_ano == 3 ? rf[3]/4 :
                arg_ano == 4 ? rf[4]/4 :
                arg_ano == 5 ? rf[5]/4 :
                arg_ano == 6 ? rf[6]/4 :
                arg_ano == 7 ? rf[7]/4 : resultc_wadr;
                
            operanda_rden   <= fpu_cnt > 1;
            operandb_rden   <= fpu_cnt > 1;
            operanda_radr   <= !opc_div || bf16_ir ? operanda_radr + 1 : operanda_radr;
            operandb_radr   <= !opc_div || bf16_ir ? operandb_radr + 1 : operandb_radr;

			// lh_rden		<= fpu_cnt > 1;
			// lh_radr		<= ra_radr;
            // resultc_wren    <= bf16_ylat
            // sram_b_dina_reg    <= bf16_ylat ? bf16_y : sram_b_dina_reg;
			// lh_wren		<= bf16_ylat;
			// lh_wdat		<= bf16_ylat ? bf16_y : lh_wdat;
			// rb_radr		<= !opc_div || bf16_ir ? rb_radr + 1 : rb_radr;
			// bf16_a		<= bf16_alat ? lh_rdat : bf16_a;
			// bf16_alat	<= 0;
			// bf16_blat	<= 1;
			// bf16_ylat	<= 0;
        
        end

		S_FOP:
		begin
			state       <= opc_div ? (bf16_ov ? (fpu_cnt == 1 ? S_FIN : state) : state) : (fpu_cnt == 1 ? S_FIN : state);
            // state		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? S_FIN : S_FOP) : state;
			// scnt		<= (opc_div ? bf16_ov : scnt == 1) ? 0 : scnt + 1;
			// fpu_cnt		<= (opc_div ? bf16_ov : scnt == 1) ? fpu_cnt - 1 : fpu_cnt;
            fpu_cnt     <= opc_div ? (bf16_ov ? fpu_cnt - 1 : fpu_cnt) : fpu_cnt - 1;
            
			// lh_rden		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
            operanda_rden   <= opc_div ? (bf16_ov ? (fpu_cnt == 1 ? 0 : 1) : 0) : (fpu_cnt == 1 ? 0 : 1);
            operandb_rden   <= opc_div ? (bf16_ov ? (fpu_cnt == 1 ? 0 : 1) : 0) : (fpu_cnt == 1 ? 0 : 1);
			// operanda_rden		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 1;
			// operandb_rden		<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 1;
            
			// lh_wadr		<= lh_wren ? lh_wadr + 1 : lh_wadr;
			resultc_wadr	<= (opc_div ? bf16_ov : 1) ? resultc_wadr + 1 : resultc_wadr;
			// lh_radr		<= (opc_div ? bf16_ov : scnt == 1) ? rb_radr : lh_radr;

            resultc_wren    <= opc_div ? (bf16_ov) : 1;
            // resultc_wren    <= (opc_div ? bf16_ov : scnt == 1) ? 
			// lh_wren		<= 0;
            operanda_radr   <= opc_div ? (bf16_ov ? operanda_radr + 1 : operanda_radr) : operanda_radr + 1;
            operandb_radr   <= opc_div ? (bf16_ov ? operanda_radr + 1 : operanda_radr) : operanda_radr + 1;
            // operanda_radr   <= !opc_div || bf16_ir ? operanda_radr + 1 : operanda_radr;
            // operandb_radr   <= !opc_div || bf16_ir ? operandb_radr + 1 : operandb_radr;

			// ra_radr		<= (opc_div ? bf16_ov : scnt == 1) ? ra_radr + 1 : ra_radr;
			bf16_iv		<= opc_div && scnt == 1;
			bf16_or		<= 1;
			// bf16_alat	<= (opc_div ? bf16_ov : scnt == 1) ? (fpu_cnt == 1 ? 0 : 1) : 0;
			// bf16_blat	<= 0;
			// bf16_b		<= bf16_blat ? lh_rdat : bf16_b;
			// bf16_ylat	<= (opc_div ? bf16_ov : scnt == 1);

            // resultc_wren    <= (opc_div ? bf16_ov : 1)
            sram_b_dina_reg    <= (opc_div ? bf16_ov : 1) ? bf16_y : sram_b_dina_reg;
		end

		S_FIN:
		begin
			state		<= S_OPC_READ;
			// bf16_ylat	<= 0;
			// lh_wren		<= bf16_ylat;
            resultc_wren    <= (opc_div ? bf16_ov : 1);
			// lh_wdat		<= bf16_ylat ? bf16_y : lh_wdat;
            sram_b_dina_reg    <= (opc_div ? bf16_ov : 1) ? bf16_y : sram_b_dina_reg;
			// lh_rden		<= 1;
            kernel_rden <= 1;
			// lh_radr		<= opc_radr;
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
always @(negedge rstn or posedge clk) 
    qi <= !rstn ? 0 : (state == S_STORE_PRE || state == S_STORE_DATA) && resultc_rden;

wire		qo		= state == S_STORE_DATA && dma_ack; 
reg	[1:0]	qc;
always @(negedge rstn or posedge clk) 
    qc <= !rstn ? 0 : (state == S_STORE_PRE && !scnt ? 0 : (qi && !qo ? qc + 1 : (!qi && qo ? qc - 1 : qc)));
reg [3*64-1:0]	q;
always @(negedge rstn or posedge clk) 
    q <= !rstn ? 0 : qi && !qo ? (qc == 0 ? {sram_b_doutb, q[0+:2*64]} : qc == 1 ? {q[2*64+:64], sram_b_doutb, q[0+:64]} : {q[1*64+:2*64], sram_b_doutb}) :
        !qi &&  qo ? (qc == 1 ? {3*64'h0} : {q[0+:2*64], 64'h0}) : 
        qi &&  qo ? (qc == 1 ? {sram_b_doutb, 64'h0, 64'h0} : qc == 2 ? {q[1*64+:64], sram_b_doutb, 64'h0} : {q[0+:2*64], sram_b_doutb}) : q;
assign		dma_writeData	= q[2*64+:64];

////----| sram 32bit write part |-------------------------------------------------
//reg	[31:0]	lh_wlsb;
//always @(negedge rstn or posedge clk) lh_wlsb <= !rstn ? 0 : lh_wren ? lh_wdat : lh_wlsb;

////----| sram 32bit read part |--------------------------------------------------
//reg		lh_rden_dly;
//always @(negedge rstn or posedge clk) 
//    lh_rden_dly <= !rstn ? 0 : lh_rden;
//reg		lh_radr0;
//always @(negedge rstn or posedge clk) 
//    lh_radr0 <= !rstn ? 0 : lh_rden ? lh_radr[0] : lh_radr0;
//reg	[31:0]	lh_rmsb;
//always @(negedge rstn or posedge clk) 
//    lh_rmsb <= !rstn ? 0 : lh_rden_dly ? sram_doutb[32+:32] : lh_rmsb;
////assign		lh_rdat		= !lh_radr0 ? sram_doutb[0+:32] : lh_rmsb;
//assign		lh_rdat		= !lh_radr0 ? sram_doutb[0+:32] : sram_doutb[32+:32];


//----| sram signal mapping |---------------------------------------------------
assign		sram_a_ena	= kernel_wren | operanda_wren;
assign		sram_a_wea	= kernel_wren | operanda_wren;
assign		sram_a_addra	= kernel_wren ? kernel_wadr / 2 : operanda_wadr / 2;
assign		sram_a_dina	= sram_a_dina_reg;
assign		sram_a_enb	= kernel_rden | operanda_rden;
assign		sram_a_addrb	= kernel_rden ? kernel_radr / 2 : operanda_radr / 2;

assign		sram_b_ena	= operandb_wren | resultc_wren;
assign		sram_b_wea	= operandb_wren | resultc_wren;
assign		sram_b_addra	= operandb_wren ? operandb_wadr / 2 : resultc_wadr / 2;
assign		sram_b_dina	= sram_b_dina_reg;
assign		sram_b_enb	= operandb_rden | resultc_rden;
assign		sram_b_addrb	= operandb_rden ? operandb_radr / 2 : resultc_radr / 2;

//----| output mapping |--------------------------------------------------------
assign		rocc_if_busy		= state != S_IDLE;
    
endmodule