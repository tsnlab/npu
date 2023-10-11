
////////////////////////////////////////////////////////////////////////////////
// Company: TSNLAB
// Engineer: junhyuk
//
// Create Date: 2023-10-10
// Design Name: TSN-NPU
// Module Name: tsn_dgcl
// Tool versions: vivado 2018.3
// Description:
//    <Description here>
// Revision:
//    0.01 junhyuk - Create
////////////////////////////////////////////////////////////////////////////////
			
module tsn_dgcl(
	input wire gemmini_clk,
	input wire fpu_clk,
	input wire reset,
	
	input wire [39:0] RCC_DRAM_ADDR,
	input wire [15:0] RCC_DPRAM_ADDR,
	input wire [15:0] RCC_LENGTH,
	output wire RCC_READY,
	input wire RCC_VALID,
	
	output wire [15:0] RCD_DPRAM_ADDR,
	output wire [127:0] RCD_READ_DATA,
	output wire [15:0] RCD_LENGTH,
	input wire RCD_READY,
	output wire RCD_VALID,
	
	input wire [39:0] WCC_DRAM_ADDR,
	input wire [15:0] WCC_DPRAM_ADDR,
	input wire [15:0] WCC_LENGTH,
	input wire [127:0] WCC_WRITE_DATA,
	output wire WCC_READY,
	input wire WCC_VALID,
	
	output wire DMA_REQ_A,
	input wire DMA_RESP_A,

	output wire DMA_WRITE_VALID_A,
	output wire [127:0] DMA_WRITE_DATA_A,
	input wire DMA_WRITE_READY_A,
	
	input wire DMA_READ_VALID_A,
	input wire [127:0] DMA_READ_DATA_A,
	output wire DMA_READ_READY_A,
	
	output wire DMA_REQ_B,
	input wire DMA_RESP_B,

	output wire DMA_WRITE_VALID_B,
	output wire [127:0] DMA_WRITE_DATA_B,
	input wire DMA_WRITE_READY_B,
	
	input wire DMA_READ_VALID_B,
	input wire [127:0] DMA_READ_DATA_B,
	output wire DMA_READ_READY_B,

	output wire DMA_REQ_C,
	input wire DMA_RESP_C,

	output wire DMA_WRITE_VALID_C,
	output wire [127:0] DMA_WRITE_DATA_C,
	input wire DMA_WRITE_READY_C,
	
	input wire DMA_READ_VALID_C,
	input wire [127:0] DMA_READ_DATA_C,
	output wire DMA_READ_READY_C,
	
	output wire DMA_REQ_D,
	input wire DMA_RESP_D,

	output wire DMA_WRITE_VALID_D,
	output wire [127:0] DMA_WRITE_DATA_D,
	input wire DMA_WRITE_READY_D,
	
	input wire DMA_READ_VALID_D,
	input wire [127:0] DMA_READ_DATA_D,
	output wire DMA_READ_READY_D
);


reg rcc_ready_r;
reg [39:0] rcc_dram_cnt=0;
reg [15:0] rcc_dpram_cnt=0;
reg [15:0] rcc_lengh_cnt=0;
	
always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		rcc_ready_r <= 1'b0;
		rcc_dram_cnt <= 0;
		rcc_dpram_cnt <= 0;
		rcc_lengh_cnt <= 0;
	end
	else begin
		if(RCC_VALID) begin
			rcc_ready_r <= 1'b1;
			rcc_dram_cnt <= rcc_dram_cnt + RCC_DRAM_ADDR;
			rcc_dpram_cnt <= rcc_dpram_cnt + RCC_DPRAM_ADDR;
			rcc_lengh_cnt <= rcc_lengh_cnt + RCC_LENGTH;
		end
		else begin
			rcc_ready_r <= 1'b0;
		end
	end
end
	
assign RCC_READY = rcc_ready_r;

reg wcc_ready_r;
reg [39:0] wcc_dram_cnt=0;
reg [15:0] wcc_dpram_cnt=0;
reg [15:0] wcc_lengh_cnt=0;
always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		wcc_ready_r <= 1'b0;
		wcc_dram_cnt <= 0;
		wcc_dpram_cnt <= 0;
		wcc_lengh_cnt <= 0;
	end
	else begin
		if(WCC_VALID) begin
			wcc_ready_r <= 1'b1;
			wcc_dram_cnt <= wcc_dram_cnt + WCC_DRAM_ADDR;
			wcc_dpram_cnt <= wcc_dpram_cnt + WCC_DPRAM_ADDR;
			wcc_lengh_cnt <= wcc_lengh_cnt + WCC_LENGTH;
		end
		else begin
			wcc_ready_r <= 1'b0;
		end
	end
end

assign WCC_READY = wcc_ready_r;

reg rcd_valid_r;
reg [39:0] rcd_dpram_addr_r=0;
reg [15:0] rcd_read_data_r=0;
reg [15:0] rcd_length_r=0;

always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		rcd_valid_r <= 0;
		rcd_dpram_addr_r <= 0;
		rcd_read_data_r <= 0;
		rcd_length_r <= 0;

	end
	else begin
		if(RCD_READY) begin
			rcd_dpram_addr_r <= rcd_dpram_addr_r + 1;
			rcd_read_data_r <= rcd_read_data_r + 1;
			rcd_length_r <= rcd_length_r + 1;
			rcd_valid_r <= 1;
		end
		else begin
			rcd_valid_r <= 0;
		
		end

	end
end


assign RCD_DPRAM_ADDR = rcd_dpram_addr_r;
assign RCD_READ_DATA = rcd_read_data_r;
assign RCD_LENGTH = rcd_length_r;
assign RCD_VALID = rcd_valid_r;

reg[127:0] cnt;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		cnt <= 0;
	end
	else begin
		cnt <= cnt + 1;
	end
end

//*****************************************************
assign DMA_WRITE_VALID_A = DMA_WRITE_READY_A ? 1'd1 : 1'd0;
assign DMA_WRITE_DATA_A = DMA_WRITE_READY_A ? cnt : 128'd0;

reg[127:0] rd_cnt_a=0;
reg dma_read_ready_a_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_a_r <= 0;
	end
	else begin
		if(DMA_READ_VALID_A) begin
			rd_cnt_a <= rd_cnt_a + DMA_READ_DATA_A;
			dma_read_ready_a_r <= 1;
		end
		else begin
			dma_read_ready_a_r <= 0;
		
		end
	end
end

assign DMA_READ_READY_A = dma_read_ready_a_r;

assign DMA_REQ_A = DMA_RESP_A;

//*****************************************************

assign DMA_WRITE_VALID_B = DMA_WRITE_READY_B ? 1'd1 : 1'd0;
assign DMA_WRITE_DATA_B = DMA_WRITE_READY_B ? cnt : 128'd0;

reg[127:0] rd_cnt_b=0;
reg dma_read_ready_b_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_b_r <= 0;
	end
	else begin
		if(DMA_READ_VALID_B) begin
			rd_cnt_b <= rd_cnt_b + DMA_READ_DATA_B;
			dma_read_ready_b_r <= 1;
		end
		else begin
			dma_read_ready_b_r <= 0;
		
		end
	end
end

assign DMA_READ_READY_B = dma_read_ready_b_r;

assign DMA_REQ_B = DMA_RESP_B;

//*****************************************************

assign DMA_WRITE_VALID_C = DMA_WRITE_READY_C ? 1'd1 : 1'd0;
assign DMA_WRITE_DATA_C = DMA_WRITE_READY_C ? cnt : 128'd0;

reg[127:0] rd_cnt_c=0;
reg dma_read_ready_c_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_c_r <= 0;
	end
	else begin
		if(DMA_READ_VALID_C) begin
			rd_cnt_c <= rd_cnt_c + DMA_READ_DATA_C;
			dma_read_ready_c_r <= 1;
		end
		else begin
			dma_read_ready_c_r <= 0;
		
		end
	end
end

assign DMA_READ_READY_C = dma_read_ready_c_r;

assign DMA_REQ_C = DMA_RESP_C;

//*****************************************************

assign DMA_WRITE_VALID_D = DMA_WRITE_READY_D ? 1'd1 : 1'd0;
assign DMA_WRITE_DATA_D = DMA_WRITE_READY_D ? cnt : 128'd0;

reg[127:0] rd_cnt_d=0;
reg dma_read_ready_d_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_d_r <= 0;
	end
	else begin
		if(DMA_READ_VALID_D) begin
			rd_cnt_d <= rd_cnt_d + DMA_READ_DATA_D;
			dma_read_ready_d_r <= 1;
		end
		else begin
			dma_read_ready_d_r <= 0;
		
		end
	end
end

assign DMA_READ_READY_D = dma_read_ready_d_r;

assign DMA_REQ_D = DMA_RESP_D;

//*****************************************************

endmodule
