
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
	
	input wire [39:0] rcc_dram_addr,
	input wire [15:0] rcc_dpram_addr,
	input wire [15:0] rcc_length,
	output wire rcc_ready,
	input wire rcc_valid,
	
	output wire [15:0] rcd_dpram_addr,
	output wire [127:0] rcd_read_data,
	output wire [15:0] rcd_length,
	input wire rcd_ready,
	output wire rcd_valid,
	
	input wire [39:0] wcc_dram_addr,
	input wire [15:0] wcc_dpram_addr,
	input wire [15:0] wcc_length,
	input wire [127:0] wcc_write_data,
	output wire wcc_ready,
	input wire wcc_valid,
	
	output wire dma_req_a,
	input wire dma_resp_a,

	output wire dma_write_valid_a,
	output wire [127:0] dma_write_data_a,
	input wire dma_write_ready_a,
	
	input wire dma_read_valid_a,
	input wire [127:0] dma_read_data_a,
	output wire dma_read_ready_a,
	
	output wire dma_req_b,
	input wire dma_resp_b,

	output wire dma_write_valid_b,
	output wire [127:0] dma_write_data_b,
	input wire dma_write_ready_b,
	
	input wire dma_read_valid_b,
	input wire [127:0] dma_read_data_b,
	output wire dma_read_ready_b,

	output wire dma_req_c,
	input wire dma_resp_c,

	output wire dma_write_valid_c,
	output wire [127:0] dma_write_data_c,
	input wire dma_write_ready_c,
	
	input wire dma_read_valid_c,
	input wire [127:0] dma_read_data_c,
	output wire dma_read_ready_c,
	
	output wire dma_req_d,
	input wire dma_resp_d,

	output wire dma_write_valid_d,
	output wire [127:0] dma_write_data_d,
	input wire dma_write_ready_d,
	
	input wire dma_read_valid_d,
	input wire [127:0] dma_read_data_d,
	output wire dma_read_ready_d
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
		if(rcc_valid) begin
			rcc_ready_r <= 1'b1;
			rcc_dram_cnt <= rcc_dram_cnt + rcc_dram_addr;
			rcc_dpram_cnt <= rcc_dpram_cnt + rcc_dpram_addr;
			rcc_lengh_cnt <= rcc_lengh_cnt + rcc_length;
		end
		else begin
			rcc_ready_r <= 1'b0;
		end
	end
end
	
assign rcc_ready = rcc_ready_r;

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
		if(wcc_valid) begin
			wcc_ready_r <= 1'b1;
			wcc_dram_cnt <= wcc_dram_cnt + wcc_dram_addr;
			wcc_dpram_cnt <= wcc_dpram_cnt + wcc_dpram_addr;
			wcc_lengh_cnt <= wcc_lengh_cnt + wcc_length;
		end
		else begin
			wcc_ready_r <= 1'b0;
		end
	end
end

assign wcc_ready = wcc_ready_r;

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
		if(rcd_ready) begin
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


assign rcd_dpram_addr = rcd_dpram_addr_r;
assign rcd_read_data = rcd_read_data_r;
assign rcd_length = rcd_length_r;
assign rcd_valid = rcd_valid_r;

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
assign dma_write_valid_a = dma_write_ready_a ? 1'd1 : 1'd0;
assign dma_write_data_a = dma_write_ready_a ? cnt : 128'd0;

reg[127:0] rd_cnt_a=0;
reg dma_read_ready_a_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_a_r <= 0;
	end
	else begin
		if(dma_read_valid_a) begin
			rd_cnt_a <= rd_cnt_a + dma_read_data_a;
			dma_read_ready_a_r <= 1;
		end
		else begin
			dma_read_ready_a_r <= 0;
		
		end
	end
end

assign dma_read_ready_a = dma_read_ready_a_r;

assign dma_req_a = dma_resp_a;

//*****************************************************

assign dma_write_valid_b = dma_write_ready_b ? 1'd1 : 1'd0;
assign dma_write_data_b = dma_write_ready_b ? cnt : 128'd0;

reg[127:0] rd_cnt_b=0;
reg dma_read_ready_b_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_b_r <= 0;
	end
	else begin
		if(dma_read_valid_b) begin
			rd_cnt_b <= rd_cnt_b + dma_read_data_b;
			dma_read_ready_b_r <= 1;
		end
		else begin
			dma_read_ready_b_r <= 0;
		
		end
	end
end

assign dma_read_ready_b = dma_read_ready_b_r;

assign dma_req_b = dma_resp_b;

//*****************************************************

assign dma_write_valid_c = dma_write_ready_c ? 1'd1 : 1'd0;
assign dma_write_data_c = dma_write_ready_c ? cnt : 128'd0;

reg[127:0] rd_cnt_c=0;
reg dma_read_ready_c_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_c_r <= 0;
	end
	else begin
		if(dma_read_valid_c) begin
			rd_cnt_c <= rd_cnt_c + dma_read_data_c;
			dma_read_ready_c_r <= 1;
		end
		else begin
			dma_read_ready_c_r <= 0;
		
		end
	end
end

assign dma_read_ready_c = dma_read_ready_c_r;

assign dma_req_c = dma_resp_c;

//*****************************************************

assign dma_write_valid_d = dma_write_ready_d ? 1'd1 : 1'd0;
assign dma_write_data_d = dma_write_ready_d ? cnt : 128'd0;

reg[127:0] rd_cnt_d=0;
reg dma_read_ready_d_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_ready_d_r <= 0;
	end
	else begin
		if(dma_read_valid_d) begin
			rd_cnt_d <= rd_cnt_d + dma_read_data_d;
			dma_read_ready_d_r <= 1;
		end
		else begin
			dma_read_ready_d_r <= 0;
		
		end
	end
end

assign dma_read_ready_d = dma_read_ready_d_r;

assign dma_req_d = dma_resp_d;

//*****************************************************

endmodule
