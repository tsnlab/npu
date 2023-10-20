
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
			
module TSN_DGCL(
	input wire gemmini_clk,
	input wire fpu_clk,
	input wire reset,
	
	output wire [39:0] rcc_dram_addr,
	output wire [15:0] rcc_dpram_addr,
	output wire [15:0] rcc_length,
	input wire rcc_ready,
	output wire rcc_valid,
	
	input wire [15:0] rcd_dpram_addr,
	input wire [127:0] rcd_read_data,
	input wire [15:0] rcd_length,
	output wire rcd_ready,
	input wire rcd_valid,
	
	output wire [39:0] wcc_dram_addr,
	output wire [15:0] wcc_dpram_addr,
	output wire [15:0] wcc_length,
	output wire [127:0] wcc_write_data,
	input wire wcc_ready,
	output wire wcc_valid,
	
	input wire dma_req_a,
	output wire dma_resp_a,

	input wire dma_write_valid_a,
	input wire [127:0] dma_write_data_a,
	output wire dma_write_ready_a,
	
	output wire dma_read_valid_a,
	output wire [127:0] dma_read_data_a,
	input wire dma_read_ready_a,
	
	input wire dma_req_b,
	output wire dma_resp_b,

	input wire dma_write_valid_b,
	input wire [127:0] dma_write_data_b,
	output wire dma_write_ready_b,
	
	output wire dma_read_valid_b,
	output wire [127:0] dma_read_data_b,
	input wire dma_read_ready_b,

	input wire dma_req_c,
	output wire dma_resp_c,

	input wire dma_write_valid_c,
	input wire [127:0] dma_write_data_c,
	output wire dma_write_ready_c,
	
	output wire dma_read_valid_c,
	output wire [127:0] dma_read_data_c,
	input wire dma_read_ready_c,
	
	input wire dma_req_d,
	output wire dma_resp_d,

	input wire dma_write_valid_d,
	input wire [127:0] dma_write_data_d,
	output wire dma_write_ready_d,
	
	output wire dma_read_valid_d,
	output wire [127:0] dma_read_data_d,
	input wire dma_read_ready_d
);


reg rcd_ready_r;
reg [39:0] rcd_dram_cnt=0;
reg [15:0] rcd_dpram_cnt=0;
reg [15:0] rcd_lengh_cnt=0;
	
always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		rcd_ready_r <= 1'b0;
		// rcd_dram_cnt <= 0;
		rcd_dpram_cnt <= 0;
		rcd_lengh_cnt <= 0;
	end
	else begin
		if(rcd_valid) begin
			rcd_ready_r <= 1'b1;
			// rcd_dram_cnt <= rcd_dram_cnt + rcd_dram_addr;
			rcd_dpram_cnt <= rcd_dpram_cnt + rcd_dpram_addr;
			rcd_lengh_cnt <= rcd_lengh_cnt + rcd_length;
		end
		else begin
			rcd_ready_r <= 1'b0;
		end
	end
end
	
assign rcd_ready = rcd_ready_r;

//*****************************************************

reg wcc_valid_r;
reg [39:0] wcc_dpram_addr_r=0;
reg [15:0] wcc_read_data_r=0;
reg [15:0] wcc_length_r=0;

always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		wcc_valid_r <= 0;
		wcc_dpram_addr_r <= 0;
		wcc_read_data_r <= 0;
		wcc_length_r <= 0;

	end
	else begin
		if(wcc_ready) begin
			wcc_dpram_addr_r <= wcc_dpram_addr_r + 1;
			wcc_read_data_r <= wcc_read_data_r + 1;
			wcc_length_r <= wcc_length_r + 1;
			wcc_valid_r <= 1;
		end
		else begin
			// wcc_dpram_cntcc_valid_r <= 0;
		
		end

	end
end

assign wcc_dpram_addr = wcc_dpram_addr_r;
assign wcc_read_data = wcc_read_data_r;
assign wcc_length = wcc_length_r;
assign wcc_valid = wcc_valid_r;
assign wcc_write_data = cnt;

//*****************************************************
reg rcc_valid_r;
reg [39:0] rcc_dpram_addr_r=0;
reg [15:0] rcc_read_data_r=0;
reg [15:0] rcc_length_r=0;

always@(posedge gemmini_clk or posedge reset) begin
	if(reset) begin
		rcc_valid_r <= 0;
		rcc_dpram_addr_r <= 0;
		rcc_read_data_r <= 0;
		rcc_length_r <= 0;

	end
	else begin
		if(rcc_ready) begin
			rcc_dpram_addr_r <= rcc_dpram_addr_r + 1;
			rcc_read_data_r <= rcc_read_data_r + 1;
			rcc_length_r <= rcc_length_r + 1;
			rcc_valid_r <= 1;
		end
		else begin
			rcc_valid_r <= 0;
		
		end

	end
end


assign rcc_dpram_addr = rcc_dpram_addr_r;
assign rcc_read_data = rcc_read_data_r;
assign rcc_length = rcc_length_r;
assign rcc_valid = rcc_valid_r;

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
assign dma_read_valid_a = dma_read_ready_a ? 1'd1 : 1'd0;
assign dma_read_data_a = dma_read_ready_a ? cnt : 128'd0;

reg[127:0] rd_cnt_a=0;
reg dma_write_ready_a_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_write_ready_a_r <= 0;
	end
	else begin
		if(dma_write_valid_a) begin
			rd_cnt_a <= rd_cnt_a + dma_write_data_a;
			dma_write_ready_a_r <= 1;
		end
		else begin
			dma_write_ready_a_r <= 0;
		
		end
	end
end

assign dma_write_ready_a = dma_write_ready_a_r;

assign dma_resp_a = dma_req_a;

//*****************************************************
assign dma_read_valid_b = dma_read_ready_b ? 1'd1 : 1'd0;
assign dma_read_data_b = dma_read_ready_b ? cnt : 128'd0;

reg[127:0] rd_cnt_b=0;
reg dma_write_ready_b_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_write_ready_b_r <= 0;
	end
	else begin
		if(dma_write_valid_b) begin
			rd_cnt_b <= rd_cnt_b + dma_write_data_b;
			dma_write_ready_b_r <= 1;
		end
		else begin
			dma_write_ready_b_r <= 0;
		
		end
	end
end

assign dma_write_ready_b = dma_write_ready_b_r;

assign dma_resp_b = dma_req_b;

//*****************************************************
assign dma_read_valid_c = dma_read_ready_c ? 1'd1 : 1'd0;
assign dma_read_data_c = dma_read_ready_c ? cnt : 128'd0;

reg[127:0] rd_cnt_c=0;
reg dma_write_ready_c_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_write_ready_c_r <= 0;
	end
	else begin
		if(dma_write_valid_c) begin
			rd_cnt_c <= rd_cnt_c + dma_write_data_c;
			dma_write_ready_c_r <= 1;
		end
		else begin
			dma_write_ready_c_r <= 0;
		
		end
	end
end

assign dma_write_ready_c = dma_write_ready_c_r;

assign dma_resp_c = dma_req_c;

//*****************************************************
assign dma_read_valid_d = dma_read_ready_d ? 1'd1 : 1'd0;
assign dma_read_data_d = dma_read_ready_d ? cnt : 128'd0;

reg[127:0] rd_cnt_d=0;
reg dma_write_ready_d_r = 0;
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_write_ready_d_r <= 0;
	end
	else begin
		if(dma_write_valid_d) begin
			rd_cnt_d <= rd_cnt_d + dma_write_data_d;
			dma_write_ready_d_r <= 1;
		end
		else begin
			dma_write_ready_d_r <= 0;
		
		end
	end
end

assign dma_write_ready_d = dma_write_ready_d_r;

assign dma_resp_d = dma_req_d;

//*****************************************************
endmodule
