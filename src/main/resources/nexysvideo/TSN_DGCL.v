
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
	
	// Read Control Command Signal
	output wire [39:0] rcc_dram_addr,
	output wire [15:0] rcc_dpram_addr,
	output wire [15:0] rcc_length,
	input wire rcc_ready,
	output wire rcc_valid,

	// Read Control Data Signal
	input wire [15:0] rcd_dpram_addr,
	input wire [127:0] rcd_read_data,
	input wire [15:0] rcd_length,
	output wire rcd_ready,
	input wire rcd_valid,
	
	// Write Control Command Signal
	output wire [39:0] wcc_dram_addr,
	output wire [15:0] wcc_dpram_addr,
	output wire [15:0] wcc_length,
	output wire [127:0] wcc_write_data,
	input wire wcc_ready,
	output wire wcc_valid,
	
	// TSN-DMA Signal A
	input wire dma_req_a,
	output wire dma_resp_a,

	input wire dma_write_valid_a,
	input wire [127:0] dma_write_data_a,
	output wire dma_write_ready_a,
	
	output wire dma_read_valid_a,
	output wire [127:0] dma_read_data_a,
	input wire dma_read_ready_a,
	
	// TSN-DMA Signal B
	input wire dma_req_b,
	output wire dma_resp_b,

	input wire dma_write_valid_b,
	input wire [127:0] dma_write_data_b,
	output wire dma_write_ready_b,
	
	output wire dma_read_valid_b,
	output wire [127:0] dma_read_data_b,
	input wire dma_read_ready_b,

	// TSN-DMA Signal C
	input wire dma_req_c,
	output wire dma_resp_c,

	input wire dma_write_valid_c,
	input wire [127:0] dma_write_data_c,
	output wire dma_write_ready_c,
	
	output wire dma_read_valid_c,
	output wire [127:0] dma_read_data_c,
	input wire dma_read_ready_c,
	
	// TSN-DMA Signal D
	input wire dma_req_d,
	output wire dma_resp_d,

	input wire dma_write_valid_d,
	input wire [127:0] dma_write_data_d,
	output wire dma_write_ready_d,
	
	output wire dma_read_valid_d,
	output wire [127:0] dma_read_data_d,
	input wire dma_read_ready_d
);

wire rcc_ff_wr_en;
wire [71:0] rcc_ff_wr_data;
wire rcc_ff_full;
wire [7:0] rcc_ff_wr_cnt;
wire rcc_ff_rd_en;
wire [71:0] rcc_ff_rd_data;
wire rcc_ff_empty;
wire [7:0] rcc_ff_rd_cnt;

wire rcd_ff_c_wr_en;
wire [31:0] rcd_ff_c_wr_data;
wire rcd_ff_c_full;
wire [8:0] rcd_ff_c_wr_cnt;
wire rcd_ff_c_rd_en;
wire [31:0] rcd_ff_c_rd_data;
wire rcd_ff_c_empty;
wire [8:0] rcd_ff_c_rd_cnt;

wire rcd_ff_d_wr_en;
wire [127:0] rcd_ff_d_wr_data;
wire rcd_ff_d_full;
wire [9:0] rcd_ff_d_wr_cnt;
wire rcd_ff_d_rd_en;
wire [127:0] rcd_ff_d_rd_data;
wire rcd_ff_d_empty;
wire [9:0] rcd_ff_d_rd_cnt;

localparam rcd_st = 2'd0;
localparam rcd_s0 = 2'd1;
localparam rcd_end = 2'd2;
reg [1:0] rcdcon = rcd_st;

reg rcd_cmd_valid_d ;

wire dma_read_ff_wr_en;
wire [127:0] dma_read_ff_d_wr_data;
wire dma_read_ff_d_full;
wire [9:0] dma_read_ff_d_wr_cnt;
wire dma_read_ff_d_rd_en;
wire [127:0] dma_read_ff_d_rd_data;
wire dma_read_ff_d_empty;
wire [9:0] dma_read_ff_d_rd_cnt;


localparam fwdc_st = 4'd0;
localparam fwdc_p0 = 4'd1;
localparam fwdc_p1 = 4'd2;
localparam fwdc_p2 = 4'd3;
localparam fwdc_p3 = 4'd4;
localparam fwdc_s0 = 4'd5;
localparam fwdc_s1 = 4'd6;
localparam fwdc_s2 = 4'd7;

reg[3:0] fwdcon = fwdc_st;

as72x512_ft rcc_fifo (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(fpu_clk),        // input wire wr_clk
	.wr_en			(rcc_ff_wr_en),         // input wire wr_en
	.din			(rcc_ff_wr_data),           // input wire [71 : 0] din
	.full			(rcc_ff_full),          // output wire full
	.wr_data_count	(rcc_ff_wr_cnt), // output wire [8 : 0] wr_data_count

	.rd_clk			(gemmini_clk),        // input wire rd_clk
	.rd_en			(rcc_ff_rd_en),         // input wire rd_en
	.dout			(rcc_ff_rd_data),          // output wire [71 : 0] dout
	.empty			(rcc_ff_empty),         // output wire empty
	.rd_data_count	(rcc_ff_rd_cnt)  // output wire [8 : 0] rd_data_count
);

assign rcc_ff_rd_en = (~rcc_ff_empty) && (rcc_ready);
assign rcc_valid = (~rcc_ff_empty) && (rcc_ready);
assign rcc_dram_addr = rcc_ff_rd_data[39:0];
assign rcc_dpram_addr = rcc_ff_rd_data[55:40];
assign rcc_length = rcc_ff_rd_data[71:56];

always@(posedge gemmini_clk or posedge reset) begin
	if(reset)begin
		rcd_cmd_valid_d <= 1'b0;
	end
	else begin
		rcd_cmd_valid_d <= rcd_valid&&(~cd_ff_c_full);
		rcd_ff_c_wr_data[15:0] <= rcd_dpram_addr;
		rcd_ff_c_wr_data[31:16] <= rcd_length;
	end
end

assign rcd_ff_c_wr_en = (rcd_cmd_valid_d);
assign rcd_ff_d_wr_en = (rcd_valid && (~rcd_ff_d_full));
assign rcd_ready = ((~rcd_ff_d_full) && (rcd_ff_c_full));
assign rcd_ff_c_rd_data = rcd_read_data;

as32x512_ft rcd_com_fifo (
	.rst			(reset),           // input wire rst

	.wr_clk			(gemmini_clk),        // input wire wr_clk
	.wr_en			(rcd_ff_c_wr_en),         // input wire wr_en
	.din			(rcd_ff_c_wr_data),           // input wire [31 : 0] din
	.full			(rcd_ff_c_full),          // output wire full
	.wr_data_count	(rcd_ff_c_wr_cnt), // output wire [8 : 0] wr_data_count

	.rd_clk			(fpu_clk),        // input wire rd_clk
	.rd_en			(rcd_ff_c_rd_en),         // input wire rd_en
	.dout			(rcd_ff_c_rd_data),          // output wire [31 : 0] dout
	.empty			(rcd_ff_c_empty),         // output wire empty
	.rd_data_count	(rcd_ff_c_rd_cnt)  // output wire [8 : 0] rd_data_count
);

as128x1024 rcd_data_fifo (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(gemmini_clk),        // input wire wr_clk
	.wr_en			(rcd_ff_d_wr_en),         // input wire wr_en
	.din			(rcd_ff_d_wr_data),           // input wire [127 : 0] din
	.full			(rcd_ff_d_full),          // output wire full
	.wr_data_count	(rcd_ff_d_wr_cnt), // output wire [9 : 0] wr_data_count
  
	.rd_clk			(fpu_clk),        // input wire rd_clk
	.rd_en			(rcd_ff_d_rd_en),         // input wire rd_en
	.dout			(rcd_ff_d_rd_data),          // output wire [127 : 0] dout
	.empty			(rcd_ff_d_empty),         // output wire empty
	.rd_data_count	(rcd_ff_d_rd_cnt)  // output wire [9 : 0] rd_data_count
);

reg[15:0] rcdcon_count;
reg[15:0] rcdcon_length;
reg dma_read_vld;
reg[127:0] dma_read_data;
reg[1:0] dma_mux;
wire dma_read_ready_mux;


always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		dma_read_vld <= 1'b0;
		rcd_ff_d_rd_en <= 1'b0;
		dma_read_data <= 128'd0;
		rcdcon_count<= 16'd0;
		dma_mux <= 2'b00;
		rcdcon <= rcd_st;

	end
	else begin
		case(rcdcon)
			rcd_st : begin
				if((~rcd_ff_c_empty)&&(~rcd_ff_d_empty)) begin
					rcd_ff_c_rd_en <= 1'b1;
					rcdcon_length <= rcd_ff_c_rd_data[31:16];

					dma_read_data[15:0] <= rcd_ff_c_rd_data[15:0];
					dma_read_data[55:0] <= 0;
					dma_read_data[71:56] <= rcd_ff_c_rd_data[31:16];
					dma_read_data[79:72] <= 8'h02;
					dma_read_data[127:80] <= 0;
					dma_mux <= rcd_ff_c_rd_data[15:14];

					dma_read_vld <= 1'b1;
					rcdcon <= rcd_s0;
				end
				else begin
					rcd_ff_c_rd_en <= 1'b0;
					rcdcon_length <= 0;
					dma_read_vld <= 1'b0;
					dma_read_data <= 0;
					rcdcon <= rcd_st;
				end
			end
			rcd_s0 : begin
				if(rcdcon_length > rcdcon_count)begin
					if(dma_read_ready_mux) begin
						dma_read_vld <= 1'b1;
						rcd_ff_d_rd_en <= 1'b1;
						dma_read_data <= rcd_ff_d_rd_data;
						rcdcon_count<= rcdcon_count + 1;
						rcdcon <= rcd_s0;

					end
					else begin
						dma_read_vld <= 1'b1;
						rcd_ff_d_rd_en <= 1'b0;
						dma_read_data <= rcd_ff_d_rd_data;
						rcdcon_count<= rcdcon_count;
						rcdcon <= rcd_s0;
					end
				end
				else begin
					dma_read_vld <= 1'b0;
					rcd_ff_d_rd_en <= 1'b0;
					dma_read_data <= 128'd0;
					rcdcon_count<= 16'd0;
					rcdcon <= rcd_s1;
				end
			end
			rcd_end : begin
				dma_read_vld <= 1'b0;
				rcd_ff_d_rd_en <= 1'b0;
				dma_read_data <= 128'd0;
				rcdcon_count<= 16'd0;
				rcdcon <= rcd_st;
			end
			default : begin
			end
		endcase

	end
end

assign dma_read_valid_a = (dma_mux==2'b00)? dma_read_vld : 1'b0;
assign dma_read_data_a = dma_read_data;
assign dma_read_ready_mux = (dma_mux==2'b00)? dma_read_ready_a : (dma_mux==2'b01)? dma_read_ready_b : (dma_mux==2'b10)? dma_read_ready_c : (dma_mux==2'b11)? dma_read_ready_d : 1'b0; 

reg[1:0] fpu_sel =2'd0;
reg fpu_resp;
reg fpu_write_rdy;
wire [127:0] fpu_write_data;
wire fpu_write_vld;
reg[15:0] fpu_write_length;
reg[15:0] fpu_write_cnt;


assign fpu_write_data = (fpu_sel == 2'd0)? dma_write_data_a : (fpu_sel == 2'd1)? dma_write_data_b : (fpu_sel == 2'd2)? dma_write_data_c : (fpu_sel == 2'd2)? dma_write_data_d : 128'd0;
assign fpu_write_vld = (fpu_sel == 2'd0)? dma_write_valid_a : (fpu_sel == 2'd1)? dma_write_valid_b : (fpu_sel == 2'd2)? dma_write_valid_c : (fpu_sel == 2'd2)? dma_write_valid_d : 1'b0;

always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		
	end
	else begin
		case(fwdcon) 
			fwdc_st : begin
				
			end
			fwdc_p0 : begin
				if(dma_req_a) begin fpu_sel <= 2'd0; end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
				end
			end
			fwdc_p1 : begin
				if(dma_req_b) begin
					fpu_sel <= 2'd1;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
				end
				
			end
			fwdc_p2 : begin
				if(dma_req_c) begin
					fpu_sel <= 2'd2;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
				end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
				end
			end
			fwdc_p3 : begin
				if(dma_req_d) begin
					fpu_sel <= 2'd3;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
				end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
				end
				
			end
			fwdc_s0 : begin
				fpu_resp <= 1'b1;
				if(fpu_write_vld&&fpu_write_rdy)begin
					fpu_write_length <= fpu_write_data[71:56];
					fwdcon <= fwdc_s1;
				end
				
			end
			fwdc_s1 : begin
				fpu_resp <= 1'b1;
				if(fpu_write_vld&&fpu_write_rdy)begin
					fpu_write_cnt <= fpu_write_cnt + 1;
					fwdcon <= fwdc_s2;
				end
			end
			fwdc_s2 : begin
				
			end
			fwdc_s3 : begin
				
			end
			default : begin
				
			end
		endcase
		
	end
end

assign dma_resp_a = (fpu_sel == 2'd0)?  fpu_resp : 1'b0;
assign dma_resp_b = (fpu_sel == 2'd1)?  fpu_resp : 1'b0;
assign dma_resp_c = (fpu_sel == 2'd2)?  fpu_resp : 1'b0;
assign dma_resp_d = (fpu_sel == 2'd3)?  fpu_resp : 1'b0;
assign dma_write_ready_a = (fpu_sel == 2'd0)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_b = (fpu_sel == 2'd1)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_c = (fpu_sel == 2'd2)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_d = (fpu_sel == 2'd3)?  fpu_write_rdy : 1'b0;

wire fpu_ff_wr_en;
wire [127:0] fpu_ff_wr_data;
wire fpu_ff_full;
wire [9:0] fpu_ff_wr_cnt;
wire fpu_ff_rd_en;
wire [127:0] fpu_ff_rd_data;
wire fpu_ff_empty;
wire [9:0] fpu_ff_rd_cnt;

assign fpu_ff_wr_en = fpu_write_vld &&(~fpu_ff_full)&&(fpu_resp);
assign fpu_ff_wr_data = fpu_write_data;
assign fpu_write_rdy = (~fpu_ff_full)&&(fpu_resp);

as128x1024 fpu_write_fifo (
	.rst			(rsresett),           // input wire rst
	
	.wr_clk			(fpu_clk),        // input wire wr_clk
	.wr_en			(fpu_ff_wr_en),         // input wire wr_en
	.din			(fpu_ff_wr_data),           // input wire [127 : 0] din
	.full			(fpu_ff_full),          // output wire full
	.wr_data_count	(fpu_ff_wr_cnt), // output wire [9 : 0] wr_data_count
  
	.rd_clk			(fpu_clk),        // input wire rd_clk
	.rd_en			(fpu_ff_rd_en),         // input wire rd_en
	.dout			(fpu_ff_rd_data),          // output wire [127 : 0] dout
	.empty			(fpu_ff_empty),         // output wire empty
	.rd_data_count	(fpu_ff_rd_cnt)  // output wire [9 : 0] rd_data_count
);


as32x512_ft as32x512_ft (
	.rst			(rst),           // input wire rst

	.wr_clk			(wr_clk),        // input wire wr_clk
	.wr_en			(wr_en),         // input wire wr_en
	.din			(din),           // input wire [31 : 0] din
	.full			(full),          // output wire full
	.wr_data_count	(wr_data_count), // output wire [8 : 0] wr_data_count

	.rd_clk			(rd_clk),        // input wire rd_clk
	.rd_en			(rd_en),         // input wire rd_en
	.dout			(dout),          // output wire [31 : 0] dout
	.empty			(empty),         // output wire empty
	.rd_data_count	(rd_data_count)  // output wire [8 : 0] rd_data_count
);

as72x512_ft as72x512_ft (
	.rst			(rst),           // input wire rst
	
	.wr_clk			(wr_clk),        // input wire wr_clk
	.wr_en			(wr_en),         // input wire wr_en
	.din			(din),           // input wire [71 : 0] din
	.full			(full),          // output wire full
	.wr_data_count	(wr_data_count), // output wire [8 : 0] wr_data_count

	.rd_clk			(rd_clk),        // input wire rd_clk
	.rd_en			(rd_en),         // input wire rd_en
	.dout			(dout),          // output wire [71 : 0] dout
	.empty			(empty),         // output wire empty
	.rd_data_count	(rd_data_count)  // output wire [8 : 0] rd_data_count
);

as128x1024 as128x1024 (
	.rst			(rst),           // input wire rst
	
	.wr_clk			(wr_clk),        // input wire wr_clk
	.wr_en			(wr_en),         // input wire wr_en
	.din			(din),           // input wire [127 : 0] din
	.full			(full),          // output wire full
	.wr_data_count	(wr_data_count), // output wire [9 : 0] wr_data_count
  
	.rd_clk			(rd_clk),        // input wire rd_clk
	.rd_en			(rd_en),         // input wire rd_en
	.dout			(dout),          // output wire [127 : 0] dout
	.empty			(empty),         // output wire empty
	.rd_data_count	(rd_data_count)  // output wire [9 : 0] rd_data_count
);


endmodule
