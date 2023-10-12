
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

as32x512_ft as32x512_ft (
  .rst				(rst),           // input wire rst

  .wr_clk			(wr_clk),        // input wire wr_clk
  .wr_en			(wr_en),         // input wire wr_en
  .din				(din),           // input wire [31 : 0] din
  .full				(full),          // output wire full
  .wr_data_count	(wr_data_count), // output wire [8 : 0] wr_data_count

  .rd_clk			(rd_clk),        // input wire rd_clk
  .rd_en			(rd_en),         // input wire rd_en
  .dout				(dout),          // output wire [31 : 0] dout
  .empty			(empty),         // output wire empty
  .rd_data_count	(rd_data_count)  // output wire [8 : 0] rd_data_count
);

as72x512_ft as72x512_ft (
  .rst				(rst),           // input wire rst
  
  .wr_clk			(wr_clk),        // input wire wr_clk
  .wr_en			(wr_en),         // input wire wr_en
  .din				(din),           // input wire [71 : 0] din
  .full				(full),          // output wire full
  .wr_data_count	(wr_data_count), // output wire [8 : 0] wr_data_count

  .rd_clk			(rd_clk),        // input wire rd_clk
  .rd_en			(rd_en),         // input wire rd_en
  .dout				(dout),          // output wire [71 : 0] dout
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
