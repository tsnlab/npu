
////////////////////////////////////////////////////////////////////////////////
// Company: TSNLAB
// Engineer: junhyuk
//
// Create Date: 2023-10-10
// Design Name: TSN-NPU
// Module Name: DMAPathController
// Tool versions: vivado 2018.3
// Description:
//    <Description here>
// Revision:
//    0.01 junhyuk - Create
////////////////////////////////////////////////////////////////////////////////
			
module DMAPathController(
	input wire risc_clk,
	input wire fpu_clk,
	input wire reset,
	
	//***** RISC DMA Block
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
	
	//***** TSN-FPU Block
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

//***** Read Controller Command Path FIFO Signal
reg rcc_ff_wr_en;
reg [71:0] rcc_ff_wr_data;
wire rcc_ff_full;
wire [4:0] rcc_ff_wr_cnt;
wire rcc_ff_rd_en;
wire [71:0] rcc_ff_rd_data;
wire rcc_ff_empty;
wire [4:0] rcc_ff_rd_cnt;


//***** Read Controller Data Path Header FIFO Signal
wire rcd_ff_c_wr_en;
reg rcd_ff_c_wr_en_d;
reg [31:0] rcd_ff_c_wr_data;
wire rcd_ff_c_full;
wire [7:0] rcd_ff_c_wr_cnt;
wire rcd_ff_c_rd_en;
wire [31:0] rcd_ff_c_rd_data;
wire rcd_ff_c_empty;
wire [7:0] rcd_ff_c_rd_cnt;

//***** Read Controller Data Path Data FIFO Signal
wire rcd_ff_d_wr_en;
wire [127:0] rcd_ff_d_wr_data;
wire rcd_ff_d_full;
wire [7:0] rcd_ff_d_wr_cnt;
wire rcd_ff_d_rd_en;
wire [127:0] rcd_ff_d_rd_data;
wire rcd_ff_d_empty;
wire [7:0] rcd_ff_d_rd_cnt;

//***** Read Controller Data State Machine
localparam rcd_st = 2'd0,
			rcd_s0 = 2'd1,
			rcd_end = 2'd2;
reg [1:0] rcdcon = rcd_st;

reg rcd_cmd_valid_d ;
reg[127:0] dma_r_da;

reg dpc_rdy;

//***** DMA Token Control State Machine
localparam dtc_st = 4'd0,
			dtc_s0 = 4'd1,
			dtc_s1 = 4'd2,
			dtc_s2 = 4'd3,
			dtc_s3 = 4'd4,
			dtc_s4 = 4'd5;

reg[3:0] dtpcon = dtc_st;
reg fpu_sel_a;
reg fpu_sel_b;
reg fpu_sel_c;
reg fpu_sel_d;
reg fpu_resp_done;

//***** FPU Write Date State Machine
localparam fwdc_st = 4'd0,
			fwdc_s0 = 4'd1,
			fwdc_s1 = 4'd2,
			fwdc_end = 4'd3;
reg[3:0] fwdcon = fwdc_st;


wire[3:0] npu_addr;
reg[7:0] ad_wr_ptr;
reg[1:0]npu_ad_mem[0:255];
reg[7:0] ad_rd_ptr;

//***** Read Controller Data Signal
reg[15:0] rcdcon_count;
reg[15:0] rcdcon_length;
wire dma_read_vld;
wire[127:0] dma_read_data;
wire[1:0] dma_mux;
wire dma_read_ready_mux;

//***** FPU Token Control Signal
reg[1:0] fpu_sel =2'd0;
reg fpu_resp;
wire fpu_write_rdy;
wire [127:0] fpu_write_data;
wire fpu_write_vld;
reg[15:0] fpu_write_length;
reg[7:0] msg_form;
reg[15:0] fpu_write_cnt;
wire [3:0] dma_req;

//***** FPU FIFO Control Signal
wire fpu_ff_wr_en;
wire [127:0] fpu_ff_wr_data;
wire fpu_ff_full;
wire [7:0] fpu_ff_wr_cnt;
wire fpu_ff_rd_en;
wire [127:0] fpu_ff_rd_data;
wire fpu_ff_empty;
wire [7:0] fpu_ff_rd_cnt;

//***** DMA Command Control State Machine
localparam dcc_st = 4'd0,
			dcc_s0 = 4'd1,
			dcc_rc0 = 4'd2,
			dcc_rc1 = 4'd3,
			dcc_wc0 = 4'd4,
			dcc_wc1 = 4'd5,
			dcc_wc2 = 4'd6;

reg[3:0] dcccon = dcc_st;

reg[1:0] npu_mask;
reg[15:0]dcc_length =0;
reg[15:0]dcc_count=0;
reg[127:0] dcc_header_r;
reg[7:0] mess_form;

//***** Write Controller Command Signal
reg wcc_ff_wr_en;
reg [71:0] wcc_ff_wr_data;
wire wcc_ff_full;
wire [4:0] wcc_ff_wr_cnt;
reg wcc_ff_rd_en;
wire [71:0] wcc_ff_rd_data;
wire wcc_ff_empty;
wire [4:0] wcc_ff_rd_cnt;


//***** Write Controller Data Signal
reg wcd_read_en;
reg wcd_read_da_en;
wire wcd_ff_wr_en;
wire [127:0] wcd_ff_wr_data;
wire wcd_ff_full;
wire [7:0] wcd_ff_wr_cnt;
wire wcd_ff_rd_en;
wire [127:0] wcd_ff_rd_data;
wire wcd_ff_empty;
wire [7:0] wcd_ff_rd_cnt;

//***** Group Write Control State Machine
localparam gwcd_st = 4'd0,
			gwcd_s0 = 4'd1,
			gwcd_end = 4'd2;
reg[3:0] gwcdcon;
reg[15:0] gwcd_length;
reg[15:0] gwcd_count;
reg gwcd_en;

//***** FPU DMA Token Control
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		fpu_sel_a <= 1'b0;
		fpu_sel_b <= 1'b0;
		fpu_sel_c <= 1'b0;
		fpu_sel_d <= 1'b0;
		ad_wr_ptr <= 8'd0;
		dtpcon <= dtc_st;
	end
	else begin
		case(dtpcon)
			dtc_st : begin
				fpu_sel_a <= 1'b0;
				fpu_sel_b <= 1'b0;
				fpu_sel_c <= 1'b0;
				fpu_sel_d <= 1'b0;
				dtpcon <= dtc_s0;
			end
			dtc_s0 : begin
				if(dma_req_a) begin 
					fpu_sel <= 2'd0; 
					fpu_sel_a <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
					fpu_sel_b <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
					fpu_sel_c <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
					fpu_sel_d <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else begin
					dtpcon <= dtc_s1;
				end
			end
			dtc_s1 : begin
				if(dma_req_b) begin
					fpu_sel <= 2'd1;
					fpu_sel_b <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
					fpu_sel_c <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
					fpu_sel_d <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
					fpu_sel_a <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else begin
					dtpcon <= dtc_s2;
				end
			end
			dtc_s2 : begin
				if(dma_req_c) begin
					fpu_sel <= 2'd2;
					fpu_sel_c <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_d) begin
					fpu_sel <= 2'd3;
					fpu_sel_d <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
					fpu_sel_a <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
					fpu_sel_b <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else begin
					dtpcon <= dtc_s3;
				end
			end
			dtc_s3 : begin
				if(dma_req_d) begin
					fpu_sel <= 2'd3;
					fpu_sel_d <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_a) begin
					fpu_sel <= 2'd0;
					fpu_sel_a <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_b) begin
					fpu_sel <= 2'd1;
					fpu_sel_b <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else if(dma_req_c) begin
					fpu_sel <= 2'd2;
					fpu_sel_c <= 1'b1;
					dtpcon <= dtc_s4;
				end
				else begin
					dtpcon <= dtc_s0;
				end
			end
			dtc_s4 : begin
				if(fpu_resp_done) begin
					if(fpu_sel_a)begin
						fpu_sel_a <= 1'b0;
						npu_ad_mem[ad_wr_ptr] <= 2'd0;
						ad_wr_ptr <= ad_wr_ptr + 8'd1;
						dtpcon <= dtc_s1;
					end
					else if(fpu_sel_b) begin
						fpu_sel_b <= 1'b0;
						npu_ad_mem[ad_wr_ptr] <= 2'd1;
						ad_wr_ptr <= ad_wr_ptr + 8'd1;
						dtpcon <= dtc_s2;
					end
					else if(fpu_sel_c) begin
						fpu_sel_c <= 1'b0;
						npu_ad_mem[ad_wr_ptr] <= 2'd2;
						ad_wr_ptr <= ad_wr_ptr + 8'd1;
						dtpcon <= dtc_s3;
					end
					else if(fpu_sel_d) begin
						fpu_sel_d <= 1'b0;
						npu_ad_mem[ad_wr_ptr] <= 2'd3;
						ad_wr_ptr <= ad_wr_ptr + 8'd1;
						dtpcon <= dtc_s0;
					end
					else begin
						fpu_sel_a <= 1'b0;
						fpu_sel_b <= 1'b0;
						fpu_sel_c <= 1'b0;
						fpu_sel_d <= 1'b0;
						dtpcon <= dtc_s0;
					end
				end
				else begin
					dtpcon <= dtc_s4;
				end
			end
			default : begin
				fpu_sel_a <= 1'b0;
				fpu_sel_b <= 1'b0;
				fpu_sel_c <= 1'b0;
				fpu_sel_d <= 1'b0;
				dtpcon <= dtc_s0;
			end
		endcase
	end
end

assign npu_addr = {fpu_sel_d,fpu_sel_c,fpu_sel_b,fpu_sel_a};

//***** FPU DMA Write Data control 
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		fwdcon <= fwdc_st;
		fpu_resp <= 1'b0;
		fpu_write_cnt <= 16'd0;
		fpu_write_length <= 16'd0;
		msg_form <= 8'd0;
		fpu_resp_done <= 1'b0;
		dpc_rdy <= 1'b0;

	end
	else begin
		case(fwdcon) 
			fwdc_st : begin
				fpu_resp_done <= 1'b0;
				fpu_write_cnt <= 16'd0;
				fpu_resp <= 1'b0;
				
				if(fpu_sel_a || fpu_sel_b || fpu_sel_c || fpu_sel_d) begin
					
					fwdcon <= fwdc_s0;
				end
				else begin
					fwdcon <= fwdc_st;
				end
			end
			fwdc_s0 : begin
				fpu_resp <= 1'b1;
				dpc_rdy <= 1'b1;
				if(fpu_write_vld&&fpu_write_rdy)begin
					fpu_write_cnt <= fpu_write_cnt + 1;
					fpu_write_length <= fpu_write_data[71:56];
					msg_form <= fpu_write_data[79:72];
					fpu_resp <= 1'b0;
					fwdcon <= fwdc_s1;
				end
				
			end
			fwdc_s1 : begin
				if(msg_form==8'h01) begin
					dpc_rdy <= 1'b0;
					fwdcon <= fwdc_end;
					fpu_resp <= 1'b0;
				end
				else if(fpu_write_cnt >= fpu_write_length) begin
					dpc_rdy <= 1'b0;
					fwdcon <= fwdc_end;
					fpu_resp <= 1'b0;
				end
				else if(fpu_write_vld&&fpu_write_rdy)begin
					dpc_rdy <= 1'b1;
					fpu_write_cnt <= fpu_write_cnt + 1;
					fwdcon <= fwdc_s1;
					fpu_resp <= 1'b0;
				end
			end
			fwdc_end : begin
				fpu_resp_done <= 1'b1;
				fpu_write_cnt <= 16'd0;
				fpu_write_length <= 16'd0;
				fpu_resp <= 1'b0;
				dpc_rdy <= 1'b0;
				fwdcon <= fwdc_st;
			end
			default : begin
				fpu_resp_done <= 1'b0;
				fpu_write_cnt <= 16'd0;
				fpu_write_length <= 16'd0;
				fpu_resp <= 1'b0;
				dpc_rdy <= 1'b0;
				fwdcon <= fwdc_st;
			end
		endcase
		
	end
end

assign dma_resp_a = (fpu_sel_a)?  fpu_resp : 1'b0;
assign dma_resp_b = (fpu_sel_b)?  fpu_resp : 1'b0;
assign dma_resp_c = (fpu_sel_c)?  fpu_resp : 1'b0;
assign dma_resp_d = (fpu_sel_d)?  fpu_resp : 1'b0;
assign dma_write_ready_a = (fpu_sel_a)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_b = (fpu_sel_b)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_c = (fpu_sel_c)?  fpu_write_rdy : 1'b0;
assign dma_write_ready_d = (fpu_sel_d)?  fpu_write_rdy : 1'b0;

assign fpu_write_data = (fpu_sel_a)? dma_write_data_a : (fpu_sel_b)? dma_write_data_b : (fpu_sel_c)? dma_write_data_c : (fpu_sel_d)? dma_write_data_d : 128'd0;
assign fpu_write_vld = (fpu_sel_a)? dma_write_valid_a : (fpu_sel_b)? dma_write_valid_b : (fpu_sel_c)? dma_write_valid_c : (fpu_sel_d)? dma_write_valid_d : 1'b0;
assign dma_req = {dma_req_c,dma_req_d,dma_req_b,dma_req_a};

assign fpu_ff_wr_en = fpu_write_vld &&(~fpu_ff_full)&&(dpc_rdy);
assign fpu_ff_wr_data = fpu_write_data;
assign fpu_write_rdy = (~fpu_ff_full)&&(dpc_rdy);

as128x256_ft fpu_write_fifo (
	.rst			(reset),           // input wire rst
	
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


//***** Select write or read data
always@(posedge fpu_clk or posedge reset) begin
	if(reset) begin
		wcd_read_en <= 1'b0;
		dcc_length <= 0;
		mess_form <= 0;
		dcc_header_r <= 0;
		rcc_ff_wr_data <= 0;
		dcc_count <= 16'd1;
		dcccon <= dcc_st;
		wcc_ff_wr_en <= 1'b0;
		wcd_read_da_en <= 1'b0;
		rcc_ff_wr_en <= 1'b0;
		wcc_ff_wr_data <= 128'd0;
		ad_rd_ptr <= 8'd0;
	end
	else begin
		case(dcccon)
			dcc_st : begin
				wcc_ff_wr_data <= 128'd0;
				dcc_count <= 16'd1;
				rcc_ff_wr_en <= 1'b0;
				if(~fpu_ff_empty) begin
					wcd_read_en <= 1'b1;
					wcd_read_da_en <= 1'b0;
					dcc_length <= fpu_ff_rd_data[71:56];
					mess_form <= fpu_ff_rd_data[79:72];
					dcc_header_r <= fpu_ff_rd_data;
					npu_mask <= npu_ad_mem[ad_rd_ptr];
					ad_rd_ptr <= ad_rd_ptr + 8'd1;
					dcccon <= dcc_s0;
				end
				else begin
					dcccon <= dcc_st;
				end 
			end
			dcc_s0 : begin
				wcd_read_en <= 1'b0;
				wcd_read_da_en <= 1'b0;
				if(mess_form == 8'h01) begin
					dcccon <= dcc_rc0;
				end
				else if(mess_form == 8'h03) begin
					dcccon <= dcc_wc0;
				end
			end
			dcc_rc0 : begin
				rcc_ff_wr_en <= 1'b1;
				rcc_ff_wr_data <= {dcc_header_r[71:16],npu_mask,2'b00,dcc_header_r[11:0]};
				dcccon <= dcc_rc1;
			end
			dcc_rc1 : begin
				rcc_ff_wr_en <= 1'b0;
				dcc_header_r <= 128'd0;
				dcccon <= dcc_st;
			end
			dcc_wc0 : begin
				wcc_ff_wr_en <= 1'b1;
				wcc_ff_wr_data <= {dcc_header_r[71:16],npu_mask,2'b00,dcc_header_r[11:0]};
				dcccon <= dcc_wc1;
			end
			dcc_wc1 : begin
				wcc_ff_wr_en <= 1'b0;
				if(dcc_length == dcc_count) begin
						wcd_read_en <= 1'b0;
						wcd_read_da_en <= 1'b0;
						dcc_count <= 16'd1;
						dcccon <= dcc_wc2;

				end
				else if(~wcd_ff_full&&wcd_ff_wr_en) begin
					wcd_read_en <= 1'b1;
					wcd_read_da_en <= 1'b1; 
					dcc_count <= dcc_count + 1;
					dcccon <= dcc_wc1;


				end
				else begin
					wcd_read_en <= 1'b1;
					wcd_read_da_en <= 1'b1;
					dcc_count <= dcc_count;
					dcccon <= dcc_wc1;
				end

			end
			dcc_wc2 : begin
				wcd_read_en <= 1'b0;
				wcd_read_da_en <= 1'b0;
				dcc_count <= 16'd1;
				dcccon <= dcc_st;
			end
			default : begin
				wcd_read_en <= 1'b0;
				wcd_read_da_en <= 1'b0;
				dcc_count <= 16'd1;
				dcccon <= dcc_st;
			end
		endcase
	end
end

assign wcd_ff_wr_en = (wcd_read_da_en && (~wcd_ff_full) &&(~fpu_ff_empty));
assign fpu_ff_rd_en = (wcd_read_en && (~wcd_ff_full) &&(~fpu_ff_empty));
assign wcd_ff_wr_data = fpu_ff_rd_data;

as72x32_ft wcc_ff (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(fpu_clk),        // input wire wr_clk
	.wr_en			(wcc_ff_wr_en),         // input wire wr_en
	.din			(wcc_ff_wr_data),           // input wire [71 : 0] din
	.full			(wcc_ff_full),          // output wire full
	.wr_data_count	(wcc_ff_wr_cnt), // output wire [8 : 0] wr_data_count

	.rd_clk			(risc_clk),        // input wire rd_clk
	.rd_en			(wcc_ff_rd_en),         // input wire rd_en
	.dout			(wcc_ff_rd_data),          // output wire [71 : 0] dout
	.empty			(wcc_ff_empty),         // output wire empty
	.rd_data_count	(wcc_ff_rd_cnt)  // output wire [8 : 0] rd_data_count
);

as128x256_ft wcd_ff (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(fpu_clk),        // input wire wr_clk
	.wr_en			(wcd_ff_wr_en),         // input wire wr_en
	.din			(wcd_ff_wr_data),           // input wire [127 : 0] din
	.full			(wcd_ff_full),          // output wire full
	.wr_data_count	(wcd_ff_wr_cnt), // output wire [9 : 0] wr_data_count
  
	.rd_clk			(risc_clk),        // input wire rd_clk
	.rd_en			(wcd_ff_rd_en),         // input wire rd_en
	.dout			(wcd_ff_rd_data),          // output wire [127 : 0] dout
	.empty			(wcd_ff_empty),         // output wire empty
	.rd_data_count	(wcd_ff_rd_cnt)  // output wire [9 : 0] rd_data_count
);

//***** Risc-v Stream Write inter face
reg [39:0] gwcd_daddr;
reg [15:0] gwcd_dpaddr;
always@(posedge risc_clk or posedge reset)begin
	if(reset)begin
		gwcdcon <= gwcd_st;
		gwcd_length <= 0;
		wcc_ff_rd_en <= 1'b0;
		gwcd_length <= 0;
		gwcd_count <= 1;
		gwcd_en <= 1'b0; 
		gwcd_daddr <= 40'd0;
		gwcd_dpaddr <= 16'd0;
	end
	else begin
		case(gwcdcon)
			gwcd_st : begin
				
				gwcd_count <= 1;
				if((~wcc_ff_empty)&&(~wcd_ff_empty)) begin
					if(wcc_ready) begin
						wcc_ff_rd_en <= 1;
						gwcdcon <= gwcd_s0;
						gwcd_length <= wcc_ff_rd_data[71:56];
						gwcd_daddr <= wcc_ff_rd_data[55:16];
						gwcd_dpaddr <= wcc_ff_rd_data[15:0];
					end
					else begin
						gwcdcon <= gwcd_st;
					end
				end
				else begin
					gwcdcon <= gwcd_st;
					gwcd_length <= wcc_ff_rd_data[71:56];
				end
			end
			gwcd_s0 : begin
				wcc_ff_rd_en <= 1'b0;
				if(gwcd_count==gwcd_length) begin
					gwcd_en <= 1'b0; 
					gwcd_count <= 1;
					gwcdcon <= gwcd_end;
				end
				else begin
					gwcd_en <= 1'b1; 
					gwcdcon <= gwcd_s0;
					if(wcc_ready)begin
						gwcd_count <= gwcd_count + 1;
					end
				end
			end
			gwcd_end: begin
				wcc_ff_rd_en <= 1'b0;
				gwcd_length <= 0;
				gwcd_count <= 1;
				gwcdcon <= gwcd_st;
			end
			default : begin
				wcc_ff_rd_en <= 1'b0;
				gwcd_length <= 0;
				gwcd_count <= 1;
				gwcdcon <= gwcd_st;
			end
		endcase
	end
end

assign wcd_ff_rd_en = (gwcd_en &&wcc_ready);
assign wcc_dram_addr = gwcd_daddr;
assign wcc_dpram_addr = gwcd_dpaddr;
assign wcc_length = gwcd_length;
assign wcc_write_data = wcd_ff_rd_data;
assign wcc_valid = gwcd_en;



as72x32_ft rcc_fifo (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(fpu_clk),        // input wire wr_clk
	.wr_en			(rcc_ff_wr_en),         // input wire wr_en
	.din			(rcc_ff_wr_data),           // input wire [71 : 0] din
	.full			(rcc_ff_full),          // output wire full
	.wr_data_count	(rcc_ff_wr_cnt), // output wire [8 : 0] wr_data_count

	.rd_clk			(risc_clk),        // input wire rd_clk
	.rd_en			(rcc_ff_rd_en),         // input wire rd_en
	.dout			(rcc_ff_rd_data),          // output wire [71 : 0] dout
	.empty			(rcc_ff_empty),         // output wire empty
	.rd_data_count	(rcc_ff_rd_cnt)  // output wire [8 : 0] rd_data_count
);

assign rcc_ff_rd_en = (~rcc_ff_empty) && (rcc_ready);
assign rcc_valid = (~rcc_ff_empty) && (rcc_ready);
assign rcc_dram_addr = ((~rcc_ff_empty) && (rcc_ready))? rcc_ff_rd_data[55:16] : rcc_dram_addr;
assign rcc_dpram_addr = ((~rcc_ff_empty) && (rcc_ready))? rcc_ff_rd_data[15:0] : rcc_dpram_addr;
assign rcc_length = ((~rcc_ff_empty) && (rcc_ready))? rcc_ff_rd_data[71:56] : rcc_length;

//***** Read Controller Date Path Interface
always@(posedge risc_clk or posedge reset) begin
	if(reset)begin
		rcd_cmd_valid_d <= 1'b0;
		rcd_ff_c_wr_data <= 32'd0;
	end
	else begin
		rcd_cmd_valid_d <= rcd_valid&&(~rcd_ff_c_full);
		rcd_ff_c_wr_en_d <= rcd_ff_c_wr_en;
		rcd_ff_c_wr_data[15:0] <= rcd_dpram_addr;
		rcd_ff_c_wr_data[31:16] <= rcd_length;
	end
end

// assign rcd_ff_c_wr_en = (~rcd_cmd_valid_d)&&(rcd_valid);
assign rcd_ff_c_wr_en = (rcd_valid)&&(rcd_ready);
assign rcd_ff_d_wr_en = (rcd_valid && (rcd_ready));
assign rcd_ready = ((~rcd_ff_d_full) && (~rcd_ff_c_full));
assign rcd_ff_d_wr_data = rcd_read_data;

as32x256_ft rcd_com_fifo (
	.rst			(reset),           // input wire rst

	.wr_clk			(risc_clk),        // input wire wr_clk
	.wr_en			(rcd_ff_c_wr_en_d),         // input wire wr_en
	.din			(rcd_ff_c_wr_data),           // input wire [31 : 0] din
	.full			(rcd_ff_c_full),          // output wire full
	.wr_data_count	(rcd_ff_c_wr_cnt), // output wire [8 : 0] wr_data_count

	.rd_clk			(fpu_clk),        // input wire rd_clk
	.rd_en			(rcd_ff_c_rd_en),         // input wire rd_en
	.dout			(rcd_ff_c_rd_data),          // output wire [31 : 0] dout
	.empty			(rcd_ff_c_empty),         // output wire empty
	.rd_data_count	(rcd_ff_c_rd_cnt)  // output wire [8 : 0] rd_data_count
);

as128x256_ft rcd_data_fifo (
	.rst			(reset),           // input wire rst
	
	.wr_clk			(risc_clk),        // input wire wr_clk
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


// always@(posedge fpu_clk or posedge reset) begin
// 	if(reset) begin
// 		dma_read_vld <= 1'b0;
// 		rcd_ff_d_rd_en <= 1'b0;
// 		dma_r_da <= 128'd0;
// 		rcdcon_count<= 16'd1;
// 		dma_mux <= 2'b00;
// 		rcdcon <= rcd_st;

// 	end
// 	else begin
// 		case(rcdcon)
// 			rcd_st : begin
// 				if((~rcd_ff_c_empty)&&(~rcd_ff_d_empty)) begin
// 					rcd_ff_c_rd_en <= 1'b1;
// 					rcdcon_length <= rcd_ff_c_rd_data[31:16]+1;

// 					dma_r_da[15:0] <= rcd_ff_c_rd_data[15:0];
// 					dma_r_da[55:0] <= 0;
// 					dma_r_da[71:56] <= rcd_ff_c_rd_data[31:16];
// 					dma_r_da[79:72] <= 8'h02;
// 					dma_r_da[127:80] <= 0;
// 					dma_mux <= rcd_ff_c_rd_data[15:14];

// 					dma_read_vld <= 1'b1;
// 					rcdcon <= rcd_s0;
// 				end
// 				else begin
// 					rcd_ff_c_rd_en <= 1'b0;
// 					rcdcon_length <= 0;
// 					dma_read_vld <= 1'b0;
// 					dma_r_da <= 0;
// 					rcdcon <= rcd_st;
// 				end
// 			end
// 			rcd_s0 : begin
				
// 				rcd_ff_c_rd_en <= 1'b0;
// 				if(rcdcon_length == rcdcon_count)begin
// 					dma_read_vld <= 1'b0;
// 					rcd_ff_d_rd_en <= 1'b0;
// 					dma_r_da <= 128'd0;
// 					rcdcon_count<= 16'd1;
// 					rcdcon <= rcd_end;

// 				end
// 				else begin
// 					if(dma_read_ready_mux) begin
// 						dma_read_vld <= 1'b1;
// 						rcd_ff_d_rd_en <= 1'b1;
// 						dma_r_da <= rcd_ff_d_rd_data;
// 						rcdcon_count<= rcdcon_count + 1;
// 						rcdcon <= rcd_s0;

// 					end
// 					else begin
// 						dma_read_vld <= 1'b1;
// 						rcd_ff_d_rd_en <= 1'b0;
// 						dma_r_da <= rcd_ff_d_rd_data;
// 						rcdcon_count<= rcdcon_count;
// 						rcdcon <= rcd_s0;
// 					end
// 				end
// 			end
// 			rcd_end : begin
// 				dma_read_vld <= 1'b0;
// 				rcd_ff_d_rd_en <= 1'b0;
// 				dma_r_da <= 128'd0;
// 				rcdcon_count<= 16'd1;
// 				rcdcon <= rcd_st;
// 			end
// 			default : begin
// 			end
// 		endcase

// 	end
// end

assign dma_mux = rcd_ff_c_rd_data[15:14];
assign dma_read_vld = (~rcd_ff_d_empty && ~rcd_ff_c_empty && dma_read_ready_mux)? 1'b1 : 1'b0;
assign rcd_ff_c_rd_en = (~rcd_ff_d_empty && ~rcd_ff_c_empty && dma_read_ready_mux)? 1'b1 : 1'b0;
assign rcd_ff_d_rd_en = (~rcd_ff_d_empty && ~rcd_ff_c_empty && dma_read_ready_mux)? 1'b1 : 1'b0;

// assign dma_read_data = (rcd_ff_d_rd_en)? rcd_ff_d_rd_data : dma_r_da;
assign dma_read_data = rcd_ff_d_rd_data;

assign dma_read_valid_a = (dma_mux==2'b00)? dma_read_vld : 1'b0;
assign dma_read_data_a = dma_read_data;

assign dma_read_valid_b = (dma_mux==2'b01)? dma_read_vld : 1'b0;
assign dma_read_data_b = dma_read_data;

assign dma_read_valid_c = (dma_mux==2'b10)? dma_read_vld : 1'b0;
assign dma_read_data_c = dma_read_data;

assign dma_read_valid_d = (dma_mux==2'b11)? dma_read_vld : 1'b0;
assign dma_read_data_d = dma_read_data;

assign dma_read_ready_mux = (dma_mux==2'b00)? dma_read_ready_a : (dma_mux==2'b01)? dma_read_ready_b : (dma_mux==2'b10)? dma_read_ready_c : (dma_mux==2'b11)? dma_read_ready_d : 1'b0; 

endmodule
