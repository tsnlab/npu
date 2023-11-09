`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/25/2023 10:41:19 AM
// Design Name: 
// Module Name: tb_tsn_dgcl
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_dpc(

	);

	
logic risc_clk=0;
logic fpu_clk=0;
logic reset;

// Read Control Command Signal
logic [39:0] rcc_dram_addr;
logic [15:0] rcc_dpram_addr;
logic [15:0] rcc_length;
logic rcc_ready;
logic rcc_valid;
// Read Control Data Signal
logic [15:0] rcd_dpram_addr;
logic [127:0] rcd_read_data;
logic [15:0] rcd_length;
logic rcd_ready;
logic rcd_valid=0;

// Write Control Command Signal
logic [39:0] wcc_dram_addr;
logic [15:0] wcc_dpram_addr;
logic [15:0] wcc_length;
logic [127:0] wcc_write_data;
logic wcc_ready;
logic wcc_valid;

// TSN-DMA Signal A
logic dma_req_a;
logic dma_resp_a;
logic dma_write_valid_a;
logic [127:0] dma_write_data_a;
logic dma_write_ready_a;

logic dma_read_valid_a;
logic [127:0] dma_read_data_a;
logic dma_read_ready_a;

// TSN-DMA Signal B
logic dma_req_b;
logic dma_resp_b;
logic dma_write_valid_b;
logic [127:0] dma_write_data_b;
logic dma_write_ready_b;

logic dma_read_valid_b;
logic [127:0] dma_read_data_b;
logic dma_read_ready_b;

// TSN-DMA Signal C
logic dma_req_c;
logic dma_resp_c;
logic dma_write_valid_c;
logic [127:0] dma_write_data_c;
logic dma_write_ready_c;

logic dma_read_valid_c;
logic [127:0] dma_read_data_c;
logic dma_read_ready_c;

// TSN-DMA Signal D
logic dma_req_d;
logic dma_resp_d;
logic dma_write_valid_d;
logic [127:0] dma_write_data_d;
logic dma_write_ready_d;

logic dma_read_valid_d;
logic [127:0] dma_read_data_d;
logic dma_read_ready_d;

tsn_dgcl u0(
	.risc_clk			(risc_clk),
	.fpu_clk				(fpu_clk	),
	.reset					(reset		),
	
	// Read Control Command Signal
	.rcc_dram_addr			(rcc_dram_addr	),
	.rcc_dpram_addr			(rcc_dpram_addr	),
	.rcc_length				(rcc_length		),
	.rcc_ready				(rcc_ready		),
	.rcc_valid				(rcc_valid		),

	// Read Control Data Signal
	.rcd_dpram_addr			(rcd_dpram_addr	),
	.rcd_read_data			(rcd_read_data	),
	.rcd_length				(rcd_length		),
	.rcd_ready				(rcd_ready		),
	.rcd_valid				(rcd_valid		),
	
	// Write Control Command Signal
	.wcc_dram_addr			(wcc_dram_addr	),
	.wcc_dpram_addr			(wcc_dpram_addr	),
	.wcc_length				(wcc_length		),
	.wcc_write_data			(wcc_write_data	),
	.wcc_ready				(wcc_ready		),
	.wcc_valid				(wcc_valid		),
	
	// TSN-DMA Signal A
	.dma_req_a				(dma_req_a	),
	.dma_resp_a				(dma_resp_a	),

	.dma_write_valid_a		(dma_write_valid_a),
	.dma_write_data_a		(dma_write_data_a),
	.dma_write_ready_a		(dma_write_ready_a),
	
	.dma_read_valid_a		(dma_read_valid_a),
	.dma_read_data_a		(dma_read_data_a),
	.dma_read_ready_a		(dma_read_ready_a),
	
	// TSN-DMA Signal B
	.dma_req_b				(dma_req_b),
	.dma_resp_b				(dma_resp_b),

	.dma_write_valid_b		(dma_write_valid_b),
	.dma_write_data_b		(dma_write_data_b),
	.dma_write_ready_b		(dma_write_ready_b),
	
	.dma_read_valid_b		(dma_read_valid_b),
	.dma_read_data_b		(dma_read_data_b),
	.dma_read_ready_b		(dma_read_ready_b),

	// TSN-DMA Signal C
	.dma_req_c				(dma_req_c	),
	.dma_resp_c				(dma_resp_c	),

	.dma_write_valid_c		(dma_write_valid_c	),
	.dma_write_data_c		(dma_write_data_c	),
	.dma_write_ready_c		(dma_write_ready_c	),
	
	.dma_read_valid_c		(dma_read_valid_c),
	.dma_read_data_c		(dma_read_data_c),
	.dma_read_ready_c		(dma_read_ready_c),
	
	// TSN-DMA Signal D
	.dma_req_d				(dma_req_d	),
	.dma_resp_d				(dma_resp_d	),

	.dma_write_valid_d		(dma_write_valid_d),
	.dma_write_data_d		(dma_write_data_d),
	.dma_write_ready_d		(dma_write_ready_d),
	
	.dma_read_valid_d		(dma_read_valid_d),
	.dma_read_data_d		(dma_read_data_d),
	.dma_read_ready_d		(dma_read_ready_d)		
);
assign rcc_ready = 1'b1;
assign wcc_ready = 1'b1;
assign dma_read_ready_a = 1'b1;
assign dma_read_ready_b = 1'b1;
assign dma_read_ready_c = 1'b1;
assign dma_read_ready_d = 1'b1;
logic clk_en;
logic gclk_en;
int data_size=16;
int mess_size=4;
int rcd_index =0;
logic flag=0;

int i,j;

logic [127:0]rcc_arry[0:255];
logic [127:0]rcd_arry[0:255];
logic [31:0]rcd_h_arry[0:255];
int rcd_list[0:255];
logic [127:0]wcd_arry[0:255];

logic [127:0]fpu_a_arry[0:255];
int fpu_a_list[0:255];
logic [127:0]fpu_b_arry[0:255];
int fpu_b_list[0:255];
logic [127:0]fpu_c_arry[0:255];
int fpu_c_list[0:255];
logic [127:0]fpu_d_arry[0:255];
int fpu_d_list[0:255];
int fpu_index=0;


int rcc_index=0;
int wcd_index=0;
int read_index=0;

initial begin
	clk_en <= 1'b0;
	gclk_en <= 1'b0;
	reset <= 1'b1;
	#50;
	clk_en <= 1'b1;
	#10;
	gclk_en <= 1'b1;
	#10;

	reset <= 1'b0;
	#100;
	for(j = 0; j<mess_size;j++) begin
		for(i =0; i < data_size; i++) begin
			if(i==0) begin
				// rcd_arry[rcd_index] <= {48'd0,8'd2,data_size[15:0],40'd0,j[1:0],14'd0};
				rcd_arry[rcd_index] <= {i,i,i,i};
				rcd_h_arry[rcd_index] <= {data_size[15:0],j[1:0],14'd0};
				// $display("header");
			end
			else begin
				rcd_arry[rcd_index] <= {i,i,i,i};
				rcd_h_arry[rcd_index] <= {data_size[15:0],j[1:0],14'd0};
			end
			// $display("test i= %0d index = %0d data size = %0d",i,rcd_index,data_size[15:0]);
			rcd_list[j] <= i;
			rcd_index ++;
		end
	end
	for(j=0;j<mess_size;j++) begin
		for(i =0; i < data_size+1; i++) begin
			if(i==0) begin
				if(j[0]==1) begin
					fpu_a_arry[fpu_index] <= {48'h0000_0000_0000,8'h01,16'd0,8'd0,i[31:0],2'd0,14'd0};
					fpu_b_arry[fpu_index] <= {48'h0000_0000_0000,8'h01,16'd0,8'd0,i[31:0],2'd1,14'd0};
					fpu_c_arry[fpu_index] <= {48'h0000_0000_0000,8'h01,16'd0,8'd0,i[31:0],2'd2,14'd0};
					fpu_d_arry[fpu_index] <= {48'h0000_0000_0000,8'h01,16'd0,8'd0,i[31:0],2'd3,14'd0};
					fpu_index ++;
					fpu_a_list[j]=1;
					fpu_b_list[j]=1;
					fpu_c_list[j]=1;
					fpu_d_list[j]=1;
					break;
				end
				else begin
					fpu_a_arry[fpu_index] <= {48'h0000_0000_0000,8'h03,data_size[15:0],8'd0,i[31:0],i[1:0],14'd0};
					fpu_b_arry[fpu_index] <= {48'h0000_0000_0000,8'h03,data_size[15:0],8'd0,i[31:0],i[1:0],14'd0};
					fpu_c_arry[fpu_index] <= {48'h0000_0000_0000,8'h03,data_size[15:0],8'd0,i[31:0],i[1:0],14'd0};
					fpu_d_arry[fpu_index] <= {48'h0000_0000_0000,8'h03,data_size[15:0],8'd0,i[31:0],i[1:0],14'd0};
					fpu_index ++;
				end
			end
			else begin
				fpu_a_arry[fpu_index] <= {i,i,i,i};
				fpu_b_arry[fpu_index] <= {i,i,i,i};
				fpu_c_arry[fpu_index] <= {i,i,i,i};
				fpu_d_arry[fpu_index] <= {i,i,i,i};
				fpu_a_list[j]=i;
				fpu_b_list[j]=i;
				fpu_c_list[j]=i;
				fpu_d_list[j]=i;
				fpu_index ++;
			end
		end
	end
	#1000;
	forever begin
		flag <= 1'b1;
		#5;
	end
end


always fpu_clk = #10 (clk_en)? ~ fpu_clk : 1'b0;
always risc_clk = #10 (gclk_en)? ~ risc_clk : 1'b0;


always@(posedge risc_clk)begin
	if(rcc_valid) begin
		rcc_arry[rcc_index] <= {48'h0000_0000_0000,8'h00,rcc_length,rcc_dram_addr,rcc_dpram_addr};
		rcc_index ++;
	end

end


always@(posedge risc_clk)begin
	if(wcc_valid) begin
		wcd_arry[wcd_index] <= {48'h0000_0000_0000,8'h00,wcc_length,wcc_dram_addr,wcc_dpram_addr};
		wcd_index ++;
	end

end

int p_index=0;
int d_index=0;

always@(posedge risc_clk)begin
	if(rcd_ready) begin
		if(read_index==rcd_index) begin
			rcd_valid <= 1'b0;
		end
		else begin
			if(rcd_list[p_index] < d_index) begin
				rcd_valid <= 1'b0;
				d_index <= 0;
				p_index++;

			end
			else begin
				rcd_valid <= 1'b1;
				rcd_dpram_addr	<= rcd_h_arry[read_index][15:0];
				rcd_read_data	<= rcd_arry[read_index];
				rcd_length		<= rcd_h_arry[read_index][31:16];
				d_index ++;
				read_index ++;
			end
		end
	end
	else begin
	end

end

logic[3:0] st_a =0;
logic wr_a_en=0;
int Na = 0;
int a_list_cnt =0;
int pa_cnt =0;
always@(posedge fpu_clk)begin
	
	case(st_a)
		4'd0 : begin
			if(flag) begin
				if(Na==fpu_index) begin
					wr_a_en <= 1'b0;
					st_a <= 4'd3;
				end
				else begin
					dma_req_a <= 1'b1;
					if(dma_resp_a) begin
						st_a <= 4'd1;
						dma_req_a <= 1'b0;
					end
				end
				
			end
		end
		4'd1 : begin
			if(fpu_a_list[a_list_cnt]-1<=pa_cnt) begin
				wr_a_en <= 1'b1;
				st_a <= 4'd2;
				Na <= Na + 1;
				// $display("test1");
			end
			else if(dma_write_ready_a)begin
				if(dma_write_valid_a &&dma_write_ready_a) begin
					Na <= Na + 1;
					pa_cnt ++;
					// $display("test2");
					
				end
				else begin
				end 
				wr_a_en <= 1'b1;
				$display("test1");
			end
			else begin
				wr_a_en <= 1'b0;
				st_a <= 4'd1;
			end
		end
		4'd2 : begin
			a_list_cnt ++;
			pa_cnt <= 0;
			if(Na==fpu_index) begin
				wr_a_en <= 1'b0;
				st_a <= 4'd3;
			end
			else begin
				if(pa_cnt == 0) begin
					Na <= Na + 1;
				end
				wr_a_en <= 1'b0;
				st_a <= 4'd0;
			end
		end
		default : begin
		end
	endcase
end

assign dma_write_valid_a = wr_a_en;
assign dma_write_data_a =fpu_a_arry[Na];


logic[3:0] st_b =0;
logic wr_b_en=0;
int Nb = 0;
int b_list_cnt =0;
int pb_cnt =0;
always@(posedge fpu_clk)begin
	
	case(st_b)
		4'd0 : begin
			if(flag) begin
				if(Nb==fpu_index) begin
					wr_b_en <= 1'b0;
					st_b <= 4'd3;
				end
				else begin
					dma_req_b <= 1'b1;
					if(dma_resp_b) begin
						st_b <= 4'd1;
						dma_req_b <= 1'b0;
					end
				end
			end
		end
		4'd1 : begin
			if(fpu_b_list[b_list_cnt]-1<=pb_cnt) begin
				wr_b_en <= 1'b1;
				st_b <= 4'd2;
				Nb <= Nb + 1;
			end
			else if(dma_write_ready_b)begin
				if(dma_write_valid_b &&dma_write_ready_b) begin
					Nb <= Nb + 1;
					pb_cnt ++;
				end
				else begin
				end 
				wr_b_en <= 1'b1;
			end
			else begin
				wr_b_en <= 1'b0;
				st_b <= 4'd1;
			end
		end
		4'd2 : begin
			b_list_cnt ++;
			pb_cnt <= 0;
			if(Nb==fpu_index) begin
				wr_b_en <= 1'b0;
				st_b <= 4'd3;
			end
			else begin
				if(pb_cnt == 0) begin
					Nb <= Nb + 1;
				end
				wr_b_en <= 1'b0;
				st_b <= 4'd0;
			end
		end
		default : begin
		end
	endcase
end

assign dma_write_valid_b = wr_b_en;
assign dma_write_data_b =fpu_b_arry[Nb];



logic[3:0] st_c =0;
logic wr_c_en=0;
int Nc = 0;
int c_list_cnt =0;
int pc_cnt =0;
always@(posedge fpu_clk)begin
	
	case(st_c)
		4'd0 : begin
			if(flag) begin
				if(Nc==fpu_index) begin
					wr_c_en <= 1'b0;
					st_c <= 4'd3;
				end
				else begin
					dma_req_c <= 1'b1;
					if(dma_resp_c) begin
						st_c <= 4'd1;
						dma_req_c <= 1'b0;
					end
				end
			end
		end
		4'd1 : begin
			if(fpu_c_list[c_list_cnt]-1<=pc_cnt) begin
				wr_c_en <= 1'b1;
				st_c <= 4'd2;
				Nc <= Nc + 1;
			end
			else if(dma_write_ready_c)begin
				if(dma_write_valid_c &&dma_write_ready_c) begin
					Nc <= Nc + 1;
					pc_cnt ++;
				end
				else begin
				end 
				wr_c_en <= 1'b1;
			end
			else begin
				wr_c_en <= 1'b0;
				st_c <= 4'd1;
			end
		end
		4'd2 : begin
			c_list_cnt ++;
			pc_cnt <= 0;
			if(Nc==fpu_index) begin
				wr_c_en <= 1'b0;
				st_c <= 4'd3;
			end
			else begin
				if(pc_cnt == 0) begin
					Nc <= Nc + 1;
				end
				wr_c_en <= 1'b0;
				st_c <= 4'd0;
			end
		end
		default : begin
		end
	endcase
end


assign dma_write_valid_c = wr_c_en;
assign dma_write_data_c =fpu_c_arry[Nc];



logic[3:0] st_d =0;
logic wr_d_en=0;
int Nd = 0;
int d_list_cnt =0;
int pd_cnt =0;
always@(posedge fpu_clk)begin
	
	case(st_d)
		4'd0 : begin
			if(flag) begin
				if(Nd==fpu_index) begin
					wr_d_en <= 1'b0;
					st_d <= 4'd3;
				end
				else begin
					dma_req_d <= 1'b1;
					if(dma_resp_d) begin
						st_d <= 4'd1;
						dma_req_d <= 1'b0;
					end
				end
			end
		end
		4'd1 : begin
			if(fpu_d_list[d_list_cnt]-1<=pd_cnt) begin
				wr_d_en <= 1'b1;
				st_d <= 4'd2;
				Nd <= Nd + 1;
			end
			else if(dma_write_ready_d)begin
				if(dma_write_valid_d &&dma_write_ready_d) begin
					Nd <= Nd + 1;
					pd_cnt ++;
				end
				else begin
				end 
				wr_d_en <= 1'b1;
			end
			else begin
				wr_d_en <= 1'b0;
				st_d <= 4'd1;
			end
		end
		4'd2 : begin
			d_list_cnt ++;
			pd_cnt <= 0;
			if(Nd==fpu_index) begin
				wr_d_en <= 1'b0;
				st_d <= 4'd3;
			end
			else begin
				if(pd_cnt == 0) begin
					Nd <= Nd + 1;
				end
				wr_d_en <= 1'b0;
				st_d <= 4'd0;
			end
		end
		default : begin
		end
	endcase
end

assign dma_write_valid_d = wr_d_en;
assign dma_write_data_d =fpu_d_arry[Nd];

endmodule
