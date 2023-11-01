`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/10/31 16:16:39
// Design Name: output wire 
// Module Name: tb_lsc
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


module tb_lsc(
    );


logic clk =0;
logic reset;

logic core_req;
logic core_ready;
logic core_rwn;
logic [39:0] core_host_addr;
logic [13:0] core_local_addr;
logic [39:0] core_transfer_length;
logic core_ack;
logic [127:0] core_write_data=0;
logic [127:0] core_read_data;

logic dma_req;
logic dma_resp;
logic dma_write_valid;
logic [127:0] dma_write_data;
logic dma_write_ready;
logic dma_read_valid;
logic [127:0] dma_read_data;
logic dma_read_ready;

always clk = #10 ~clk;

dmaController u0(
	.clk(clk),
	.reset(reset),

	.core_req(core_req),
	.core_ready(core_ready),
  .core_rwn(core_rwn),
	.core_host_addr(core_host_addr),
  .core_local_addr(core_local_addr),
	.core_transfer_length(core_transfer_length),
	.core_ack(core_ack),
	.core_write_data(core_write_data),
	.core_read_data(core_read_data),

  .dma_req(dma_req),
  .dma_resp(dma_resp),
  .dma_write_valid(dma_write_valid),
  .dma_write_data(dma_write_data),
  .dma_write_ready(dma_write_ready),
  .dma_read_valid(dma_read_valid),
  .dma_read_data(dma_read_data),
	.dma_read_ready(dma_read_ready)
);

logic[127:0] wr_data[0:255];
logic[127:0] rd_data[0:255];

initial begin
  reset <= 1;
  #100;
  reset <= 0;
  for(int i = 0; i <256; i++) begin
    if(i==0) begin
      wr_data[i] <={48'd0,8'h03,16'd255,40'd119119,14'd14};
      rd_data[i] <={48'd0,8'h01,16'd255,40'd119119,14'd14};
    end
    else begin
      wr_data[i] <={i,i,i,i};
      rd_data[i] <={i,i,i,i};
    end
  end

end

reg[3:0] st_fpu=0;
reg dma_end=0;
reg[127:0] data;
always@(posedge clk or posedge reset) begin
  if(reset) begin
  end
  else begin
    case(st_fpu)
      4'd0 : begin
        core_req <= 1'b1;
        core_rwn <= 1'b0;
        core_host_addr <= 40'd119119;
        core_local_addr <= 14'd14;
        core_transfer_length <= 16'd255;
        if(core_ready) begin
          st_fpu <= 4'd1;
        end
      end
      4'd1 : begin
        if(dma_end) begin
          st_fpu <= 4'd2;
        end
        if(core_ack) begin
          core_write_data <= core_write_data + 1;
        end
        core_req <= 1'b0;
        
      end
      4'd2 : begin
        core_req <= 1'b1;
        core_rwn <= 1'b1;
        core_host_addr <= 40'd119119;
        core_local_addr <= 14'd14;
        core_transfer_length <= 16'd255;
        if(core_ready) begin
          st_fpu <= 4'd3;
        end
      end
      4'd3 : begin
        if(dma_end) begin
          st_fpu <= 4'd4;
        end
        core_req <= 1'b0;
        
      end
      4'd2 : begin
      end
      4'd3 : begin
      end
      4'd4 : begin
      end
    endcase
  end

end

reg[3:0] st_dpc=0;
reg[15:0] dma_cnt=0;
reg rd_st;
always@(posedge clk or posedge reset) begin
  if(reset) begin
  end
  else begin
    case(st_dpc)
      4'd0 : begin
        if(dma_req) begin
          dma_resp <= 1'b1;
          st_dpc <= 4'd1;
        end
      end
      4'd1 : begin
        dma_resp <= 1'b1;
        st_dpc <= 4'd2;
      end
      4'd2 : begin
        dma_write_ready <= 1'b1;
        if(dma_write_valid) begin
          if(dma_cnt >=255) begin
            st_dpc <= 4'd3;
            dma_end <= 1'b1;
          end
          else begin
            dma_cnt <= dma_cnt + 1;
          end
        end
        dma_resp <= 1'b0;
      end
      4'd3 : begin
        dma_end <= 1'b0;
        st_dpc <= 4'd4;

      end
      4'd4 : begin
        if(dma_req) begin
          dma_resp <= 1'b1;
          st_dpc <= 4'd5;
        end
      end
      4'd5 : begin
        dma_resp <= 1'b1;
        st_dpc <= 4'd6;
      end
      4'd6 : begin
        dma_write_ready <= 1'b1;
        if(dma_write_valid) begin
            st_dpc <= 4'd7;
            dma_end <= 1'b1;
            dma_cnt <= dma_cnt + 1;
            rd_st <= 1'b1;
        end
        dma_resp <= 1'b0;
      end
      4'd7 : begin
        dma_end <= 1'b0;
        st_dpc <= 4'd7;
        rd_st <= 1'b0;

      end
    endcase

  end
end
reg[15:0] rd_cnt=0;

always@(posedge clk or posedge reset) begin
  if(reset) begin
  end
  else begin
    if(rd_cnt < 16'd257 && rd_cnt >0) begin
      dma_read_valid <= 1'b1;
      dma_read_data <= rd_data[rd_cnt-1];
      rd_cnt <= rd_cnt+16'd1;
    end
    else if(rd_st)begin
      rd_cnt <= 16'd1;
    end
    else begin
      dma_read_valid <= 1'b0;
      dma_read_data <= 0;
      rd_cnt <= 0;
    end
  end
end

endmodule
