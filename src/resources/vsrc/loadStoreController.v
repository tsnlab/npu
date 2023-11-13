
////////////////////////////////////////////////////////////////////////////////
// Company: TSNLAB
// Engineer: junhyuk
//
// Create Date: 2023-10-30
// Design Name: TSN-NPU
// Module Name: dmaController
// Tool versions: vivado 2018.3
// Description:
//    <Description here>
// Revision:
//    0.01 junhyuk - Create
////////////////////////////////////////////////////////////////////////////////
			
module loadStoreController(
  input wire clk,
  input wire rst,

  //***** FPU Core block
  input wire core_req,
  output reg core_ready,
  input wire core_rwn,
  input wire [39:0] core_hostAddr,
  input wire [11:0] core_localAddr,
  input wire [15:0] core_transferLength,
  output wire core_ack,
  input wire [127:0] core_writeData,
  output wire [127:0] core_readData,

  //***** DMA Path Controller block
  output reg dma_req,
  input wire dma_resp,
  output wire dma_write_valid,
  output reg [127:0] dma_write_data,
  input wire dma_write_ready,
  input wire dma_read_valid,
  input wire [127:0] dma_read_data,
  output wire dma_read_ready
);

//***** core fpu control state
localparam [3:0] cfc_idle = 4'd0;
localparam [3:0] cfc_req = 4'd1;
localparam [3:0] cfc_resp = 4'd2;
localparam [3:0] cfc_end = 4'd3;

reg[3:0] cfcon = cfc_idle;
reg data_st;
reg data_done;
//***** fpu reqest process
always@(posedge clk or posedge rst)begin
  if(rst) begin
    dma_req <= 1'b0;
    cfcon <= cfc_idle;
    data_st <= 1'b0;
    core_ready <= 1'b0;

  end
  else begin
    case(cfcon)
      cfc_idle : begin
        if(core_req) begin
          dma_req <= 1'b1;
          cfcon <= cfc_req;
        end
        else begin
          cfcon <= cfc_idle;
        end
      end
      cfc_req : begin
        if(dma_resp) begin
          data_st <= 1'b1;
          dma_req <= 1'b0;
          core_ready <= 1'b1;
          cfcon <= cfc_resp;
        end
        else begin
        end
      end
      cfc_resp : begin
        data_st <= 1'b0; 
        if(core_req) begin
          core_ready <= 1'b1;
        end 
        else begin
          core_ready <= 1'b0;
        end
        if(data_done) begin
          cfcon <= cfc_end;

        end
      end
      cfc_end : begin
        core_ready <= 1'b0;
        data_st <= 1'b0;
        cfcon <= cfc_idle;
      end

    endcase
  end

end

//***** DMA Path controller state
localparam [3:0] dpc_idle = 4'd0;
localparam [3:0] dpc_wr_data0 = 4'd1;
localparam [3:0] dpc_wr_data1 = 4'd2;
localparam [3:0] dpc_rd_data = 4'd3;
localparam [3:0] dpc_end = 4'd4;

reg[3:0] dpcon;
reg ack_en;
reg wr_en;
reg[15:0] dpcon_cnt;
reg[15:0] dpcon_lengh;
reg read_valid;
reg rd_en;

//***** DMA data Path Control
always@(posedge clk or posedge rst) begin
  if(rst) begin
    data_done <= 1'b0;
    ack_en <= 1'b0;
    wr_en <= 1'b0;
    rd_en <= 1'b0;
    dpcon <= dpc_idle;
    dpcon_lengh <= 0;
    dma_write_data <= 0;
    dpcon_cnt <=16'd0;
  end
  else begin
    case(dpcon)
      dpc_idle : begin
        dma_write_data <= 0;
        data_done <= 1'b0;
        wr_en <= 1'b0;
        ack_en <= 1'b0;
        dpcon_cnt <=16'd0;
        rd_en <= 1'b0;
        if(data_st) begin
          if(core_rwn) begin
            dpcon <= dpc_rd_data;
          end
          else begin
            dpcon <= dpc_wr_data0;
            dpcon_lengh <= core_transferLength;
          end
        end
      end
      dpc_wr_data0 : begin
        if(dma_write_ready) begin
          dpcon <= dpc_wr_data1;
          wr_en <= 1'b1;
          dma_write_data <= {48'd0,8'h03,core_transferLength,core_hostAddr,4'b0000,core_localAddr};
        end
        else begin
          wr_en <= 1'b0;
          dma_write_data <= {48'd0,8'h03,core_transferLength,core_hostAddr,4'b0000,core_localAddr};
          dpcon <= dpc_wr_data0;
        end
      end
      dpc_wr_data1 : begin
        if(dpcon_cnt >= dpcon_lengh) begin
          wr_en <= 1'b0;
          dma_write_data <= core_writeData;
          dpcon <= dpc_end;
        end
        else begin
          wr_en <= 1'b1;
          ack_en <= 1'b1;
          dma_write_data <= core_writeData;
          if(dma_write_valid) begin
            dpcon_cnt <= dpcon_cnt + 1;
          end
          else begin
            dpcon_cnt <= dpcon_cnt;
          end
          dpcon <= dpc_wr_data1;
        end

      end
      dpc_rd_data : begin
        if(dma_write_ready) begin
          rd_en <= 1'b1;
          dma_write_data <= {48'd0,8'h01,core_transferLength,core_hostAddr,4'b0000,core_localAddr};
          dpcon <= dpc_end;
        end
      end
      dpc_end : begin
        dpcon_cnt <=16'd0;
        data_done <= 1'b1;
        wr_en <= 1'b0;
        ack_en <= 1'b0;
        rd_en <= 1'b0;
        dpcon <= dpc_idle;
      end
    endcase
    
  end

end

//***** Read Data Path control
always@(posedge clk or posedge rst) begin
  if(rst) begin
    read_valid <= 1'b0;
  end
  else begin
    read_valid <= dma_read_valid;
  end
end


// assign core_ack = ((ack_en && dma_write_ready) || (dma_read_valid && read_valid));
assign core_ack = ((ack_en && dma_write_ready) || (dma_read_valid));
assign dma_write_valid = ((wr_en || rd_en) && dma_write_ready);
assign core_readData = dma_read_data;
assign dma_read_ready = !rst;

endmodule
