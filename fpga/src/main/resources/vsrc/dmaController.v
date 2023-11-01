module dmaController
(
    input clk,
    input rst,

    input core_req,
    output reg core_ready,
    input core_rwn,
    input [39:0] core_hostAddr,
    input [13:0] core_localAddr,
    input [15:0] core_tansferLength,
    input [127:0] core_writeData,
    output reg [127:0] core_readData,
    output reg core_ack,

    output reg dma_req,
    input dma_resp,
    output reg dma_write_valid,
    output reg [127:0] dma_write_data,
    input dma_write_ready,
    input dma_read_valid,
    input [127:0] dma_read_data,
    output reg dma_read_ready
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            core_ready <= 0;
            core_readData <= 0;
            core_ack <= 0;
            dma_req <= 0;
            dma_write_valid <= 0;
            dma_write_data <= 0;
            dma_read_ready <= 0;
        end else begin
            core_ready <= 0;
            core_readData <= 0;
            core_ack <= 0;
            dma_req <= 0;
            dma_write_valid <= 0;
            dma_write_data <= 0;
            dma_read_ready <= 0;
        end
    end    
    
endmodule