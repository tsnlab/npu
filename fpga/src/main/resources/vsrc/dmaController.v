module dmaController
(
    input clk,
    input rst,

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
            dma_req <= 0;
            dma_write_valid <= 0;
            dma_write_data <= 0;
            dma_read_ready <= 0;
        end else begin
            dma_req <= dma_resp;
            dma_write_valid <= dma_read_valid;
            dma_write_data <= dma_read_data;
            dma_read_ready <= dma_write_ready;
        end
    end    
    
endmodule