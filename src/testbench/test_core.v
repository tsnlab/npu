`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/10/31 08:13:29
// Design Name: 
// Module Name: test_core
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


module test_core;

reg clk;      // Clock signal
reg rst;
reg rstn;     // Reset signal

reg [39:0] rocc_if_host_mem_offset = 40'd0; // ROCC Interface signals
reg [15:0] rocc_if_size = 16'd0;
reg [15:0] rocc_if_local_mem_offset = 16'd0;
reg [6:0] rocc_if_funct = 7'd2;
reg rocc_if_cmd_vld;
wire rocc_if_fin;
wire rocc_if_busy;

wire [1:0] bf16_opc;
wire [15:0] bf16_a;
wire [15:0] bf16_b;
wire [15:0] bf16_y;
wire bf16_iv;
wire bf16_or;
wire bf16_ov;
wire bf16_ir;
wire bf16_isSqrt;
wire bf16_kill;

wire dma_req;
reg dma_ready;
wire dma_rwn;
wire [39:0]  dma_hostAddr;
wire [15:0]  dma_localAddr;
wire [15:0]  dma_tansferLength;
wire [127:0] dma_writeData;
reg dma_readData;
reg dma_ack;

wire sram_a_ena;    // SRAM write signals as wire
wire sram_a_wea;
wire [11:0] sram_a_addra;
wire [127:0] sram_a_dina;
wire sram_a_enb;    // SRAM read signals as wire
wire [11:0] sram_a_addrb;
reg [127:0] sram_a_doutb;
wire sram_b_ena;    // SRAM write signals as wire
wire sram_b_wea;
wire [11:0] sram_b_addra;
wire [127:0] sram_b_dina;
wire sram_b_enb;    // SRAM read signals as wire
wire [11:0] sram_b_addrb;
wire [127:0] sram_b_doutb;

reg [127:0] A [0:16*256-1];
  // Clock generation
initial begin
    clk = 0;
    forever begin
        #5 clk = ~clk;
    end
end

// Reset initialization
initial begin
    $readmemh("kernel.txt", A);
    rstn = 0; // Assert reset
    rst = 1;
    // Initialize other inputs here
    // Apply initial inputs
    #10 rstn = 1; // Deassert reset
    rst = 0;
    dma_ready = 1;
    dma_readData = 1;
    dma_ack = 1;
    #10 rocc_if_cmd_vld = 1;
    #10 rocc_if_cmd_vld = 0;
end
  
NPUCore uut0 (
    .clk(clk),
    .rstn(rstn),
    .rocc_if_host_mem_offset(rocc_if_host_mem_offset), // Initialize these signals as needed
    .rocc_if_size(rocc_if_size),
    .rocc_if_local_mem_offset(rocc_if_local_mem_offset),
    .rocc_if_funct(rocc_if_funct),
    .rocc_if_cmd_vld(rocc_if_cmd_vld),
    .rocc_if_fin(rocc_if_fin), // Connect other output signals as needed
    .rocc_if_busy(rocc_if_busy),
    .bf16_opc(bf16_opc),
    .bf16_a(bf16_a),
    .bf16_b(bf16_b),
    .bf16_y(bf16_y),
    .bf16_iv(bf16_iv),
    .bf16_or(bf16_or),
    .bf16_ov(bf16_ov),
    .bf16_ir(bf16_ir),
    .bf16_isSqrt(bf16_isSqrt),
    .bf16_kill(bf16_kill),
    .dma_req(dma_req),
    .dma_ready(dma_ready),
    .dma_rwn(dma_rwn),
    .dma_hostAddr(dma_hostAddr),
    .dma_localAddr(dma_localAddr),
    .dma_tansferLength(dma_tansferLength),
    .dma_writeData(dma_writeData),
    .dma_readData(dma_readData),
    .dma_ack(dma_ack),
    .sram_a_ena(sram_a_ena), // Connect SRAM signals as needed
    .sram_a_wea(sram_a_wea),
    .sram_a_addra(sram_a_addra),
    .sram_a_dina(sram_a_dina),
    .sram_a_enb(sram_a_enb),
    .sram_a_addrb(sram_a_addrb),
    .sram_a_doutb(sram_a_doutb),
    .sram_b_ena(sram_b_ena),
    .sram_b_wea(sram_b_wea),
    .sram_b_addra(sram_b_addra),
    .sram_b_dina(sram_b_dina),
    .sram_b_enb(sram_b_enb),
    .sram_b_addrb(sram_b_addrb),
    .sram_b_doutb(sram_b_doutb)
);
BF16Unit uut1 (
    .clock(clk),
    .reset(rst),
    .io_opc(bf16_opc),
    .io_a(bf16_a),
    .io_b(bf16_b),
    .io_in_valid(bf16_iv),
    .io_out_ready(bf16_or),
    .io_y(bf16_y),
    .io_in_ready(bf16_ir),
    .io_out_valid(bf16_ov),
    .io_isSqrt(bf16_isSqrt),
    .io_kill(bf16_kill)
);




//---- write operation
always @(posedge clk) if(sram_a_ena && sram_a_wea) A[sram_a_addra] <= sram_a_dina;

always @(posedge clk) if(sram_a_enb) sram_a_doutb <= A[sram_a_addrb];

//SRAM uut2(
//    .clka(clk),      // Connect write clock
//    .ena(sram_a_ena), // Connect write enable
//    .wea(sram_a_wea), // Connect write write enable
//    .addra(sram_a_addra), // Connect write address
//    .dina(sram_a_dina), // Connect write data in

//    .clkb(clk), // Connect read clock
//    .enb(sram_a_enb), // Connect read enable
//    .addrb(sram_a_addrb), // Connect read address
//    .doutb(sram_a_doutb) // Connect read data out
//);

SRAM uut3(
    .rstn(rstn),
    .clka(clk),      // Connect write clock
    .ena(sram_b_ena), // Connect write enable
    .wea(sram_b_wea), // Connect write write enable
    .addra(sram_b_addra), // Connect write address
    .dina(sram_b_dina), // Connect write data in

    .clkb(clk), // Connect read clock
    .enb(sram_b_enb), // Connect read enable
    .addrb(sram_b_addrb), // Connect read address
    .doutb(sram_b_doutb) // Connect read data out
);

endmodule
