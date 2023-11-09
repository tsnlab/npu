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

reg [39:0] rocc_if_host_mem_offset = 40'd50; // ROCC Interface signals
reg [15:0] rocc_if_size = 16'd40;
reg [15:0] rocc_if_local_mem_offset = 16'd0;
reg [6:0] rocc_if_funct = 7'd4;
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
wire dma_ready;
wire dma_rwn;
wire [39:0]  dma_hostAddr;
wire [15:0]  dma_localAddr;
wire [15:0]  dma_transferLength;
wire [127:0] dma_writeData;
wire [127:0] dma_readData;
wire dma_ack;

wire sram_a_ena;    // SRAM write signals as wire
wire sram_a_wea;
wire [11:0] sram_a_addra;
wire [127:0] sram_a_dina;
wire sram_a_enb;    // SRAM read signals as wire
wire [11:0] sram_a_addrb;
reg [127:0] sram_a_doutb;

reg [127:0] A [0:16*256-1];

wire         _DMAPathControllerDef_dma_resp_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_write_ready_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_read_valid_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [127:0] _DMAPathControllerDef_dma_read_data_a;	  
wire         _DMAPathControllerDef_dma_resp_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_write_ready_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_read_valid_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [127:0] _DMAPathControllerDef_dma_read_data_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_resp_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_write_ready_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_read_valid_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [127:0] _DMAPathControllerDef_dma_read_data_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_resp_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_write_ready_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_dma_read_valid_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [127:0] _DMAPathControllerDef_dma_read_data_d;	
wire         _NPUTile0Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile0Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire [127:0] _NPUTile0Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile0Def_LoadStoreControllerDef_dma_read_ready;
wire         _NPUTile1Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile1Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire [127:0] _NPUTile1Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile1Def_LoadStoreControllerDef_dma_read_ready;	
wire         _NPUTile2Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile2Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire [127:0] _NPUTile2Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile2Def_LoadStoreControllerDef_dma_read_ready;
wire         _NPUTile3Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile3Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire [127:0] _NPUTile3Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
wire         _NPUTile3Def_LoadStoreControllerDef_dma_read_ready;	

wire         _DMAEngineDef_io_rcc_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
reg         _DMAEngineDef_io_rcd_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
reg [15:0]  _DMAEngineDef_io_rcd_bits_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
reg [127:0] _DMAEngineDef_io_rcd_bits_data=0;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
reg [15:0]  _DMAEngineDef_io_rcd_bits_length;
wire [39:0]  _DMAPathControllerDef_rcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [15:0]  _DMAPathControllerDef_rcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [15:0]  _DMAPathControllerDef_rcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_rcc_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_rcd_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [39:0]  _DMAPathControllerDef_wcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [15:0]  _DMAPathControllerDef_wcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [15:0]  _DMAPathControllerDef_wcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire [127:0] _DMAPathControllerDef_wcc_write_data;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
wire         _DMAPathControllerDef_wcc_valid;	


  // Clock generation
initial begin
    clk = 1;
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
//    dma_readData = 0;
//    dma_ack = 0;
//    dma_ready = 1;
    #10 rocc_if_cmd_vld = 1;
    #10 rocc_if_cmd_vld = 0;
//    #50 dma_readData = 128'h1234;
//    dma_ack = 1;
//    #10 dma_readData = 128'h5678;
//    #10 dma_readData = 128'h9abc;
//    #10 dma_readData = 128'hdef0;
//    #10 dma_ack = 0;
end

//always @(posedge clk) if(dma_req) dma_ready <= 1; else dma_ready <= 0;

assign _DMAEngineDef_io_rcc_ready = 1'd1;

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
    .dma_transferLength(dma_transferLength),
    .dma_writeData(dma_writeData),
    .dma_readData(dma_readData),
    .dma_ack(dma_ack),
    .sram_ena(sram_a_ena), // Connect SRAM signals as needed
    .sram_wea(sram_a_wea),
    .sram_addra(sram_a_addra),
    .sram_dina(sram_a_dina),
    .sram_enb(sram_a_enb),
    .sram_addrb(sram_a_addrb),
    .sram_doutb(sram_a_doutb)
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

loadStoreController NPUTile1Def_LoadStoreControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .clk                 (clk),
    .rst                 (rst),
    .core_req            (dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_rwn            (dma_rwn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_hostAddr       (dma_hostAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_localAddr      (dma_localAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32, :106:50]
    .core_transferLength (dma_transferLength),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_writeData      (dma_writeData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dma_resp            (_DMAPathControllerDef_dma_resp_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_write_ready     (_DMAPathControllerDef_dma_write_ready_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_valid      (_DMAPathControllerDef_dma_read_valid_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_data       (_DMAPathControllerDef_dma_read_data_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .core_ready          (dma_ready),
    .core_readData       (dma_readData),
    .core_ack            (dma_ack),
    .dma_req             (_NPUTile0Def_LoadStoreControllerDef_dma_req),
    .dma_write_valid     (_NPUTile0Def_LoadStoreControllerDef_dma_write_valid),
    .dma_write_data      (_NPUTile0Def_LoadStoreControllerDef_dma_write_data),
    .dma_read_ready      (_NPUTile0Def_LoadStoreControllerDef_dma_read_ready)
);

  
DMAPathController DMAPathControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .risc_clk          (clk),
    .fpu_clk           (clk),
    .reset             (rst),
    .rcc_ready         (_DMAEngineDef_io_rcc_ready),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
    .rcd_dpram_addr    (_DMAEngineDef_io_rcd_bits_dpram_addr),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
    .rcd_read_data     (_DMAEngineDef_io_rcd_bits_data),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
    .rcd_length        (_DMAEngineDef_io_rcd_bits_length),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
    .rcd_valid         (_DMAEngineDef_io_rcd_valid),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
    .wcc_ready         (1'h1),	// @[generators/rocket-chip/src/main/scala/tile/LazyRoCC.scala:71:14]
    .dma_req_a         (_NPUTile0Def_LoadStoreControllerDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_valid_a (_NPUTile0Def_LoadStoreControllerDef_dma_write_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_data_a  (_NPUTile0Def_LoadStoreControllerDef_dma_write_data),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_read_ready_a  (_NPUTile0Def_LoadStoreControllerDef_dma_read_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_req_b         (_NPUTile1Def_LoadStoreControllerDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_valid_b (_NPUTile1Def_LoadStoreControllerDef_dma_write_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_data_b  (_NPUTile1Def_LoadStoreControllerDef_dma_write_data),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_read_ready_b  (_NPUTile1Def_LoadStoreControllerDef_dma_read_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_req_c         (_NPUTile2Def_LoadStoreControllerDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_valid_c (_NPUTile2Def_LoadStoreControllerDef_dma_write_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_data_c  (_NPUTile2Def_LoadStoreControllerDef_dma_write_data),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_read_ready_c  (_NPUTile2Def_LoadStoreControllerDef_dma_read_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_req_d         (_NPUTile3Def_LoadStoreControllerDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_valid_d (_NPUTile3Def_LoadStoreControllerDef_dma_write_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_write_data_d  (_NPUTile3Def_LoadStoreControllerDef_dma_write_data),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_read_ready_d  (_NPUTile3Def_LoadStoreControllerDef_dma_read_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .rcc_dram_addr     (_DMAPathControllerDef_rcc_dram_addr),
    .rcc_dpram_addr    (_DMAPathControllerDef_rcc_dpram_addr),
    .rcc_length        (_DMAPathControllerDef_rcc_length),
    .rcc_valid         (_DMAPathControllerDef_rcc_valid),
    .rcd_ready         (_DMAPathControllerDef_rcd_ready),
    .wcc_dram_addr     (_DMAPathControllerDef_wcc_dram_addr),
    .wcc_dpram_addr    (_DMAPathControllerDef_wcc_dpram_addr),
    .wcc_length        (_DMAPathControllerDef_wcc_length),
    .wcc_write_data    (_DMAPathControllerDef_wcc_write_data),
    .wcc_valid         (_DMAPathControllerDef_wcc_valid),
    .dma_resp_a        (_DMAPathControllerDef_dma_resp_a),
    .dma_write_ready_a (_DMAPathControllerDef_dma_write_ready_a),
    .dma_read_valid_a  (_DMAPathControllerDef_dma_read_valid_a),
    .dma_read_data_a   (_DMAPathControllerDef_dma_read_data_a),
    .dma_resp_b        (_DMAPathControllerDef_dma_resp_b),
    .dma_write_ready_b (_DMAPathControllerDef_dma_write_ready_b),
    .dma_read_valid_b  (_DMAPathControllerDef_dma_read_valid_b),
    .dma_read_data_b   (_DMAPathControllerDef_dma_read_data_b),
    .dma_resp_c        (_DMAPathControllerDef_dma_resp_c),
    .dma_write_ready_c (_DMAPathControllerDef_dma_write_ready_c),
    .dma_read_valid_c  (_DMAPathControllerDef_dma_read_valid_c),
    .dma_read_data_c   (_DMAPathControllerDef_dma_read_data_c),
    .dma_resp_d        (_DMAPathControllerDef_dma_resp_d),
    .dma_write_ready_d (_DMAPathControllerDef_dma_write_ready_d),
    .dma_read_valid_d  (_DMAPathControllerDef_dma_read_valid_d),
    .dma_read_data_d   (_DMAPathControllerDef_dma_read_data_d)
);

//---- write operation
always @(posedge clk) if(sram_a_ena && sram_a_wea) A[sram_a_addra] <= sram_a_dina;

always @(posedge clk) if(sram_a_enb) sram_a_doutb <= A[sram_a_addrb];


reg[15:0] rcc_len;
reg[39:0] rcc_dram;
reg[15:0] rcc_dpram;
reg[15:0] rcc_cnt =0;
reg set=0;


always@(posedge clk) begin
    if(_DMAPathControllerDef_rcc_valid) begin
        set <= 1;
        rcc_len <= _DMAPathControllerDef_rcc_length;
        rcc_dram <= _DMAPathControllerDef_rcc_dram_addr;
        rcc_dpram <= _DMAPathControllerDef_rcc_dpram_addr;
    end
    if(set==1) begin
        if(rcc_cnt >=rcc_len) begin
            _DMAEngineDef_io_rcd_valid <= 1'b0;
            _DMAEngineDef_io_rcd_bits_length <= rcc_len;
            _DMAEngineDef_io_rcd_bits_dpram_addr <= rcc_dpram;
            _DMAEngineDef_io_rcd_bits_data <= 0;
            rcc_cnt <= 0;
            set <= 0;
        
        end
        else begin
            _DMAEngineDef_io_rcd_valid <= 1'b1;
            _DMAEngineDef_io_rcd_bits_length <= rcc_len;
            _DMAEngineDef_io_rcd_bits_dpram_addr <= rcc_dpram;
            _DMAEngineDef_io_rcd_bits_data <= _DMAEngineDef_io_rcd_bits_data + 1;
            rcc_cnt <= rcc_cnt + 1;
        end
    end
    else begin
        _DMAEngineDef_io_rcd_valid <= 1'b0;
        _DMAEngineDef_io_rcd_bits_length <= rcc_len;
        _DMAEngineDef_io_rcd_bits_dpram_addr <= rcc_dpram;
        _DMAEngineDef_io_rcd_bits_data <= 0;
        rcc_cnt <= 0;
    end


end


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

//SRAM uut3(
//    .rstn(rstn),
//    .clka(clk),      // Connect write clock
//    .ena(sram_b_ena), // Connect write enable
//    .wea(sram_b_wea), // Connect write write enable
//    .addra(sram_b_addra), // Connect write address
//    .dina(sram_b_dina), // Connect write data in

//    .clkb(clk), // Connect read clock
//    .enb(sram_b_enb), // Connect read enable
//    .addrb(sram_b_addrb), // Connect read address
//    .doutb(sram_b_doutb) // Connect read data out
//);

endmodule
