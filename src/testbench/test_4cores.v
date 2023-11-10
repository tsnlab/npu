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

reg [39:0] rocc_if_host_mem_offset_tile0 = 40'd128; // ROCC Interface signals
reg [15:0] rocc_if_size_tile0 = 16'd8;
reg [15:0] rocc_if_local_mem_offset_tile0 = 16'd0;
reg [39:0] rocc_if_host_mem_offset_tile1 = 40'd256; // ROCC Interface signals
reg [15:0] rocc_if_size_tile1 = 16'd8;
reg [15:0] rocc_if_local_mem_offset_tile1 = 16'd0;
reg [39:0] rocc_if_host_mem_offset_tile2 = 40'd384; // ROCC Interface signals
reg [15:0] rocc_if_size_tile2 = 16'd8;
reg [15:0] rocc_if_local_mem_offset_tile2 = 16'd0;
reg [39:0] rocc_if_host_mem_offset_tile3 = 40'd512; // ROCC Interface signals
reg [15:0] rocc_if_size_tile3 = 16'd8;
reg [15:0] rocc_if_local_mem_offset_tile3 = 16'd0;
reg [6:0] rocc_if_funct = 7'd3;
reg rocc_if_cmd_vld;
//wire rocc_if_fin_tile0;
//wire rocc_if_busy_tile0;
//wire rocc_if_fin_tile1;
//wire rocc_if_busy_tile1;
//wire rocc_if_fin_tile2;
//wire rocc_if_busy_tile2;
//wire rocc_if_fin_tile3;
//wire rocc_if_busy_tile3;

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
wire [11:0]  dma_localAddr;
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

//wire         _DMAPathControllerDef_dma_resp_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_write_ready_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_read_valid_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [127:0] _DMAPathControllerDef_dma_read_data_a;	  
//wire         _DMAPathControllerDef_dma_resp_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_write_ready_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_read_valid_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [127:0] _DMAPathControllerDef_dma_read_data_b;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_resp_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_write_ready_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_read_valid_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [127:0] _DMAPathControllerDef_dma_read_data_c;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_resp_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_write_ready_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_dma_read_valid_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [127:0] _DMAPathControllerDef_dma_read_data_d;	
//wire         _NPUTile0Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile0Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire [127:0] _NPUTile0Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile0Def_LoadStoreControllerDef_dma_read_ready;
//wire         _NPUTile1Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile1Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire [127:0] _NPUTile1Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile1Def_LoadStoreControllerDef_dma_read_ready;	
//wire         _NPUTile2Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile2Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire [127:0] _NPUTile2Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile2Def_LoadStoreControllerDef_dma_read_ready;
//wire         _NPUTile3Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile3Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire [127:0] _NPUTile3Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
//wire         _NPUTile3Def_LoadStoreControllerDef_dma_read_ready;	


//wire         _DMAEngineDef_io_rcc_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
//reg         _DMAEngineDef_io_rcd_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
//reg [15:0]  _DMAEngineDef_io_rcd_bits_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
//reg [127:0] _DMAEngineDef_io_rcd_bits_data=0;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
//reg [15:0]  _DMAEngineDef_io_rcd_bits_length;
//wire [39:0]  _DMAPathControllerDef_rcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [15:0]  _DMAPathControllerDef_rcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [15:0]  _DMAPathControllerDef_rcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_rcc_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_rcd_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [39:0]  _DMAPathControllerDef_wcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [15:0]  _DMAPathControllerDef_wcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [15:0]  _DMAPathControllerDef_wcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire [127:0] _DMAPathControllerDef_wcc_write_data;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
//wire         _DMAPathControllerDef_wcc_valid;	

  wire [39:0]  _DMAPathControllerDef_rcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [15:0]  _DMAPathControllerDef_rcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [15:0]  _DMAPathControllerDef_rcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_rcc_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_rcd_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [39:0]  _DMAPathControllerDef_wcc_dram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [15:0]  _DMAPathControllerDef_wcc_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [15:0]  _DMAPathControllerDef_wcc_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [127:0] _DMAPathControllerDef_wcc_write_data;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_wcc_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_dma_resp_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_dma_write_ready_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAPathControllerDef_dma_read_valid_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire [127:0] _DMAPathControllerDef_dma_read_data_a;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
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
  wire [127:0] _DMAPathControllerDef_dma_read_data_d;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
  wire         _DMAEngineDef_io_rcc_ready;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  reg         _DMAEngineDef_io_rcd_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  reg [15:0]  _DMAEngineDef_io_rcd_bits_dpram_addr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  reg [127:0] _DMAEngineDef_io_rcd_bits_data;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  reg [15:0]  _DMAEngineDef_io_rcd_bits_length;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire         _DMAEngineDef_io_tlb_0_req_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire [39:0]  _DMAEngineDef_io_tlb_0_req_bits_tlb_req_vaddr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire         _DMAEngineDef_io_tlb_0_req_bits_status_debug;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire         _DMAEngineDef_io_tlb_1_req_valid;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire [39:0]  _DMAEngineDef_io_tlb_1_req_bits_tlb_req_vaddr;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire         _DMAEngineDef_io_tlb_1_req_bits_status_debug;	// @[fpga/src/main/scala/nexysvideo/NPU.scala:103:34]
  wire         _NPUTile3Def_LoadStoreControllerDef_core_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile3Def_LoadStoreControllerDef_core_readData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile3Def_LoadStoreControllerDef_core_ack;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile3Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile3Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile3Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile3Def_LoadStoreControllerDef_dma_read_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile3Def_SRAMADef_doutb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
  wire [15:0]  _NPUTile3Def_BF16UnitDef_io_y;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile3Def_BF16UnitDef_io_in_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile3Def_BF16UnitDef_io_out_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile3Def_NPUCoreDef_rocc_if_fin;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_rocc_if_busy;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [1:0]   _NPUTile3Def_NPUCoreDef_bf16_opc;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile3Def_NPUCoreDef_bf16_a;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile3Def_NPUCoreDef_bf16_b;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_bf16_iv;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_bf16_or;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_bf16_isSqrt;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_bf16_kill;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_dma_rwn;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [39:0]  _NPUTile3Def_NPUCoreDef_dma_hostAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile3Def_NPUCoreDef_dma_localAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile3Def_NPUCoreDef_dma_transferLength;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile3Def_NPUCoreDef_dma_writeData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_sram_ena;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_sram_wea;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile3Def_NPUCoreDef_sram_addra;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile3Def_NPUCoreDef_sram_dina;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile3Def_NPUCoreDef_sram_enb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile3Def_NPUCoreDef_sram_addrb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_LoadStoreControllerDef_core_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile2Def_LoadStoreControllerDef_core_readData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile2Def_LoadStoreControllerDef_core_ack;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile2Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile2Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile2Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile2Def_LoadStoreControllerDef_dma_read_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile2Def_SRAMADef_doutb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
  wire [15:0]  _NPUTile2Def_BF16UnitDef_io_y;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile2Def_BF16UnitDef_io_in_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile2Def_BF16UnitDef_io_out_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile2Def_NPUCoreDef_rocc_if_fin;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_rocc_if_busy;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [1:0]   _NPUTile2Def_NPUCoreDef_bf16_opc;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile2Def_NPUCoreDef_bf16_a;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile2Def_NPUCoreDef_bf16_b;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_bf16_iv;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_bf16_or;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_bf16_isSqrt;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_bf16_kill;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_dma_rwn;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [39:0]  _NPUTile2Def_NPUCoreDef_dma_hostAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile2Def_NPUCoreDef_dma_localAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile2Def_NPUCoreDef_dma_transferLength;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile2Def_NPUCoreDef_dma_writeData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_sram_ena;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_sram_wea;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile2Def_NPUCoreDef_sram_addra;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile2Def_NPUCoreDef_sram_dina;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile2Def_NPUCoreDef_sram_enb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile2Def_NPUCoreDef_sram_addrb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_LoadStoreControllerDef_core_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile1Def_LoadStoreControllerDef_core_readData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile1Def_LoadStoreControllerDef_core_ack;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile1Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile1Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile1Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile1Def_LoadStoreControllerDef_dma_read_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile1Def_SRAMADef_doutb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
  wire [15:0]  _NPUTile1Def_BF16UnitDef_io_y;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile1Def_BF16UnitDef_io_in_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile1Def_BF16UnitDef_io_out_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile1Def_NPUCoreDef_rocc_if_fin;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_rocc_if_busy;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [1:0]   _NPUTile1Def_NPUCoreDef_bf16_opc;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile1Def_NPUCoreDef_bf16_a;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile1Def_NPUCoreDef_bf16_b;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_bf16_iv;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_bf16_or;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_bf16_isSqrt;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_bf16_kill;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_dma_rwn;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [39:0]  _NPUTile1Def_NPUCoreDef_dma_hostAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile1Def_NPUCoreDef_dma_localAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile1Def_NPUCoreDef_dma_transferLength;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile1Def_NPUCoreDef_dma_writeData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_sram_ena;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_sram_wea;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile1Def_NPUCoreDef_sram_addra;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile1Def_NPUCoreDef_sram_dina;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile1Def_NPUCoreDef_sram_enb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile1Def_NPUCoreDef_sram_addrb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_LoadStoreControllerDef_core_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile0Def_LoadStoreControllerDef_core_readData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile0Def_LoadStoreControllerDef_core_ack;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile0Def_LoadStoreControllerDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile0Def_LoadStoreControllerDef_dma_write_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile0Def_LoadStoreControllerDef_dma_write_data;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire         _NPUTile0Def_LoadStoreControllerDef_dma_read_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
  wire [127:0] _NPUTile0Def_SRAMADef_doutb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
  wire [15:0]  _NPUTile0Def_BF16UnitDef_io_y;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile0Def_BF16UnitDef_io_in_ready;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile0Def_BF16UnitDef_io_out_valid;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
  wire         _NPUTile0Def_NPUCoreDef_rocc_if_fin;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_rocc_if_busy;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [1:0]   _NPUTile0Def_NPUCoreDef_bf16_opc;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile0Def_NPUCoreDef_bf16_a;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile0Def_NPUCoreDef_bf16_b;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_bf16_iv;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_bf16_or;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_bf16_isSqrt;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_bf16_kill;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_dma_req;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_dma_rwn;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [39:0]  _NPUTile0Def_NPUCoreDef_dma_hostAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile0Def_NPUCoreDef_dma_localAddr;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [15:0]  _NPUTile0Def_NPUCoreDef_dma_transferLength;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile0Def_NPUCoreDef_dma_writeData;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_sram_ena;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_sram_wea;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile0Def_NPUCoreDef_sram_addra;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [127:0] _NPUTile0Def_NPUCoreDef_sram_dina;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire         _NPUTile0Def_NPUCoreDef_sram_enb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
  wire [11:0]  _NPUTile0Def_NPUCoreDef_sram_addrb;	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]


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

NPUCore uut0 (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clk                      (clk),
    .rstn                     (rstn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:42:31]
    .rocc_if_host_mem_offset  (rocc_if_host_mem_offset_tile0),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :150:{51,64}]
    .rocc_if_size             (rocc_if_size_tile0),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :151:{40,53}]
    .rocc_if_local_mem_offset (rocc_if_local_mem_offset_tile0),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :152:{52,65}]
    .rocc_if_funct            (rocc_if_funct),	// @[src/main/scala/chisel3/util/Decoupled.scala:376:21]
    .rocc_if_cmd_vld          (rocc_if_cmd_vld),	// @[src/main/scala/chisel3/util/Decoupled.scala:52:35]
    .bf16_y                   (_NPUTile0Def_BF16UnitDef_io_y),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ov                  (_NPUTile0Def_BF16UnitDef_io_out_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ir                  (_NPUTile0Def_BF16UnitDef_io_in_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .dma_ready                (_NPUTile0Def_LoadStoreControllerDef_core_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_readData             (_NPUTile0Def_LoadStoreControllerDef_core_readData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_ack                  (_NPUTile0Def_LoadStoreControllerDef_core_ack),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .sram_doutb               (_NPUTile0Def_SRAMADef_doutb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .rocc_if_fin              (_NPUTile0Def_NPUCoreDef_rocc_if_fin),
    .rocc_if_busy             (_NPUTile0Def_NPUCoreDef_rocc_if_busy),
    .bf16_opc                 (_NPUTile0Def_NPUCoreDef_bf16_opc),
    .bf16_a                   (_NPUTile0Def_NPUCoreDef_bf16_a),
    .bf16_b                   (_NPUTile0Def_NPUCoreDef_bf16_b),
    .bf16_iv                  (_NPUTile0Def_NPUCoreDef_bf16_iv),
    .bf16_or                  (_NPUTile0Def_NPUCoreDef_bf16_or),
    .bf16_isSqrt              (_NPUTile0Def_NPUCoreDef_bf16_isSqrt),
    .bf16_kill                (_NPUTile0Def_NPUCoreDef_bf16_kill),
    .dma_req                  (_NPUTile0Def_NPUCoreDef_dma_req),
    .dma_rwn                  (_NPUTile0Def_NPUCoreDef_dma_rwn),
    .dma_hostAddr             (_NPUTile0Def_NPUCoreDef_dma_hostAddr),
    .dma_localAddr            (_NPUTile0Def_NPUCoreDef_dma_localAddr),
    .dma_transferLength       (_NPUTile0Def_NPUCoreDef_dma_transferLength),
    .dma_writeData            (_NPUTile0Def_NPUCoreDef_dma_writeData),
    .sram_ena                 (_NPUTile0Def_NPUCoreDef_sram_ena),
    .sram_wea                 (_NPUTile0Def_NPUCoreDef_sram_wea),
    .sram_addra               (_NPUTile0Def_NPUCoreDef_sram_addra),
    .sram_dina                (_NPUTile0Def_NPUCoreDef_sram_dina),
    .sram_enb                 (_NPUTile0Def_NPUCoreDef_sram_enb),
    .sram_addrb               (_NPUTile0Def_NPUCoreDef_sram_addrb)
);
BF16Unit NPUTile0Def_BF16UnitDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .clock        (clk),
    .reset        (rst),
    .io_opc       (_NPUTile0Def_NPUCoreDef_bf16_opc),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_a         (_NPUTile0Def_NPUCoreDef_bf16_a),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_b         (_NPUTile0Def_NPUCoreDef_bf16_b),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_in_valid  (_NPUTile0Def_NPUCoreDef_bf16_iv),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_out_ready (_NPUTile0Def_NPUCoreDef_bf16_or),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_isSqrt    (_NPUTile0Def_NPUCoreDef_bf16_isSqrt),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_kill      (_NPUTile0Def_NPUCoreDef_bf16_kill),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_y         (_NPUTile0Def_BF16UnitDef_io_y),
    .io_in_ready  (_NPUTile0Def_BF16UnitDef_io_in_ready),
    .io_out_valid (_NPUTile0Def_BF16UnitDef_io_out_valid)
);
SRAM NPUTile0Def_SRAMADef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .clka  (clk),
    .ena   (_NPUTile0Def_NPUCoreDef_sram_ena),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .wea   (_NPUTile0Def_NPUCoreDef_sram_wea),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addra (_NPUTile0Def_NPUCoreDef_sram_addra),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dina  (_NPUTile0Def_NPUCoreDef_sram_dina),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clkb  (clk),
    .enb   (_NPUTile0Def_NPUCoreDef_sram_enb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addrb (_NPUTile0Def_NPUCoreDef_sram_addrb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .doutb (_NPUTile0Def_SRAMADef_doutb)
);

    loadStoreController NPUTile0Def_LoadStoreControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .clk                 (clk),
    .rst                 (rst),
    .core_req            (_NPUTile0Def_NPUCoreDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_rwn            (_NPUTile0Def_NPUCoreDef_dma_rwn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_hostAddr       (_NPUTile0Def_NPUCoreDef_dma_hostAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_localAddr      (_NPUTile0Def_NPUCoreDef_dma_localAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_transferLength (_NPUTile0Def_NPUCoreDef_dma_transferLength),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_writeData      (_NPUTile0Def_NPUCoreDef_dma_writeData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dma_resp            (_DMAPathControllerDef_dma_resp_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_write_ready     (_DMAPathControllerDef_dma_write_ready_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_valid      (_DMAPathControllerDef_dma_read_valid_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_data       (_DMAPathControllerDef_dma_read_data_a),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .core_ready          (_NPUTile0Def_LoadStoreControllerDef_core_ready),
    .core_readData       (_NPUTile0Def_LoadStoreControllerDef_core_readData),
    .core_ack            (_NPUTile0Def_LoadStoreControllerDef_core_ack),
    .dma_req             (_NPUTile0Def_LoadStoreControllerDef_dma_req),
    .dma_write_valid     (_NPUTile0Def_LoadStoreControllerDef_dma_write_valid),
    .dma_write_data      (_NPUTile0Def_LoadStoreControllerDef_dma_write_data),
    .dma_read_ready      (_NPUTile0Def_LoadStoreControllerDef_dma_read_ready)
    );

NPUCore NPUTile1Def_NPUCoreDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clk                      (clk),
    .rstn                     (rstn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:42:31]
    .rocc_if_host_mem_offset  (rocc_if_host_mem_offset_tile1),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :160:{51,64}]
    .rocc_if_size             (rocc_if_size_tile1),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :161:{40,53}]
    .rocc_if_local_mem_offset (rocc_if_local_mem_offset_tile1),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :162:{52,65}]
    .rocc_if_funct            (rocc_if_funct),	// @[src/main/scala/chisel3/util/Decoupled.scala:376:21]
    .rocc_if_cmd_vld          (rocc_if_cmd_vld),	// @[src/main/scala/chisel3/util/Decoupled.scala:52:35]
    .bf16_y                   (_NPUTile1Def_BF16UnitDef_io_y),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ov                  (_NPUTile1Def_BF16UnitDef_io_out_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ir                  (_NPUTile1Def_BF16UnitDef_io_in_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .dma_ready                (_NPUTile1Def_LoadStoreControllerDef_core_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_readData             (_NPUTile1Def_LoadStoreControllerDef_core_readData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_ack                  (_NPUTile1Def_LoadStoreControllerDef_core_ack),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .sram_doutb               (_NPUTile1Def_SRAMADef_doutb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .rocc_if_fin              (_NPUTile1Def_NPUCoreDef_rocc_if_fin),
    .rocc_if_busy             (_NPUTile1Def_NPUCoreDef_rocc_if_busy),
    .bf16_opc                 (_NPUTile1Def_NPUCoreDef_bf16_opc),
    .bf16_a                   (_NPUTile1Def_NPUCoreDef_bf16_a),
    .bf16_b                   (_NPUTile1Def_NPUCoreDef_bf16_b),
    .bf16_iv                  (_NPUTile1Def_NPUCoreDef_bf16_iv),
    .bf16_or                  (_NPUTile1Def_NPUCoreDef_bf16_or),
    .bf16_isSqrt              (_NPUTile1Def_NPUCoreDef_bf16_isSqrt),
    .bf16_kill                (_NPUTile1Def_NPUCoreDef_bf16_kill),
    .dma_req                  (_NPUTile1Def_NPUCoreDef_dma_req),
    .dma_rwn                  (_NPUTile1Def_NPUCoreDef_dma_rwn),
    .dma_hostAddr             (_NPUTile1Def_NPUCoreDef_dma_hostAddr),
    .dma_localAddr            (_NPUTile1Def_NPUCoreDef_dma_localAddr),
    .dma_transferLength       (_NPUTile1Def_NPUCoreDef_dma_transferLength),
    .dma_writeData            (_NPUTile1Def_NPUCoreDef_dma_writeData),
    .sram_ena                 (_NPUTile1Def_NPUCoreDef_sram_ena),
    .sram_wea                 (_NPUTile1Def_NPUCoreDef_sram_wea),
    .sram_addra               (_NPUTile1Def_NPUCoreDef_sram_addra),
    .sram_dina                (_NPUTile1Def_NPUCoreDef_sram_dina),
    .sram_enb                 (_NPUTile1Def_NPUCoreDef_sram_enb),
    .sram_addrb               (_NPUTile1Def_NPUCoreDef_sram_addrb)
);
  BF16Unit NPUTile1Def_BF16UnitDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .clock        (clk),
    .reset        (rst),
    .io_opc       (_NPUTile1Def_NPUCoreDef_bf16_opc),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_a         (_NPUTile1Def_NPUCoreDef_bf16_a),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_b         (_NPUTile1Def_NPUCoreDef_bf16_b),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_in_valid  (_NPUTile1Def_NPUCoreDef_bf16_iv),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_out_ready (_NPUTile1Def_NPUCoreDef_bf16_or),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_isSqrt    (_NPUTile1Def_NPUCoreDef_bf16_isSqrt),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_kill      (_NPUTile1Def_NPUCoreDef_bf16_kill),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_y         (_NPUTile1Def_BF16UnitDef_io_y),
    .io_in_ready  (_NPUTile1Def_BF16UnitDef_io_in_ready),
    .io_out_valid (_NPUTile1Def_BF16UnitDef_io_out_valid)
  );
  SRAM NPUTile1Def_SRAMADef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .clka  (clk),
    .ena   (_NPUTile1Def_NPUCoreDef_sram_ena),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .wea   (_NPUTile1Def_NPUCoreDef_sram_wea),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addra (_NPUTile1Def_NPUCoreDef_sram_addra),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dina  (_NPUTile1Def_NPUCoreDef_sram_dina),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clkb  (clk),
    .enb   (_NPUTile1Def_NPUCoreDef_sram_enb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addrb (_NPUTile1Def_NPUCoreDef_sram_addrb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .doutb (_NPUTile1Def_SRAMADef_doutb)
  );
  loadStoreController NPUTile1Def_LoadStoreControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .clk                 (clk),
    .rst                 (rst),
    .core_req            (_NPUTile1Def_NPUCoreDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_rwn            (_NPUTile1Def_NPUCoreDef_dma_rwn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_hostAddr       (_NPUTile1Def_NPUCoreDef_dma_hostAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_localAddr      (_NPUTile1Def_NPUCoreDef_dma_localAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_transferLength (_NPUTile1Def_NPUCoreDef_dma_transferLength),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_writeData      (_NPUTile1Def_NPUCoreDef_dma_writeData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dma_resp            (_DMAPathControllerDef_dma_resp_b),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_write_ready     (_DMAPathControllerDef_dma_write_ready_b),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_valid      (_DMAPathControllerDef_dma_read_valid_b),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_data       (_DMAPathControllerDef_dma_read_data_b),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .core_ready          (_NPUTile1Def_LoadStoreControllerDef_core_ready),
    .core_readData       (_NPUTile1Def_LoadStoreControllerDef_core_readData),
    .core_ack            (_NPUTile1Def_LoadStoreControllerDef_core_ack),
    .dma_req             (_NPUTile1Def_LoadStoreControllerDef_dma_req),
    .dma_write_valid     (_NPUTile1Def_LoadStoreControllerDef_dma_write_valid),
    .dma_write_data      (_NPUTile1Def_LoadStoreControllerDef_dma_write_data),
    .dma_read_ready      (_NPUTile1Def_LoadStoreControllerDef_dma_read_ready)
  );
  NPUCore NPUTile2Def_NPUCoreDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clk                      (clk),
    .rstn                     (rstn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:42:31]
    .rocc_if_host_mem_offset  (rocc_if_host_mem_offset_tile2),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :170:{51,64}]
    .rocc_if_size             (rocc_if_size_tile2),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :171:{40,53}]
    .rocc_if_local_mem_offset (rocc_if_local_mem_offset_tile2),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :172:{52,65}]
    .rocc_if_funct            (rocc_if_funct),	// @[src/main/scala/chisel3/util/Decoupled.scala:376:21]
    .rocc_if_cmd_vld          (rocc_if_cmd_vld),	// @[src/main/scala/chisel3/util/Decoupled.scala:52:35]
    .bf16_y                   (_NPUTile2Def_BF16UnitDef_io_y),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ov                  (_NPUTile2Def_BF16UnitDef_io_out_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ir                  (_NPUTile2Def_BF16UnitDef_io_in_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .dma_ready                (_NPUTile2Def_LoadStoreControllerDef_core_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_readData             (_NPUTile2Def_LoadStoreControllerDef_core_readData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_ack                  (_NPUTile2Def_LoadStoreControllerDef_core_ack),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .sram_doutb               (_NPUTile2Def_SRAMADef_doutb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .rocc_if_fin              (_NPUTile2Def_NPUCoreDef_rocc_if_fin),
    .rocc_if_busy             (_NPUTile2Def_NPUCoreDef_rocc_if_busy),
    .bf16_opc                 (_NPUTile2Def_NPUCoreDef_bf16_opc),
    .bf16_a                   (_NPUTile2Def_NPUCoreDef_bf16_a),
    .bf16_b                   (_NPUTile2Def_NPUCoreDef_bf16_b),
    .bf16_iv                  (_NPUTile2Def_NPUCoreDef_bf16_iv),
    .bf16_or                  (_NPUTile2Def_NPUCoreDef_bf16_or),
    .bf16_isSqrt              (_NPUTile2Def_NPUCoreDef_bf16_isSqrt),
    .bf16_kill                (_NPUTile2Def_NPUCoreDef_bf16_kill),
    .dma_req                  (_NPUTile2Def_NPUCoreDef_dma_req),
    .dma_rwn                  (_NPUTile2Def_NPUCoreDef_dma_rwn),
    .dma_hostAddr             (_NPUTile2Def_NPUCoreDef_dma_hostAddr),
    .dma_localAddr            (_NPUTile2Def_NPUCoreDef_dma_localAddr),
    .dma_transferLength       (_NPUTile2Def_NPUCoreDef_dma_transferLength),
    .dma_writeData            (_NPUTile2Def_NPUCoreDef_dma_writeData),
    .sram_ena                 (_NPUTile2Def_NPUCoreDef_sram_ena),
    .sram_wea                 (_NPUTile2Def_NPUCoreDef_sram_wea),
    .sram_addra               (_NPUTile2Def_NPUCoreDef_sram_addra),
    .sram_dina                (_NPUTile2Def_NPUCoreDef_sram_dina),
    .sram_enb                 (_NPUTile2Def_NPUCoreDef_sram_enb),
    .sram_addrb               (_NPUTile2Def_NPUCoreDef_sram_addrb)
  );
  BF16Unit NPUTile2Def_BF16UnitDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .clock        (clk),
    .reset        (rst),
    .io_opc       (_NPUTile2Def_NPUCoreDef_bf16_opc),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_a         (_NPUTile2Def_NPUCoreDef_bf16_a),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_b         (_NPUTile2Def_NPUCoreDef_bf16_b),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_in_valid  (_NPUTile2Def_NPUCoreDef_bf16_iv),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_out_ready (_NPUTile2Def_NPUCoreDef_bf16_or),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_isSqrt    (_NPUTile2Def_NPUCoreDef_bf16_isSqrt),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_kill      (_NPUTile2Def_NPUCoreDef_bf16_kill),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_y         (_NPUTile2Def_BF16UnitDef_io_y),
    .io_in_ready  (_NPUTile2Def_BF16UnitDef_io_in_ready),
    .io_out_valid (_NPUTile2Def_BF16UnitDef_io_out_valid)
  );
  SRAM NPUTile2Def_SRAMADef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .clka  (clk),
    .ena   (_NPUTile2Def_NPUCoreDef_sram_ena),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .wea   (_NPUTile2Def_NPUCoreDef_sram_wea),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addra (_NPUTile2Def_NPUCoreDef_sram_addra),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dina  (_NPUTile2Def_NPUCoreDef_sram_dina),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clkb  (clk),
    .enb   (_NPUTile2Def_NPUCoreDef_sram_enb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addrb (_NPUTile2Def_NPUCoreDef_sram_addrb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .doutb (_NPUTile2Def_SRAMADef_doutb)
  );
  loadStoreController NPUTile2Def_LoadStoreControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .clk                 (clk),
    .rst                 (rst),
    .core_req            (_NPUTile2Def_NPUCoreDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_rwn            (_NPUTile2Def_NPUCoreDef_dma_rwn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_hostAddr       (_NPUTile2Def_NPUCoreDef_dma_hostAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_localAddr      (_NPUTile2Def_NPUCoreDef_dma_localAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_transferLength (_NPUTile2Def_NPUCoreDef_dma_transferLength),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_writeData      (_NPUTile2Def_NPUCoreDef_dma_writeData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dma_resp            (_DMAPathControllerDef_dma_resp_c),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_write_ready     (_DMAPathControllerDef_dma_write_ready_c),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_valid      (_DMAPathControllerDef_dma_read_valid_c),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_data       (_DMAPathControllerDef_dma_read_data_c),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .core_ready          (_NPUTile2Def_LoadStoreControllerDef_core_ready),
    .core_readData       (_NPUTile2Def_LoadStoreControllerDef_core_readData),
    .core_ack            (_NPUTile2Def_LoadStoreControllerDef_core_ack),
    .dma_req             (_NPUTile2Def_LoadStoreControllerDef_dma_req),
    .dma_write_valid     (_NPUTile2Def_LoadStoreControllerDef_dma_write_valid),
    .dma_write_data      (_NPUTile2Def_LoadStoreControllerDef_dma_write_data),
    .dma_read_ready      (_NPUTile2Def_LoadStoreControllerDef_dma_read_ready)
  );
  NPUCore NPUTile3Def_NPUCoreDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clk                      (clk),
    .rstn                     (rstn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:42:31]
    .rocc_if_host_mem_offset  (rocc_if_host_mem_offset_tile3),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :180:{51,65}]
    .rocc_if_size             (rocc_if_size_tile3),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :181:{40,54}]
    .rocc_if_local_mem_offset (rocc_if_local_mem_offset_tile3),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:123:22, :182:{52,66}]
    .rocc_if_funct            (rocc_if_funct),	// @[src/main/scala/chisel3/util/Decoupled.scala:376:21]
    .rocc_if_cmd_vld          (rocc_if_cmd_vld),	// @[src/main/scala/chisel3/util/Decoupled.scala:52:35]
    .bf16_y                   (_NPUTile3Def_BF16UnitDef_io_y),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ov                  (_NPUTile3Def_BF16UnitDef_io_out_valid),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .bf16_ir                  (_NPUTile3Def_BF16UnitDef_io_in_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .dma_ready                (_NPUTile3Def_LoadStoreControllerDef_core_ready),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_readData             (_NPUTile3Def_LoadStoreControllerDef_core_readData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .dma_ack                  (_NPUTile3Def_LoadStoreControllerDef_core_ack),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .sram_doutb               (_NPUTile3Def_SRAMADef_doutb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .rocc_if_fin              (_NPUTile3Def_NPUCoreDef_rocc_if_fin),
    .rocc_if_busy             (_NPUTile3Def_NPUCoreDef_rocc_if_busy),
    .bf16_opc                 (_NPUTile3Def_NPUCoreDef_bf16_opc),
    .bf16_a                   (_NPUTile3Def_NPUCoreDef_bf16_a),
    .bf16_b                   (_NPUTile3Def_NPUCoreDef_bf16_b),
    .bf16_iv                  (_NPUTile3Def_NPUCoreDef_bf16_iv),
    .bf16_or                  (_NPUTile3Def_NPUCoreDef_bf16_or),
    .bf16_isSqrt              (_NPUTile3Def_NPUCoreDef_bf16_isSqrt),
    .bf16_kill                (_NPUTile3Def_NPUCoreDef_bf16_kill),
    .dma_req                  (_NPUTile3Def_NPUCoreDef_dma_req),
    .dma_rwn                  (_NPUTile3Def_NPUCoreDef_dma_rwn),
    .dma_hostAddr             (_NPUTile3Def_NPUCoreDef_dma_hostAddr),
    .dma_localAddr            (_NPUTile3Def_NPUCoreDef_dma_localAddr),
    .dma_transferLength       (_NPUTile3Def_NPUCoreDef_dma_transferLength),
    .dma_writeData            (_NPUTile3Def_NPUCoreDef_dma_writeData),
    .sram_ena                 (_NPUTile3Def_NPUCoreDef_sram_ena),
    .sram_wea                 (_NPUTile3Def_NPUCoreDef_sram_wea),
    .sram_addra               (_NPUTile3Def_NPUCoreDef_sram_addra),
    .sram_dina                (_NPUTile3Def_NPUCoreDef_sram_dina),
    .sram_enb                 (_NPUTile3Def_NPUCoreDef_sram_enb),
    .sram_addrb               (_NPUTile3Def_NPUCoreDef_sram_addrb)
  );
  BF16Unit NPUTile3Def_BF16UnitDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:54:33]
    .clock        (clk),
    .reset        (rst),
    .io_opc       (_NPUTile3Def_NPUCoreDef_bf16_opc),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_a         (_NPUTile3Def_NPUCoreDef_bf16_a),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_b         (_NPUTile3Def_NPUCoreDef_bf16_b),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_in_valid  (_NPUTile3Def_NPUCoreDef_bf16_iv),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_out_ready (_NPUTile3Def_NPUCoreDef_bf16_or),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_isSqrt    (_NPUTile3Def_NPUCoreDef_bf16_isSqrt),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_kill      (_NPUTile3Def_NPUCoreDef_bf16_kill),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .io_y         (_NPUTile3Def_BF16UnitDef_io_y),
    .io_in_ready  (_NPUTile3Def_BF16UnitDef_io_in_ready),
    .io_out_valid (_NPUTile3Def_BF16UnitDef_io_out_valid)
  );
  SRAM NPUTile3Def_SRAMADef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:70:30]
    .clka  (clk),
    .ena   (_NPUTile3Def_NPUCoreDef_sram_ena),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .wea   (_NPUTile3Def_NPUCoreDef_sram_wea),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addra (_NPUTile3Def_NPUCoreDef_sram_addra),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dina  (_NPUTile3Def_NPUCoreDef_sram_dina),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .clkb  (clk),
    .enb   (_NPUTile3Def_NPUCoreDef_sram_enb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .addrb (_NPUTile3Def_NPUCoreDef_sram_addrb),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .doutb (_NPUTile3Def_SRAMADef_doutb)
  );
  loadStoreController NPUTile3Def_LoadStoreControllerDef (	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:96:44]
    .clk                 (clk),
    .rst                 (rst),
    .core_req            (_NPUTile3Def_NPUCoreDef_dma_req),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_rwn            (_NPUTile3Def_NPUCoreDef_dma_rwn),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_hostAddr       (_NPUTile3Def_NPUCoreDef_dma_hostAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_localAddr      (_NPUTile3Def_NPUCoreDef_dma_localAddr),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_transferLength (_NPUTile3Def_NPUCoreDef_dma_transferLength),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .core_writeData      (_NPUTile3Def_NPUCoreDef_dma_writeData),	// @[fpga/src/main/scala/nexysvideo/NPUtile.scala:39:32]
    .dma_resp            (_DMAPathControllerDef_dma_resp_d),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_write_ready     (_DMAPathControllerDef_dma_write_ready_d),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_valid      (_DMAPathControllerDef_dma_read_valid_d),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .dma_read_data       (_DMAPathControllerDef_dma_read_data_d),	// @[fpga/src/main/scala/nexysvideo/NPU.scala:145:38]
    .core_ready          (_NPUTile3Def_LoadStoreControllerDef_core_ready),
    .core_readData       (_NPUTile3Def_LoadStoreControllerDef_core_readData),
    .core_ack            (_NPUTile3Def_LoadStoreControllerDef_core_ack),
    .dma_req             (_NPUTile3Def_LoadStoreControllerDef_dma_req),
    .dma_write_valid     (_NPUTile3Def_LoadStoreControllerDef_dma_write_valid),
    .dma_write_data      (_NPUTile3Def_LoadStoreControllerDef_dma_write_data),
    .dma_read_ready      (_NPUTile3Def_LoadStoreControllerDef_dma_read_ready)
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
