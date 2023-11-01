package connx_npu

import chisel3._
import chisel3.util.HasBlackBoxResource

class NPUCore extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rstn = Input(Bool())
    val rocc_if_host_mem_offset = Input(Bits(40.W))
    val rocc_if_size = Input(Bits(16.W))
    val rocc_if_local_mem_offset = Input(Bits(16.W))
    val rocc_if_funct = Input(Bits(7.W))
    val rocc_if_cmd_vld = Input(Bool())
    val rocc_if_fin = Output(Bool())
    val rocc_if_busy = Output(Bool())

    val bf16_opc = Output(Bits(2.W))
    val bf16_a = Output(Bits(16.W))
    val bf16_b = Output(Bits(16.W))
    val bf16_y = Input(Bits(16.W))
    val bf16_iv = Output(Bool())
    val bf16_or = Output(Bool())
    val bf16_ov = Input(Bool())
    val bf16_ir = Input(Bool())
    val bf16_isSqrt = Output(Bool())
    val bf16_kill = Output(Bool())

    val dma_req = Output(Bool())
    val dma_ready = Input(Bool())
    val dma_rwn = Output(Bool())
    val dma_hostAddr = Output(Bits(40.W))
    val dma_localAddr = Output(Bits(16.W))
    val dma_tansferLength = Output(Bits(16.W))
    val dma_writeData = Output(Bits(128.W))
    val dma_readData = Input(Bits(128.W))
    val dma_ack = Input(Bool())
    
    val sram_a_ena = Output(Bool())
    val sram_a_wea = Output(Bool())
    val sram_a_addra = Output(UInt(12.W))
    val sram_a_dina = Output(UInt(128.W))
    val sram_a_enb = Output(Bool())
    val sram_a_addrb = Output(UInt(12.W))
    val sram_a_doutb = Input(UInt(128.W))
    
    val sram_b_ena = Output(Bool())
    val sram_b_wea = Output(Bool())
    val sram_b_addra = Output(UInt(12.W))
    val sram_b_dina = Output(UInt(128.W))
    val sram_b_enb = Output(Bool())
    val sram_b_addrb = Output(UInt(12.W))
    val sram_b_doutb = Input(UInt(128.W))
  })
  addResource("vsrc/NPUCore.v")
}

