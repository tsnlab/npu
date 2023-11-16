package connx_npu

import chisel3._
import chisel3.util.HasBlackBoxResource

class NPUCore extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rstn = Input(Bool())
    val local_mem_addr = Input(Bits(13.W))
    val load_data = Input(Bits(64.W))
    val store_data = Output(Bits(64.W))
    val core_cmd_funct7 = Input(Bits(7.W))
    val core_cmd_vld = Input(Bool())
    val rocc_if_fin = Output(Bool())
    val rocc_if_busy = Output(Bool())
    val bf16_y_addr = Output(Bits(16.W))
    val core_resp_vld = Output(Bool())

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
    
    val sram_ena = Output(Bool())
    val sram_wea = Output(Bool())
    val sram_addra = Output(UInt(12.W))
    val sram_dina = Output(UInt(128.W))
    val sram_enb = Output(Bool())
    val sram_addrb = Output(UInt(12.W))
    val sram_doutb = Input(UInt(128.W))
  })
  addResource("vsrc/NPUCore.v")
}

