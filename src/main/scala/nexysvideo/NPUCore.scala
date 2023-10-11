package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxNPUCore extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Bool())
    val reset = Input(Bool())
    // val rocc_if_dram_offset = Input(UInt(40.W))
    // val rocc_if_size = Input(UInt(16.W))
    // val rocc_if_funct = Input(UInt(7.W))
    // val rocc_if_cmd_vld = Input(Bool())
    // val rocc_if_fin = Output(Bool())
    // val rocc_if_busy = Output(Bool())
    val rocc_if_host_mem_offset = Input(Bits(40.W))
    val rocc_if_size = Input(Bits(16.W))
    val rocc_if_local_mem_offset = Input(Bits(16.W))
    val rocc_if_funct = Input(Bits(7.W))
    val rocc_if_cmd_vld = Input(Bool())
    val rocc_if_fin = Output(Bool())
    val rocc_if_busy = Output(Bool())

    val bf16_opc = Output(Bool())
    val bf16_a = Output(Bool())
    val bf16_b = Output(Bool())
    val bf16_y = Input(Bool())
    val bf16_iv = Output(Bool())
    val bf16_or = Output(Bool())
    val bf16_ov = Input(Bool())
    val bf16_ir = Input(Bool())

    val uint32_opc = Output(Bool())
    val uint32_a = Output(Bool())
    val uint32_b = Output(Bool())
    val uint32_y = Input(Bool())
    val uint32_iv = Output(Bool())
    val uint32_or = Output(Bool())
    val uint32_ov = Input(Bool())
    val uint32_ir = Input(Bool())
  })
  addResource("nexysvideo/npuCore.v")
}

