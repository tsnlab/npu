package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxInterpreter extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Bool())
    val reset = Input(Bool())
    // val rocc_if_dram_offset = Input(UInt(40.W))
    // val rocc_if_size = Input(UInt(16.W))
    // val rocc_if_funct = Input(UInt(7.W))
    // val rocc_if_cmd_vld = Input(Bool())
    // val rocc_if_fin = Output(Bool())
    // val rocc_if_busy = Output(Bool())
    val rocc_if_dram_offset = Input(Bool())
    val rocc_if_size = Input(Bool())
    val rocc_if_funct = Input(Bool())
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

    val int32_opc = Output(Bool())
    val int32_a = Output(Bool())
    val int32_b = Output(Bool())
    val int32_y = Input(Bool())
    val int32_iv = Output(Bool())
    val int32_or = Output(Bool())
    val int32_ov = Input(Bool())
    val int32_ir = Input(Bool())
  })
  addResource("nexysvideo/interpreter.v")
}

