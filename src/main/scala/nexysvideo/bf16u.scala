package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxBf16u extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val bf16_opc = Input(Bool())
    val bf16_a = Input(Bool())
    val bf16_b = Input(Bool())
    val bf16_y = Output(Bool())
    val bf16_iv = Input(Bool())
    val bf16_or = Input(Bool())
    val bf16_ov = Output(Bool())
    val bf16_ir = Output(Bool())
  })
  addResource("nexysvideo/bf16u.v")