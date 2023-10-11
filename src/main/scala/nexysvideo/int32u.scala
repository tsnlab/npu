package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxInt32u extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val int32_opc = Input(Bool())
    val int32_a = Input(Bool())
    val int32_b = Input(Bool())
    val int32_y = Output(Bool())
    val int32_iv = Input(Bool())
    val int32_or = Input(Bool())
    val int32_ov = Output(Bool())
    val int32_ir = Output(Bool())
  })
  addResource("nexysvideo/int32u.v")