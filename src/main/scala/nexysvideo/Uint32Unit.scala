package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxUint32Unit extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val uint32_opc = Input(Bool())
    val uint32_a = Input(Bool())
    val uint32_b = Input(Bool())
    val uint32_y = Output(Bool())
    val uint32_iv = Input(Bool())
    val uint32_or = Input(Bool())
    val uint32_ov = Output(Bool())
    val uint32_ir = Output(Bool())
  })
  addResource("nexysvideo/uint32Unit.v")