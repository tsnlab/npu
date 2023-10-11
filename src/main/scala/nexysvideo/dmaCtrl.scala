package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxDmaCtrl extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Bool())
    val rst = Input(Bool())

    val load = Input(Bool())
    val store = Input(Bool())

    val load_io = Output(Bool())
    val store_io = Output(Bool())
  })
  addResource("nexysvideo/dmaCtrl.v")