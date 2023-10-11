package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxInt32u extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clka = Input(Bool())
    val ena = Input(Bool())
    val wea = Input(Bool())
    val addra = Input(Bool())
    val dina = Input(Bool())

    val clkb = Input(Bool())
    val enb = Input(Bool())
    val addrb = Input(Bool())
    val doutb = Output(Bool())
  })
  addResource("nexysvideo/dpram.v")