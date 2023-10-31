package connx_npu

import chisel3._
import chisel3.util.HasBlackBoxResource

class SRAM extends BlackBox with HasBlackBoxResource {
    val io = IO(new Bundle {
        val clka = Input(Clock())
        val ena = Input(Bool())
        val wea = Input(Bool())
        val addra = Input(UInt(12.W))
        val dina = Input(UInt(128.W))

        val clkb = Input(Clock())
        val enb = Input(Bool())
        val addrb = Input(UInt(12.W))
        val doutb = Output(UInt(128.W))

    })
    addResource("vsrc/SRAM.v")
}