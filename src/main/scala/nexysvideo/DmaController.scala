package chipyard.fpga.nexys_video

import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxDmaController extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Bool())
    val rst = Input(Bool())

    val dma_req = Output(Bool())
    val dma_resp = Input(Bool())
    val dma_write_valid = Output(Bool())
    val dma_write_data = Output(Bits(128.W))
    val dma_write_ready = Input(Bool())
    val dma_read_valid = Input(Bool())
    val dma_read_data = Input(Bits(128.W))
    val dma_read_ready = Output(Bool())
  })
  addResource("nexysvideo/dmaController.v")