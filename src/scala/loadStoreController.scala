package connx_npu

import chisel3._
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._

class loadStoreController (implicit p: Parameters) extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val rst = Input(Bool())
    
    val core_req = Input(Bool())
    val core_ready = Output(Bool())
    val core_rwn = Input(Bool())
    val core_hostAddr = Input(Bits(40.W))
    val core_localAddr = Input(Bits(12.W))
    val core_transferLength = Input(Bits(16.W))
    val core_writeData = Input(Bits(128.W))
    val core_readData = Output(Bits(128.W))
    val core_readAddr = Output(Bits(12.W))
    val core_ack = Output(Bool())

    val dma_req = Output(Bool())
    val dma_resp = Input(Bool())
    val dma_write_valid = Output(Bool())
    val dma_write_data = Output(Bits(128.W))
    val dma_write_ready = Input(Bool())
    val dma_read_valid = Input(Bool())
    val dma_read_data = Input(Bits(140.W))
    val dma_read_ready = Output(Bool())
  })
  addResource("/vsrc/loadStoreController.v")
}