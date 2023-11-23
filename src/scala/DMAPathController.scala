package connx_npu

import chisel3._
import chisel3.util.HasBlackBoxResource

class DMAPathController extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val risc_clk = Input(Clock())
	val fpu_clk = Input(Clock())
	val reset = Input(Bool())
	val rcc_dram_addr = Output(UInt(40.W))
	val rcc_dpram_addr = Output(UInt(16.W))
	val rcc_length = Output(UInt(16.W))
	val rcc_ready = Input(Bool())
	val rcc_valid = Output(Bool())
	val rcd_dpram_addr = Input(UInt(16.W))
	val rcd_read_data = Input(UInt(128.W))
	val rcd_length = Input(UInt(16.W))
	val rcd_ready = Output(Bool())
	val rcd_valid = Input(Bool())
	val wcc_dram_addr = Output(UInt(40.W))
	val wcc_dpram_addr = Output(UInt(16.W))
	val wcc_length = Output(UInt(16.W))
	val wcc_write_data = Output(UInt(128.W))
	val wcc_ready = Input(Bool())
	val wcc_valid = Output(Bool())
	val dma_req_a = Input(Bool())
	val dma_resp_a = Output(Bool())
	val dma_write_valid_a = Input(Bool())
	val dma_write_data_a = Input(UInt(128.W))
	val dma_write_ready_a = Output(Bool())
	val dma_read_valid_a = Output(Bool())
	val dma_read_data_a = Output(UInt(140.W))
	val dma_read_ready_a = Input(Bool())
	val dma_req_b = Input(Bool())
	val dma_resp_b = Output(Bool())
	val dma_write_valid_b = Input(Bool())
	val dma_write_data_b = Input(UInt(128.W))
	val dma_write_ready_b = Output(Bool())
	val dma_read_valid_b = Output(Bool())
	val dma_read_data_b = Output(UInt(140.W))
	val dma_read_ready_b = Input(Bool())
	val dma_req_c = Input(Bool())
	val dma_resp_c = Output(Bool())
	val dma_write_valid_c = Input(Bool())
	val dma_write_data_c = Input(UInt(128.W))
	val dma_write_ready_c = Output(Bool())
	val dma_read_valid_c = Output(Bool())
	val dma_read_data_c = Output(UInt(140.W))
	val dma_read_ready_c = Input(Bool())
	val dma_req_d = Input(Bool())
	val dma_resp_d = Output(Bool())
	val dma_write_valid_d = Input(Bool())
	val dma_write_data_d = Input(UInt(128.W))
	val dma_write_ready_d = Output(Bool())
	val dma_read_valid_d = Output(Bool())
	val dma_read_data_d = Output(UInt(140.W))
	val dma_read_ready_d = Input(Bool())
  })
  addResource("/vsrc/DMAPathController.v")
//   addResource("/vsrc/as128x1024/as128x1024.xci")
//   addResource("/vsrc/as32x512_ft/as32x512_ft.xci")
//   addResource("/vsrc/as72x512_ft/as72x512_ft.xci")
}