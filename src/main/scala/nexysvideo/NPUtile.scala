package chipyard.fpga.nexys_video

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

@instantiable
class NPUTile extends Module {
    @public val io = IO(new Bundle {
        val clk = Input(Bool())
        val rst = Input(Bool())

        val rocc_if_host_mem_offset = Input(Bits(40.W))
        val rocc_if_size = Input(Bits(16.W))
        val rocc_if_local_mem_offset = Input(Bits(16.W))
        val rocc_if_funct = Input(Bits(7.W))
        val rocc_if_cmd_vld = Input(Bool())
        val rocc_if_fin = Output(Bool())
        val rocc_if_busy = Output(Bool())

        val dma_req = Output(Bool())
        val dma_resp = Input(Bool())
        val dma_write_valid = Output(Bool())
        val dma_write_data = Output(Bits(128.W))
        val dma_write_ready = Input(Bool())
        val dma_read_valid = Input(Bool())
        val dma_read_data = Input(Bits(128.W))
        val dma_read_ready = Output(Bool())
    })
    val NPUCoreDef = Definition(new BlackBoxInterpreter)
    val Bf16UnitDef = Definition(new BlackBoxBf16u)
    val Uint32UnitDef = Definition(new BlackBoxInt32u)
    val RPRAMDef = Definition(new BlackBoxDpram)
    val DMAControllerDef = Definition(new BlackBoxDmaCtrl)
}
