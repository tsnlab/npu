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

        val dma_io = new DMAIO
        // val dma_req = Output(Bool())
        // val dma_resp = Input(Bool())
        // val dma_write_valid = Output(Bool())
        // val dma_write_data = Output(Bits(128.W))
        // val dma_write_ready = Input(Bool())
        // val dma_read_valid = Input(Bool())
        // val dma_read_data = Input(Bits(128.W))
        // val dma_read_ready = Output(Bool())
    })
    val NPUCoreDef = Definition(new BlackBoxNPUCore)
    val Bf16UnitDef = Definition(new BlackBoxBf16u)
    val Uint32UnitDef = Definition(new BlackBoxInt32u)
    val DPRAMDef = Definition(new BlackBoxDpram)
    val DMAControllerDef = Definition(new BlackBoxDmaController)

    NPUCoreDef.io.clk := io.clk
    NPUCoreDef.io.rst := io.rst
    NPUCoreDef.io.rocc_if_dram_offset := io.rocc_if_host_mem_offset
    NPUCoreDef.io.rocc_if_size := io.rocc_if_size
    NPUCoreDef.io.rocc_if_local_mem_offset := io.rocc_if_local_mem_offset
    NPUCoreDef.io.rocc_if_funct := io.rocc_if_funct
    NPUCoreDef.io.rocc_if_cmd_vld := io.rocc_if_cmd_vld
    io.rocc_if_fin := NPUCoreDef.io.rocc_if_fin
    io.rocc_if_busy := NPUCoreDef.io.rocc_if_busy

    DMAControllerDef.io.clk := io.clk
    DMAControllerDef.io.rst := io.rst
    io.dma_io.req := DMAControllerDef.io.dma_req
    DMAControllerDef.io.dma_resp := io.dma_io.resp
    io.dma_io.write_data.valid := DMAControllerDef.io.dma_write_valid
    io.dma_io.write_data.bits := DMAControllerDef.io.dma_write_data
    DMAControllerDef.io.dma_write_ready := io.dma_io.write_data.ready
    DMAControllerDef.io.dma_read_valid := io.dma_io.read_data.valid
    DMAControllerDef.io.dma_read_data := io.dma_io.read_data.bits
    io.dma_io.read_data.ready := DMAControllerDef.io.dma_read_ready
    
}
