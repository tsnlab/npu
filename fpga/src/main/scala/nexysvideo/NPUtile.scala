package connx_npu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

@instantiable
class NPUTile extends Module {
    @public val io = IO(new Bundle {
        val clk = Input(Clock())
        val rst = Input(Bool())

        val rocc_if_host_mem_offset = Input(Bits(40.W))
        val rocc_if_size = Input(Bits(16.W))
        val rocc_if_local_mem_offset = Input(Bits(16.W))
        val rocc_if_funct = Input(Bits(7.W))
        val rocc_if_cmd_vld = Input(Bool())
        val rocc_if_fin = Output(Bool())
        val rocc_if_busy = Output(Bool())

        // val dma_io = new DMAIO
        val dma_req = Output(Bool())
        val dma_resp = Input(Bool())
        val dma_write_valid = Output(Bool())
        val dma_write_data = Output(Bits(128.W))
        val dma_write_ready = Input(Bool())
        val dma_read_valid = Input(Bool())
        val dma_read_data = Input(Bits(128.W))
        val dma_read_ready = Output(Bool())
    })
    // val NPUCoreDef = Module(new BlackBoxNPUCore)
    // val Bf16UnitDef = Definition(new BlackBoxBf16u)
    // val Uint32UnitDef = Definition(new BlackBoxInt32u)
    // val DPRAMDef = Definition(new BlackBoxDpram)
    val DMAControllerDef = Module(new dmaController)

    // NPUCoreDef.io.clock := io.clk
    // NPUCoreDef.io.reset := io.rst
    // NPUCoreDef.io.rocc_if_host_mem_offset := io.rocc_if_host_mem_offset
    // NPUCoreDef.io.rocc_if_size := io.rocc_if_size
    // NPUCoreDef.io.rocc_if_local_mem_offset := io.rocc_if_local_mem_offset
    // NPUCoreDef.io.rocc_if_funct := io.rocc_if_funct
    // NPUCoreDef.io.rocc_if_cmd_vld := io.rocc_if_cmd_vld
    // io.rocc_if_fin := NPUCoreDef.io.rocc_if_fin
    // io.rocc_if_busy := NPUCoreDef.io.rocc_if_busy
    io.rocc_if_fin := true.B
    io.rocc_if_busy := false.B

    DMAControllerDef.io.clk := io.clk
    DMAControllerDef.io.rst := io.rst
    io.dma_req := DMAControllerDef.io.dma_req
    DMAControllerDef.io.dma_resp := io.dma_resp
    io.dma_write_valid := DMAControllerDef.io.dma_write_valid
    io.dma_write_data := DMAControllerDef.io.dma_write_data
    DMAControllerDef.io.dma_write_ready := io.dma_write_ready
    DMAControllerDef.io.dma_read_valid := io.dma_read_valid
    DMAControllerDef.io.dma_read_data := io.dma_read_data
    io.dma_read_ready := DMAControllerDef.io.dma_read_ready
    
}
