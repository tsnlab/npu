package connx_npu

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

class NPUTile (implicit p: Parameters) extends LazyModule {

    lazy val module = new Impl
    class Impl extends LazyModuleImp(this) {
        val io = IO(new Bundle {
            // val clk = Input(Clock())
            // val rst = Input(Bool())

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

        val BF16UnitDef = Module(new BF16Unit)
        BF16UnitDef.io.opc := 0.U
        BF16UnitDef.io.a := 0.U
        BF16UnitDef.io.b := 0.U
        BF16UnitDef.io.in_valid := 0.U
        BF16UnitDef.io.out_ready := 0.U



        // import outer.DMAControllerDef
        //요기서부터 마저 해야함.

        // val Uint32UnitDef = Definition(new BlackBoxInt32u)
        // val DPRAMDef = Definition(new BlackBoxDpram)

        val DMAControllerDef = Module(new dmaController)


        DMAControllerDef.io.clk := clock
        DMAControllerDef.io.rst := reset
        io.dma_req := DMAControllerDef.io.dma_req
        DMAControllerDef.io.dma_resp := io.dma_resp
        io.dma_write_valid := DMAControllerDef.io.dma_write_valid
        io.dma_write_data := DMAControllerDef.io.dma_write_data
        DMAControllerDef.io.dma_write_ready := io.dma_write_ready
        DMAControllerDef.io.dma_read_valid := io.dma_read_valid
        DMAControllerDef.io.dma_read_data := io.dma_read_data
        io.dma_read_ready := DMAControllerDef.io.dma_read_ready

        // DMAControllerDef.io.dma_req := 0.U
        // DMAControllerDef.io.dma_write_valid := 0.U
        // DMAControllerDef.io.dma_write_data := 0.U
        // DMAControllerDef.io.dma_read_ready := 0.U

    }
    
}
