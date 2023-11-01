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


        val NPUCoreDef = Module(new NPUCore)

        NPUCoreDef.io.clk := clock
        NPUCoreDef.io.rstn := ~reset.asBool
        NPUCoreDef.io.rocc_if_host_mem_offset := io.rocc_if_host_mem_offset
        NPUCoreDef.io.rocc_if_size := io.rocc_if_size
        NPUCoreDef.io.rocc_if_local_mem_offset := io.rocc_if_local_mem_offset
        NPUCoreDef.io.rocc_if_funct := io.rocc_if_funct
        NPUCoreDef.io.rocc_if_cmd_vld := io.rocc_if_cmd_vld
        io.rocc_if_fin := NPUCoreDef.io.rocc_if_fin
        io.rocc_if_busy := NPUCoreDef.io.rocc_if_busy

        // io.rocc_if_fin := true.B
        // io.rocc_if_busy := false.B

        val BF16UnitDef = Module(new BF16Unit)
        
        BF16UnitDef.io.opc := NPUCoreDef.io.bf16_opc
        BF16UnitDef.io.a := NPUCoreDef.io.bf16_a
        BF16UnitDef.io.b := NPUCoreDef.io.bf16_b
        NPUCoreDef.io.bf16_y := BF16UnitDef.io.y
        BF16UnitDef.io.in_valid := NPUCoreDef.io.bf16_iv
        BF16UnitDef.io.out_ready := NPUCoreDef.io.bf16_or
        NPUCoreDef.io.bf16_ov := BF16UnitDef.io.out_valid
        NPUCoreDef.io.bf16_ir := BF16UnitDef.io.in_ready
        BF16UnitDef.io.isSqrt := NPUCoreDef.io.bf16_isSqrt
        BF16UnitDef.io.kill := NPUCoreDef.io.bf16_kill



        // val Uint32UnitDef = Definition(new BlackBoxInt32u)
        val SRAMADef = Module(new SRAM)
        val SRAMBDef = Module(new SRAM)

        SRAMADef.io.rstn :=  ~reset.asBool
        SRAMADef.io.clka := clock
        SRAMADef.io.ena := NPUCoreDef.io.sram_a_ena
        SRAMADef.io.wea := NPUCoreDef.io.sram_a_wea
        SRAMADef.io.addra := NPUCoreDef.io.sram_a_addra
        SRAMADef.io.dina := NPUCoreDef.io.sram_a_dina
        SRAMADef.io.clkb := clock
        SRAMADef.io.enb := NPUCoreDef.io.sram_a_enb
        SRAMADef.io.addrb := NPUCoreDef.io.sram_a_addrb
        NPUCoreDef.io.sram_a_doutb := SRAMADef.io.doutb

        SRAMBDef.io.rstn :=  ~reset.asBool
        SRAMBDef.io.clka := clock
        SRAMBDef.io.ena := NPUCoreDef.io.sram_b_ena
        SRAMBDef.io.wea := NPUCoreDef.io.sram_b_wea
        SRAMBDef.io.addra := NPUCoreDef.io.sram_b_addra
        SRAMBDef.io.dina := NPUCoreDef.io.sram_b_dina
        SRAMBDef.io.clkb := clock
        SRAMBDef.io.enb := NPUCoreDef.io.sram_b_enb
        SRAMBDef.io.addrb := NPUCoreDef.io.sram_b_addrb
        NPUCoreDef.io.sram_b_doutb := SRAMBDef.io.doutb


        val DMAControllerDef = Module(new dmaController)


        DMAControllerDef.io.clk := clock
        DMAControllerDef.io.rst := reset

        DMAControllerDef.io.core_req := NPUCoreDef.io.dma_req
        NPUCoreDef.io.dma_ready := DMAControllerDef.io.core_ready
        DMAControllerDef.io.core_rwn := NPUCoreDef.io.dma_rwn
        DMAControllerDef.io.core_hostAddr := NPUCoreDef.io.dma_hostAddr
        DMAControllerDef.io.core_localAddr := NPUCoreDef.io.dma_localAddr
        DMAControllerDef.io.core_tansferLength := NPUCoreDef.io.dma_tansferLength
        DMAControllerDef.io.core_writeData := NPUCoreDef.io.dma_writeData
        NPUCoreDef.io.dma_readData := DMAControllerDef.io.core_readData
        NPUCoreDef.io.dma_ack := DMAControllerDef.io.core_ack

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
