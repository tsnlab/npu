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
            val local_mem_addr = Input(Bits(13.W))
            val load_data = Input(Bits(64.W))
            val store_data = Output(Bits(64.W))
            val core_cmd_funct7 = Input(Bits(7.W))
            val core_cmd_vld = Input(Bool())
            val rocc_if_fin = Output(Bool())
            val rocc_if_busy = Output(Bool())
            val bf16_y_addr = Output(Bits(16.W))
            val core_resp_vld = Output(Bool())
        })

        val NPUCoreDef = Module(new NPUCore)

        NPUCoreDef.io.clk := clock
        NPUCoreDef.io.rstn := ~reset.asBool
        NPUCoreDef.io.local_mem_addr := io.local_mem_addr
        NPUCoreDef.io.load_data := io.load_data
        NPUCoreDef.io.core_cmd_funct7 := io.core_cmd_funct7
        NPUCoreDef.io.core_cmd_vld := io.core_cmd_vld
        io.store_data := NPUCoreDef.io.store_data
        io.rocc_if_fin := NPUCoreDef.io.rocc_if_fin
        io.rocc_if_busy := NPUCoreDef.io.rocc_if_busy
        io.bf16_y_addr := NPUCoreDef.io.bf16_y_addr
        io.core_resp_vld := NPUCoreDef.io.core_resp_vld

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



        val SRAMADef = Module(new SRAM)

        SRAMADef.io.clka := clock
        SRAMADef.io.ena := NPUCoreDef.io.sram_ena
        SRAMADef.io.wea := NPUCoreDef.io.sram_wea
        SRAMADef.io.addra := NPUCoreDef.io.sram_addra
        SRAMADef.io.dina := NPUCoreDef.io.sram_dina
        SRAMADef.io.clkb := clock
        SRAMADef.io.enb := NPUCoreDef.io.sram_enb
        SRAMADef.io.addrb := NPUCoreDef.io.sram_addrb
        NPUCoreDef.io.sram_doutb := SRAMADef.io.doutb
    }
    
}
