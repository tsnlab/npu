package connx_npu

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}
import chisel3.util._
import gemmini._
import freechips.rocketchip.diplomacy._
// import freechips.rocketchip.diplomacy.{LazyModule, LazyModuleImp}
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
// import freechips.rocketchip.rocket.MStatus
import freechips.rocketchip.rocket._
import org.chipsalliance.cde.config._

class RCCIO (implicit p: Parameters) extends CoreBundle() {
  val dram_addr = Bits(40.W)
  val dpram_addr = Bits(16.W)
  val length = Bits(16.W)
}
class RCDIO extends Bundle() {
  val dpram_addr = Bits(16.W)
  val data = Bits(128.W)
  val length = Bits(16.W)
}
class WCCIO extends Bundle() {
  val data = Bits(128.W)
  val dram_addr = Bits(40.W)
  val length = Bits(16.W)
}

class DMAEngine[T <: Data, U <: Data, V <: Data] (config: GemminiArrayConfig[T, U, V])(implicit p: Parameters) extends LazyModule {
// class DMAEngineImp(outer: DMAEngine) (implicit p: Parameters) extends LazyModuleImp(outer) {

    import config._
    
    val id_node = TLIdentityNode()
    val xbar_node = TLXbar()

    // val max_in_flight_mem_reqs = 16
    val dataBits = 128 // dma_buswidth
    val maxBytes = 64 // dma_maxbytes

    val spad_w = 128
    val acc_w = 128
    // val aligned_to = 1
    val sp_bank_entries = 65536
    val acc_bank_entries = 65536
    val block_rows = 1
    val block_cols = 1
    // val mstatus = Output(new MStatus)

    val reader = LazyModule(new StreamReader(config, max_in_flight_mem_reqs, dataBits, maxBytes, spad_w, acc_w, aligned_to,
        sp_bank_entries, acc_bank_entries, block_rows, use_tlb_register_filter,
        use_firesim_simulation_counters))
    val writer = LazyModule(new StreamWriter(max_in_flight_mem_reqs, dataBits, maxBytes,
        spad_w, aligned_to, Float(28, 100), block_cols, use_tlb_register_filter,
        use_firesim_simulation_counters))

    xbar_node := TLBuffer() := reader.node
    xbar_node := TLBuffer() := writer.node
    id_node := TLWidthWidget(16) := TLBuffer() := xbar_node

    lazy val module = new Impl
    class Impl extends LazyModuleImp(this) with HasCoreParameters {
        // val io = new Bundle {
        val io = IO(new Bundle {
            val rcc = Flipped(Decoupled(new RCCIO))
            val rcd = Decoupled(new RCDIO)
            val wcc = Flipped(Decoupled(new WCCIO))
            // val rcc = new Bundle {
            //     val valid = Input(Bool())
            //     val ready = Output(Bool())
            //     val dram_addr = Input(UInt(40.W))
            //     val dpram_addr = Input(UInt(16.W))
            //     val length = Input(UInt(16.W))
            // }

            // val rcd = new Bundle {
            //     val valid = Output(Bool())
            //     val ready = Input(Bool())
            //     val dpram_addr = Output(UInt(16.W))
            //     val data = Output(UInt(128.W))
            //     val length = Output(UInt(16.W))
            // }

            // val wcc = new Bundle {
            //     val valid = Input(Bool())
            //     val ready = Output(Bool())
            //     val data = Input(UInt(128.W))
            //     val length = Input(UInt(16.W))
            //     val dram_addr = Input(UInt(40.W))
            // }
            val mstatus = Input(new MStatus)
            // val ptw = new TLBPTWIO
            // val readerNode = new TLBuffer()
            // val writerNode = new TLBuffer()
            val tlb = Vec(2, new FrontendTLBIO)
            val busy = Output(Bool())
            val flush = Input(Bool())
            // val counter = new CounterEventIO()
        })
        
        reader.module.io.req.valid := io.rcc.valid
        io.rcc.ready := reader.module.io.req.ready
        reader.module.io.req.bits.vaddr := io.rcc.bits.dram_addr
        reader.module.io.req.bits.spaddr := io.rcc.bits.dpram_addr
        reader.module.io.req.bits.len := io.rcc.bits.length
        reader.module.io.req.bits.repeats := 0.U
        reader.module.io.req.bits.pixel_repeats := 0.U
        reader.module.io.req.bits.scale := 1.U
        reader.module.io.req.bits.is_acc := false.B
        reader.module.io.req.bits.accumulate := false.B
        reader.module.io.req.bits.has_acc_bitwidth := false.B
        reader.module.io.req.bits.block_stride := 1.U
        reader.module.io.req.bits.status := io.mstatus
        reader.module.io.req.bits.cmd_id := 0.U

        io.rcd.valid := reader.module.io.resp.valid
        reader.module.io.resp.ready := io.rcd.ready
        io.rcd.bits.data := reader.module.io.resp.bits.data
        io.rcd.bits.dpram_addr := reader.module.io.resp.bits.addr
        io.rcd.bits.length := reader.module.io.resp.bits.len

        writer.module.io.req.valid := io.wcc.valid
        io.wcc.ready := writer.module.io.req.ready
        writer.module.io.req.bits.vaddr := io.wcc.bits.dram_addr
        writer.module.io.req.bits.len := io.wcc.bits.length
        writer.module.io.req.bits.data := io.wcc.bits.data
        writer.module.io.req.bits.block := 16.U
        writer.module.io.req.bits.status := io.mstatus
        writer.module.io.req.bits.pool_en := false.B 
        writer.module.io.req.bits.store_en := false.B

        // implicit val edge = id_node.edges.out.head
        // val tlb = Module(new FrontendTLB(2, dummyConfig.tlb_size, dummyConfig.dma_maxbytes, use_tlb_register_filter, use_firesim_simulation_counters, use_shared_tlb))
        // tlb.io.clients(0) <> reader.module.io.tlb
        // tlb.io.clients(1) <> writer.module.io.tlb

        // module.io.ptw <> tlb.io.ptw
        reader.module.io.flush := false.B
        reader.module.io.counter.external_reset := false.B
        writer.module.io.flush := false.B
        writer.module.io.counter.external_reset := false.B

        io.tlb(0) <> writer.module.io.tlb
        io.tlb(1) <> reader.module.io.tlb

        writer.module.io.flush := io.flush
        reader.module.io.flush := io.flush
        io.busy := writer.module.io.busy || reader.module.io.busy
        
        // Counter connection
        // io.counter.collect(reader.module.io.counter)
        // io.counter.collect(writer.module.io.counter)
    }
    // val use_tlb_register_filter = true
    // val use_firesim_simulation_counters = false
    // val use_shared_tlb = true

    
}