// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.
package connx_npu
// package freechips.rocketchip.tile

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import chisel3.experimental.IntParam
import gemmini._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.util.ClockGate
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.InOrderArbiter


class DMAWrite extends Bundle() {
  val data = Bits(128.W)
}
class DMARead extends Bundle() {
  val data = Bits(128.W)
}

class DMAIO extends Bundle() {
    val req = Input(Bool())
    val resp = Output(Bool())
    val write = Decoupled(new DMAWrite)
    val read = Flipped(Decoupled(new DMARead))
}

class NPU(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes, nPTWPorts = 1) {

    val defaultConfig = GemminiCustomConfigs.defaultConfig

    val dummyConfig = GemminiArrayConfig[DummySInt, Float, Float](
        opcodes = opcodes,
        inputType = DummySInt(128),
        accType = DummySInt(128),
        spatialArrayOutputType = DummySInt(128),
        tileRows     = 1,
        tileColumns  = 1,
        meshRows     = 2,
        meshColumns  = 2,
        dataflow     = defaultConfig.dataflow,
        sp_capacity  = CapacityInKilobytes(128),
        acc_capacity = CapacityInKilobytes(64),
        sp_banks     = defaultConfig.sp_banks,
        acc_banks    = defaultConfig.acc_banks,
        sp_singleported = defaultConfig.sp_singleported,
        acc_singleported = defaultConfig.acc_singleported,
        has_training_convs = false,
        has_max_pool = false,
        has_nonlinear_activations = false,
        reservation_station_entries_ld = defaultConfig.reservation_station_entries_ld,
        reservation_station_entries_st = defaultConfig.reservation_station_entries_st,
        reservation_station_entries_ex = defaultConfig.reservation_station_entries_ex,
        ld_queue_length = defaultConfig.ld_queue_length,
        st_queue_length = defaultConfig.st_queue_length,
        ex_queue_length = defaultConfig.ex_queue_length,
        max_in_flight_mem_reqs = 16,
        dma_maxbytes = 64,
        dma_buswidth = 128,
        shifter_banks = 1,
        aligned_to = 1,
        tlb_size = defaultConfig.tlb_size,

        mvin_scale_args = Some(ScaleArguments(
        (t: DummySInt, f: Float) => t.dontCare,
        4, Float(28, 100), 4,
        identity = "1.0",
        c_str = "()"
        )),

        mvin_scale_acc_args = None,
        mvin_scale_shared = defaultConfig.mvin_scale_shared,

        acc_scale_args = Some(ScaleArguments(
        (t: DummySInt, f: Float ) => t.dontCare,
        1, Float(28, 100), -1,
        identity = "1.0",
        c_str = "()"
        )),

        num_counter = 0,

        acc_read_full_width = false,
        acc_read_small_width = defaultConfig.acc_read_small_width,

        ex_read_from_spad = defaultConfig.ex_read_from_spad,
        ex_read_from_acc = false,
        ex_write_to_spad = false,
        ex_write_to_acc = defaultConfig.ex_write_to_acc,
    )
    val xLen = p(XLen)
    val NPUTile0Def = LazyModule(new NPUTile)
    val NPUTile1Def = LazyModule(new NPUTile)
    val NPUTile2Def = LazyModule(new NPUTile)
    val NPUTile3Def = LazyModule(new NPUTile)
    val DMAEngineDef = LazyModule(new DMAEngine(dummyConfig))
    // lazy val DMAEngineDef = new DMAEngine(dummyConfig)
    override lazy val module = new NPUModuleImp(this)
    override val tlNode = DMAEngineDef.id_node
    // override val tlNode = module.id_node
    // override val atlNode = TLIdentityNode()
    val node = tlNode
    val n = 16
}
// class NPUModuleImp(outer: NPU)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
class NPUModuleImp(outer: NPU) extends LazyRoCCModuleImp(outer)
    with HasCoreParameters {

    import outer.dummyConfig._
    import outer.DMAEngineDef
    import outer.NPUTile0Def
    import outer.NPUTile1Def
    import outer.NPUTile2Def
    import outer.NPUTile3Def

    val regfile = Mem(outer.n, UInt(xLen.W))
    val busy = RegInit(VecInit(Seq.fill(outer.n){false.B}))

    val cmd = Queue(io.cmd)
    val funct = cmd.bits.inst.funct
    val addr = cmd.bits.rs2(log2Up(outer.n)-1,0)
    val doSetReg = funct === 0.U
    val doGetReg = funct === 1.U
    val doExec = funct === 2.U
    val doLoad = funct === 3.U
    val doStore = funct === 4.U


    // datapath
    when (cmd.fire() && doSetReg) {
        regfile(addr) := cmd.bits.rs1
    }

    // val NPUTile0Def = Module(new NPUTile)
    // val NPUTile1Def = Module(new NPUTile)
    // val NPUTile2Def = Module(new NPUTile)
    // val NPUTile3Def = Module(new NPUTile)
    val DMAGroupControllerDef = Module(new TSN_DGCL)
    // val DMAEngineDef = LazyModule(new DMAEngine()(p))

    // NPUTile0Def.io.clk := clock
    // NPUTile0Def.io.rst := reset
    NPUTile0Def.module.io.rocc_if_host_mem_offset := regfile(1)(40, 0)
    NPUTile0Def.module.io.rocc_if_size := regfile(2)(16, 0)
    NPUTile0Def.module.io.rocc_if_local_mem_offset := regfile(3)(16, 0)
    NPUTile0Def.module.io.rocc_if_funct := funct
    NPUTile0Def.module.io.rocc_if_cmd_vld := cmd.fire()
    val NPUTile0Fin = NPUTile0Def.module.io.rocc_if_fin
    val NPUTile0Busy = NPUTile0Def.module.io.rocc_if_busy

    // NPUTile1Def.io.clk := clock
    // NPUTile1Def.io.rst := reset
    NPUTile1Def.module.io.rocc_if_host_mem_offset := regfile(4)(40, 0)
    NPUTile1Def.module.io.rocc_if_size := regfile(5)(16, 0)
    NPUTile1Def.module.io.rocc_if_local_mem_offset := regfile(6)(16, 0)
    NPUTile1Def.module.io.rocc_if_funct := funct
    NPUTile1Def.module.io.rocc_if_cmd_vld := cmd.fire()
    val NPUTile1Fin = NPUTile1Def.module.io.rocc_if_fin
    val NPUTile1Busy = NPUTile1Def.module.io.rocc_if_busy

    // NPUTile2Def.io.clk := clock
    // NPUTile2Def.io.rst := reset
    NPUTile2Def.module.io.rocc_if_host_mem_offset := regfile(7)(40, 0)
    NPUTile2Def.module.io.rocc_if_size := regfile(8)(16, 0)
    NPUTile2Def.module.io.rocc_if_local_mem_offset := regfile(9)(16, 0)
    NPUTile2Def.module.io.rocc_if_funct := funct
    NPUTile2Def.module.io.rocc_if_cmd_vld := cmd.fire()
    val NPUTile2Fin = NPUTile2Def.module.io.rocc_if_fin
    val NPUTile2Busy = NPUTile2Def.module.io.rocc_if_busy

    // NPUTile3Def.io.clk := clock
    // NPUTile3Def.io.rst := reset
    NPUTile3Def.module.io.rocc_if_host_mem_offset := regfile(10)(40, 0)
    NPUTile3Def.module.io.rocc_if_size := regfile(11)(16, 0)
    NPUTile3Def.module.io.rocc_if_local_mem_offset := regfile(12)(16, 0)
    NPUTile3Def.module.io.rocc_if_funct := funct
    NPUTile3Def.module.io.rocc_if_cmd_vld := cmd.fire()
    val NPUTile3Fin = NPUTile3Def.module.io.rocc_if_fin
    val NPUTile3Busy = NPUTile3Def.module.io.rocc_if_busy

    DMAGroupControllerDef.io.gemmini_clk := clock
    DMAGroupControllerDef.io.fpu_clk := clock
    DMAGroupControllerDef.io.reset := reset
    DMAGroupControllerDef.io.dma_req_a := NPUTile0Def.module.io.dma_req 
    NPUTile0Def.module.io.dma_resp := DMAGroupControllerDef.io.dma_resp_a
    DMAGroupControllerDef.io.dma_write_valid_a := NPUTile0Def.module.io.dma_write_valid
    DMAGroupControllerDef.io.dma_write_data_a := NPUTile0Def.module.io.dma_write_data
    NPUTile0Def.module.io.dma_write_ready := DMAGroupControllerDef.io.dma_write_ready_a
    NPUTile0Def.module.io.dma_read_valid := DMAGroupControllerDef.io.dma_read_valid_a
    NPUTile0Def.module.io.dma_read_data := DMAGroupControllerDef.io.dma_read_data_a
    DMAGroupControllerDef.io.dma_read_ready_a := NPUTile0Def.module.io.dma_read_ready

    DMAGroupControllerDef.io.dma_req_b := NPUTile1Def.module.io.dma_req 
    NPUTile1Def.module.io.dma_resp := DMAGroupControllerDef.io.dma_resp_b
    DMAGroupControllerDef.io.dma_write_valid_b := NPUTile1Def.module.io.dma_write_valid
    DMAGroupControllerDef.io.dma_write_data_b := NPUTile1Def.module.io.dma_write_data
    NPUTile1Def.module.io.dma_write_ready := DMAGroupControllerDef.io.dma_write_ready_b
    NPUTile1Def.module.io.dma_read_valid := DMAGroupControllerDef.io.dma_read_valid_b
    NPUTile1Def.module.io.dma_read_data := DMAGroupControllerDef.io.dma_read_data_b
    DMAGroupControllerDef.io.dma_read_ready_b := NPUTile1Def.module.io.dma_read_ready

    DMAGroupControllerDef.io.dma_req_c := NPUTile2Def.module.io.dma_req 
    NPUTile2Def.module.io.dma_resp := DMAGroupControllerDef.io.dma_resp_c
    DMAGroupControllerDef.io.dma_write_valid_c := NPUTile2Def.module.io.dma_write_valid
    DMAGroupControllerDef.io.dma_write_data_c := NPUTile2Def.module.io.dma_write_data
    NPUTile2Def.module.io.dma_write_ready := DMAGroupControllerDef.io.dma_write_ready_c
    NPUTile2Def.module.io.dma_read_valid := DMAGroupControllerDef.io.dma_read_valid_c
    NPUTile2Def.module.io.dma_read_data := DMAGroupControllerDef.io.dma_read_data_c
    DMAGroupControllerDef.io.dma_read_ready_c := NPUTile2Def.module.io.dma_read_ready

    DMAGroupControllerDef.io.dma_req_d := NPUTile3Def.module.io.dma_req 
    NPUTile3Def.module.io.dma_resp := DMAGroupControllerDef.io.dma_resp_d
    DMAGroupControllerDef.io.dma_write_valid_d := NPUTile3Def.module.io.dma_write_valid
    DMAGroupControllerDef.io.dma_write_data_d := NPUTile3Def.module.io.dma_write_data
    NPUTile3Def.module.io.dma_write_ready := DMAGroupControllerDef.io.dma_write_ready_d
    NPUTile3Def.module.io.dma_read_valid := DMAGroupControllerDef.io.dma_read_valid_d
    NPUTile3Def.module.io.dma_read_data := DMAGroupControllerDef.io.dma_read_data_d
    DMAGroupControllerDef.io.dma_read_ready_d := NPUTile3Def.module.io.dma_read_ready

    DMAEngineDef.module.io.rcc.valid :=DMAGroupControllerDef.io.rcc_valid
    DMAGroupControllerDef.io.rcc_ready :=DMAEngineDef.module.io.rcc.ready
    DMAEngineDef.module.io.rcc.bits.dram_addr :=DMAGroupControllerDef.io.rcc_dram_addr
    DMAEngineDef.module.io.rcc.bits.dpram_addr :=DMAGroupControllerDef.io.rcc_dpram_addr
    DMAEngineDef.module.io.rcc.bits.length :=DMAGroupControllerDef.io.rcc_length
    
    DMAGroupControllerDef.io.rcd_valid :=DMAEngineDef.module.io.rcd.valid
    DMAEngineDef.module.io.rcd.ready :=DMAGroupControllerDef.io.rcd_ready
    DMAGroupControllerDef.io.rcd_dpram_addr :=DMAEngineDef.module.io.rcd.bits.dpram_addr
    DMAGroupControllerDef.io.rcd_read_data :=DMAEngineDef.module.io.rcd.bits.data
    DMAGroupControllerDef.io.rcd_length :=DMAEngineDef.module.io.rcd.bits.length

    DMAEngineDef.module.io.wcc.valid :=DMAGroupControllerDef.io.wcc_valid
    DMAGroupControllerDef.io.wcc_ready :=DMAEngineDef.module.io.wcc.ready
    DMAEngineDef.module.io.wcc.bits.dram_addr :=DMAGroupControllerDef.io.wcc_dram_addr
    DMAEngineDef.module.io.wcc.bits.data :=DMAGroupControllerDef.io.wcc_write_data
    DMAEngineDef.module.io.wcc.bits.length :=DMAGroupControllerDef.io.wcc_length

    // io.ptw <> DMAEngineDef.module.io.ptw

    // val mstatus = new MStatus
    DMAEngineDef.module.io.mstatus := cmd.bits.status

    // Counters
    // val counters = Module(new CounterController(outer.dummyConfig.num_counter, outer.xLen))
    // io.resp <> counters.io.out  // Counter access command will be committed immediately
    // counters.io.event_io.external_values(0) := 0.U
    // counters.io.event_io.event_signal(0) := false.B
    // counters.io.in.valid := false.B
    // counters.io.in.bits := DontCare
    // counters.io.event_io.collect(DMAEngineDef.module.io.counter)
    // TLB

    val use_tlb_register_filter = true
    val use_firesim_simulation_counters = false
    val use_shared_tlb = true

    implicit val edge = outer.DMAEngineDef.id_node.edges.out.head
    val tlb = Module(new FrontendTLB(2, tlb_size, dma_maxbytes, use_tlb_register_filter, use_firesim_simulation_counters, use_shared_tlb))
    (tlb.io.clients zip outer.DMAEngineDef.module.io.tlb).foreach(t => t._1 <> t._2)
    tlb.io.counter.external_reset := reset
    tlb.io.exp.foreach(_.flush_skip := false.B)
    tlb.io.exp.foreach(_.flush_retry := false.B)

    io.ptw <> tlb.io.ptw

    // counters.io.event_io.collect(tlb.io.counter)

    DMAEngineDef.module.io.flush := tlb.io.exp.map(_.flush()).reduce(_ || _)

    val clock_en_reg = RegInit(true.B)
    val gated_clock = if (clock_gate) ClockGate(clock, clock_en_reg, "gemmini_clock_gate") else clock
    outer.DMAEngineDef.module.clock := gated_clock

    // val id_node = DMAEngineDef.tlNode


    // val xbar_node = TLXbar()
    // id_node := TLWidthWidget(16) := TLBuffer() := xbar_node
    // xbar_node := DMAEngineDef.io.readerNode
    // xbar_node := DMAEngineDef.io.writerNode

    // when (io.mem.resp.valid) {
    //     regfile(memRespTag) := io.mem.resp.bits.data
    //     busy(memRespTag) := false.B
    // }

    // control
    // when (io.mem.req.fire()) {
    //     busy(addr) := true.B
    // }

  val doResp = cmd.bits.inst.xd
//   val stallReg = busy(addr)
//   val stallLoad = doLoad && !io.mem.req.ready
//   val stallResp = doResp && !io.resp.ready

//   cmd.ready := !stallReg && !stallLoad && !stallResp
  cmd.ready := !NPUTile0Busy && !NPUTile1Busy && !NPUTile2Busy && !NPUTile3Busy
  
    // command resolved if no stalls AND not issuing a load that will need a request

  // PROC RESPONSE INTERFACE
  io.resp.valid := cmd.valid && doResp && !NPUTile0Busy && !NPUTile1Busy && !NPUTile2Busy && !NPUTile3Busy
    // valid response if valid command, need a response, and no stalls
  io.resp.bits.rd := cmd.bits.inst.rd
    // Must respond with the appropriate tag or undefined behavior
//   io.resp.bits.data := accum
  io.resp.bits.data := regfile(addr)
    // Semantics is to always send out prior accumulator register value

  io.busy := cmd.valid || busy.reduce(_||_)
    // Be busy when have pending memory requests or committed possibility of pending requests
  io.interrupt := false.B
    // Set this true to trigger an interrupt on the processor (please refer to supervisor documentation)

  // MEMORY REQUEST INTERFACE
//   io.mem.req.valid := cmd.valid && doLoad && !stallReg && !stallResp
//   io.mem.req.bits.addr := addend
//   io.mem.req.bits.tag := addr
//   io.mem.req.bits.cmd := M_XRD // perform a load (M_XWR for stores)
//   io.mem.req.bits.size := log2Ceil(8).U
//   io.mem.req.bits.signed := false.B
//   io.mem.req.bits.data := 0.U // we're not performing any stores...
//   io.mem.req.bits.phys := false.B
//   io.mem.req.bits.dprv := cmd.bits.status.dprv
//   io.mem.req.bits.dv := cmd.bits.status.dv
}

class NPURoCCConfig extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val npu = LazyModule(new NPU(OpcodeSet.custom3))
      npu
    }
  )
})