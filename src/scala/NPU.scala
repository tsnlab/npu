// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.
package connx_npu
// package freechips.rocketchip.tile

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import chisel3.experimental.IntParam
// import gemmini._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.util.ClockGate
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.InOrderArbiter


// class DMAWrite extends Bundle() {
//   val data = Bits(128.W)
// }
// class DMARead extends Bundle() {
//   val data = Bits(128.W)
// }

// class DMAIO extends Bundle() {
//     val req = Input(Bool())
//     val resp = Output(Bool())
//     val write = Decoupled(new DMAWrite)
//     val read = Flipped(Decoupled(new DMARead))
// }

class NPU(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {

    val xLen = p(XLen)
    val NPUTile0Def = LazyModule(new NPUTile)
    val NPUTile1Def = LazyModule(new NPUTile)
    val NPUTile2Def = LazyModule(new NPUTile)
    val NPUTile3Def = LazyModule(new NPUTile)
    // val DMAEngineDef = LazyModule(new DMAEngine(dummyConfig))
    // lazy val DMAEngineDef = new DMAEngine(dummyConfig)
    override lazy val module = new NPUModuleImp(this)
    // override val tlNode = DMAEngineDef.id_node
    // override val tlNode = module.id_node
    // override val atlNode = TLIdentityNode()
    // val node = tlNode
    val n = 16
}
// class NPUModuleImp(outer: NPU)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
class NPUModuleImp(outer: NPU) extends LazyRoCCModuleImp(outer)
    with HasCoreParameters {

    // import outer.dummyConfig._
    // import outer.DMAEngineDef
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
    // val runCore0 = 0

    val cmdReg1 = RegInit(false.B)
    val cmdReg2 = RegInit(false.B)
    val cmdReg3 = RegInit(false.B)

    val funct7Reg1 = RegInit(0.U(7.W))
    val funct7Reg2 = RegInit(0.U(7.W))
    val funct7Reg3 = RegInit(0.U(7.W))

    val cmd_vld_core0 = cmd.bits.rs2(15, 14) === 0.U
    val cmd_vld_core1 = cmd.bits.rs2(15, 14) === 1.U
    val cmd_vld_core2 = cmd.bits.rs2(15, 14) === 2.U
    val cmd_vld_core3 = cmd.bits.rs2(15, 14) === 3.U
    // val load_data = Wire(UInt(64.W))
    // val local_mem_addr = Wire(UInt(13.W))
    // val local_mem_addr = Wire(13.W)
    // val load_data: UInt = Wire(UInt(64.W))
    // val local_mem_addr: UInt = Wire(UInt(13.W))
    val load_data: UInt = WireInit(0.U(64.W))
    val local_mem_addr: UInt = WireInit(0.U(13.W))

    cmdReg1 := cmd.fire()
    cmdReg2 := cmdReg1
    cmdReg3 := cmdReg2

    funct7Reg1 := funct
    funct7Reg2 := funct7Reg1
    funct7Reg3 := funct7Reg2
    // datapath
    when (cmd.fire() && doSetReg) {
        regfile(addr) := cmd.bits.rs1
    }
    when (cmd.fire() && doLoad) {
        load_data := cmd.bits.rs1
        local_mem_addr := cmd.bits.rs2(12, 0)
    }
    when (cmd.fire() && doStore) {
        local_mem_addr := cmd.bits.rs2(12, 0)
    }


    NPUTile0Def.module.io.local_mem_addr := local_mem_addr
    NPUTile0Def.module.io.load_data := load_data
    NPUTile0Def.module.io.core_cmd_funct7 := cmd.bits.inst.funct
    NPUTile0Def.module.io.core_cmd_vld := cmd_vld_core0 && cmd.fire()
    // io.resp.bits.data := NPUTile0Def.module.io.store_data
    val NPUTile0Fin = NPUTile0Def.module.io.rocc_if_fin
    val NPUTile0Busy = NPUTile0Def.module.io.rocc_if_busy
    regfile.write(1.U, NPUTile0Def.module.io.bf16_y_addr)
    // regfile(1) = NPUTile0Def.module.io.bf16_y_addr
    val NPUTile0_respValid = NPUTile0Def.module.io.core_resp_vld

    NPUTile1Def.module.io.local_mem_addr := local_mem_addr
    NPUTile1Def.module.io.load_data := load_data
    NPUTile1Def.module.io.core_cmd_funct7 := cmd.bits.inst.funct
    NPUTile1Def.module.io.core_cmd_vld := cmd_vld_core1 && cmd.fire()
    // io.resp.bits.data := NPUTile1Def.module.io.store_data
    val NPUTile1Fin = NPUTile1Def.module.io.rocc_if_fin
    val NPUTile1Busy = NPUTile1Def.module.io.rocc_if_busy
    regfile.write(2.U, NPUTile1Def.module.io.bf16_y_addr)
    // regfile(2) = NPUTile1Def.module.io.bf16_y_addr
    val NPUTile1_respValid = NPUTile1Def.module.io.core_resp_vld

    NPUTile2Def.module.io.local_mem_addr := local_mem_addr
    NPUTile2Def.module.io.load_data := load_data
    NPUTile2Def.module.io.core_cmd_funct7 := cmd.bits.inst.funct
    NPUTile2Def.module.io.core_cmd_vld := cmd_vld_core2 && cmd.fire()
    // io.resp.bits.data := NPUTile0Def.module.io.store_data
    val NPUTile2Fin = NPUTile2Def.module.io.rocc_if_fin
    val NPUTile2Busy = NPUTile2Def.module.io.rocc_if_busy
    regfile.write(3.U, NPUTile2Def.module.io.bf16_y_addr)
    // regfile(3) = NPUTile2Def.module.io.bf16_y_addr
    val NPUTile2_respValid = NPUTile2Def.module.io.core_resp_vld

    NPUTile3Def.module.io.local_mem_addr := local_mem_addr
    NPUTile3Def.module.io.load_data := load_data
    NPUTile3Def.module.io.core_cmd_funct7 := cmd.bits.inst.funct
    NPUTile3Def.module.io.core_cmd_vld := cmd_vld_core3 && cmd.fire()
    // io.resp.bits.data := NPUTile3Def.module.io.store_data
    val NPUTile3Fin = NPUTile3Def.module.io.rocc_if_fin
    val NPUTile3Busy = NPUTile3Def.module.io.rocc_if_busy
    regfile.write(4.U, NPUTile3Def.module.io.bf16_y_addr)
    // regfile(4) = NPUTile3Def.module.io.bf16_y_addr
    val NPUTile3_respValid = NPUTile3Def.module.io.core_resp_vld

    when(NPUTile0_respValid) {
        io.resp.bits.data := NPUTile0Def.module.io.store_data
    }.elsewhen(NPUTile1_respValid) {
        io.resp.bits.data := NPUTile1Def.module.io.store_data
    }.elsewhen(NPUTile2_respValid) {
        io.resp.bits.data := NPUTile2Def.module.io.store_data
    }.elsewhen(NPUTile3_respValid) {
        io.resp.bits.data := NPUTile3Def.module.io.store_data
    }.otherwise {
        io.resp.bits.data := regfile(addr)
    }

//   val doResp = cmd.bits.inst.xd
//   val stallReg = busy(addr)
//   val stallLoad = doLoad && !io.mem.req.ready
//   val doResp = NPUTile0_respValid || NPUTile1_respValid ||NPUTile2_respValid || NPUTile3_respValid

//   cmd.ready := !stallReg && !stallLoad && !stallResp
  cmd.ready := !NPUTile0Busy && !NPUTile1Busy && !NPUTile2Busy && !NPUTile3Busy
  
    // command resolved if no stalls AND not issuing a load that will need a request

  // PROC RESPONSE INTERFACE
//   io.resp.valid := cmd.valid && doResp && !NPUTile0Busy && !NPUTile1Busy && !NPUTile2Busy && !NPUTile3Busy
  io.resp.valid := NPUTile0_respValid || NPUTile1_respValid ||NPUTile2_respValid || NPUTile3_respValid
    // valid response if valid command, need a response, and no stalls
  val cmd_rd_reg = RegInit(0.U(64.W))
  when(cmd.valid){
    cmd_rd_reg := cmd.bits.inst.rd
  }
  io.resp.bits.rd := cmd_rd_reg
    // Must respond with the appropriate tag or undefined behavior
//   io.resp.bits.data := accum
//   io.resp.bits.data := regfile(addr)
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