// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.

package freechips.rocketchip.tile

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import chisel3.experimental.IntParam
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.InOrderArbiter

class FPU_TSN(opcodes: OpcodeSet, val n: Int = 8)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new FPU_TSNModuleImp(this)
}

class FPU_TSNModuleImp(outer: FPU_TSN)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
    with HasCoreParameters {
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

//   val dAddrOffset = Bits(40.W)
//   val size = Bits(64.W)

//   when (cmd.fire() && (doLoad || doStore)) {
//     dAddrOffset := cmd.bits.rs2(39,0)
//     size := cmd.bits.rs1(15,0)
//   }

  val exec = cmd.fire() && doExec
  val load = cmd.fire() && doLoad
  val store = cmd.fire() && doStore

 

  when (io.mem.resp.valid) {
    regfile(memRespTag) := io.mem.resp.bits.data
    busy(memRespTag) := false.B
  }

  // control
  when (io.mem.req.fire()) {
    busy(addr) := true.B
  }

  val doResp = cmd.bits.inst.xd
  val stallReg = busy(addr)
  val stallLoad = doLoad && !io.mem.req.ready
  val stallResp = doResp && !io.resp.ready

  cmd.ready := !stallReg && !stallLoad && !stallResp
    // command resolved if no stalls AND not issuing a load that will need a request

  // PROC RESPONSE INTERFACE
  io.resp.valid := cmd.valid && doResp && !stallReg && !stallLoad
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
  io.mem.req.valid := cmd.valid && doLoad && !stallReg && !stallResp
  io.mem.req.bits.addr := addend
  io.mem.req.bits.tag := addr
  io.mem.req.bits.cmd := M_XRD // perform a load (M_XWR for stores)
  io.mem.req.bits.size := log2Ceil(8).U
  io.mem.req.bits.signed := false.B
  io.mem.req.bits.data := 0.U // we're not performing any stores...
  io.mem.req.bits.phys := false.B
  io.mem.req.bits.dprv := cmd.bits.status.dprv
  io.mem.req.bits.dv := cmd.bits.status.dv
}