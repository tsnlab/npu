// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.

package freechips.rocketchip.tile

import chisel3._
import chisel3.util._
import chisel3.util.HasBlackBoxResource
import chisel3.experimental.IntParam
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
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
    val write_data = Decoupled(new DMAWrite)
    val read_data = Flipped(Decoupled(new DMARead))
}

class NPU(opcodes: OpcodeSet, val n: Int = 8)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new NPUModuleImp(this)
}

class NPUModuleImp(outer: NPU)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
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

    val NPUTile0Def = Definition(new NPUTile)
    val NPUTile1Def = Definition(new NPUTile)
    val NPUTile2Def = Definition(new NPUTile)
    val NPUTile3Def = Definition(new NPUTile)
    val DMAGroupControllerDef = Definition(new DMAGroupController)

    NPUTile0Def.clk := clock
    NPUTile0Def.rst := reset
    NPUTile0Def.rocc_if_host_mem_offset := regfile(1)(40, 0)
    NPUTile0Def.rocc_if_size := regfile(2)(16, 0)
    NPUTile0Def.rocc_if_local_mem_offset := regfile(3)(16, 0)
    NPUTile0Def.rocc_if_funct := funct
    NPUTile0Def.rocc_if_cmd_vld := cmd.fire()
    val NPUTile0Fin := NPUTile0Def.rocc_if_fin
    val NPUTile0Busy := NPUTile0Def.rocc_if_busy

    NPUTile1Def.clk := clock
    NPUTile1Def.rst := reset
    NPUTile1Def.rocc_if_host_mem_offset := regfile(4)(40, 0)
    NPUTile1Def.rocc_if_size := regfile(5)(16, 0)
    NPUTile1Def.rocc_if_local_mem_offset := regfile(6)(16, 0)
    NPUTile1Def.rocc_if_funct := funct
    NPUTile1Def.rocc_if_cmd_vld := cmd.fire()
    val NPUTile1Fin := NPUTile1Def.rocc_if_fin
    val NPUTile1Busy := NPUTile1Def.rocc_if_busy

    NPUTile2Def.clk := clock
    NPUTile2Def.rst := reset
    NPUTile2Def.rocc_if_host_mem_offset := regfile(7)(40, 0)
    NPUTile2Def.rocc_if_size := regfile(8)(16, 0)
    NPUTile2Def.rocc_if_local_mem_offset := regfile(9)(16, 0)
    NPUTile2Def.rocc_if_funct := funct
    NPUTile2Def.rocc_if_cmd_vld := cmd.fire()
    val NPUTile2Fin := NPUTile2Def.rocc_if_fin
    val NPUTile2Busy := NPUTile2Def.rocc_if_busy

    NPUTile3Def.clk := clock
    NPUTile3Def.rst := reset
    NPUTile3Def.rocc_if_host_mem_offset := regfile(10)(40, 0)
    NPUTile3Def.rocc_if_size := regfile(11)(16, 0)
    NPUTile3Def.rocc_if_local_mem_offset := regfile(12)(16, 0)
    NPUTile3Def.rocc_if_funct := funct
    NPUTile3Def.rocc_if_cmd_vld := cmd.fire()
    val NPUTile3Fin := NPUTile3Def.rocc_if_fin
    val NPUTile3Busy := NPUTile3Def.rocc_if_busy

    NPUTile0Def.dma_io <> DMAGroupControllerDef.io_a
    NPUTile1Def.dma_io <> DMAGroupControllerDef.io_b
    NPUTile2Def.dma_io <> DMAGroupControllerDef.io_c
    NPUTile3Def.dma_io <> DMAGroupControllerDef.io_d


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