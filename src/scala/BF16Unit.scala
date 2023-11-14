package connx_npu

import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.util.DontTouch
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}
import connx_npu.fudian._

class BF16Unit extends Module with DontTouch{
    val exp = 8
    val sig = 8
    // lazy val module = new Impl
    // class Impl extends LazyModuleImp(this) {
    val io = IO(new Bundle {
        val opc = Input(Bits(2.W))
        val a, b = Input(Bits(16.W))
        val in_valid, out_ready = Input(Bool())
        val y = Output(Bits(16.W))
        val in_ready, out_valid = Output(Bool())
        val isSqrt = Input(Bool())
        val kill = Input(Bool())
    })
    val doAdd = io.opc === 0.U
    val doSub = io.opc === 1.U
    val doMul = io.opc === 2.U
    val doDiv = io.opc === 3.U
    val b_b = Mux(doSub, Cat(~io.b(15), io.b(14, 0)), io.b)

    val bfAdd = Module(new FADD(exp, sig))
    val y_add = bfAdd.io.result
    bfAdd.io.a := io.a
    bfAdd.io.b := b_b
    bfAdd.io.rm := 1.U

    val bfMul = Module(new FMUL(exp, sig))
    val y_mul = bfMul.io.result
    bfMul.io.a := io.a
    bfMul.io.b := b_b
    bfMul.io.rm := 1.U

    // val bfDiv = Module(new FDIV(exp, sig))
    val bfDiv = Module(new FDIV(exp, sig+3))
    // val y_div = bfDiv.io.result
    val y_div = bfDiv.io.result(exp+sig+3 - 1,3)
    // bfDiv.io.a := io.a
    bfDiv.io.a := Cat(io.a,0.U(3.W)) 
    // bfDiv.io.b := b_b
    bfDiv.io.b :=Cat(b_b,0.U(3.W)) 
    bfDiv.io.rm := 1.U
    bfDiv.io.specialIO.isSqrt := io.isSqrt
    bfDiv.io.specialIO.kill := io.kill
    bfDiv.io.specialIO.in_valid := io.in_valid
    bfDiv.io.specialIO.out_ready :=io.out_ready
    io.in_ready := bfDiv.io.specialIO.in_ready
    io.out_valid := bfDiv.io.specialIO.out_valid
    // io.in_ready := 0.U
    // io.out_valid := 0.U
    
    when(doMul) {
        io.y := y_mul
    }.elsewhen(doDiv) {
        io.y := y_div
    }.elsewhen(doAdd || doSub) {
        io.y := y_add
    }.otherwise {
        io.y := 0.U
    }
    // }

}
