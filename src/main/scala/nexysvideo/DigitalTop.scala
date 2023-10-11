package chipyard.fpga.nexys_video

import chisel3._

import freechips.rocketchip.subsystem._
import freechips.rocketchip.system._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import chipyard.example._

import chipyard.{DigitalTop, DigitalTopModule}

// ------------------------------------
// NexysVideo DigitalTop
// ------------------------------------

class NexysVideoDigitalTop(implicit p: Parameters) extends DigitalTop
  with sifive.blocks.devices.i2c.HasPeripheryI2C
  with testchipip.HasPeripheryTSIHostWidget
{
  override lazy val module = new NexysVideoDigitalTopModule(this)
}

class NexysVideoDigitalTopModule[+L <: NexysVideoDigitalTop](l: L) extends DigitalTopModule(l)
  with sifive.blocks.devices.i2c.HasPeripheryI2CModuleImp
  with chipyard.example.CanHavePeripheryGCDModuleImp