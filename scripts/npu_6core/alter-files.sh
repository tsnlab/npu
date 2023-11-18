#!/bin/bash

FPGADIR=chipyard/fpga
NPUDIR=$FPGADIR/src/main/scala/nexysvideo
mkdir $FPGADIR/src/main/resources/vsrc
VSRCDIR=$FPGADIR/src/main/resources/vsrc
TCLDIR=$FPGADIR/fpga-shells/xilinx/nexys_video/tcl

cp -f src/scala/BF16Unit.scala $NPUDIR
cp -f src/scala/Configs.scala $NPUDIR
cp -f src/scala/DMAEngine.scala $NPUDIR
# cp -f src/scala/DMAPathController.scala $NPUDIR
cp -f src/scala/DMAPathController_6p.scala $NPUDIR
cp -f src/scala/Harness.scala $NPUDIR
cp -f src/scala/HarnessBinders.scala $NPUDIR
# cp -f src/scala/NPU.scala $NPUDIR
cp -f src/scala/NPU_6core.scala $NPUDIR
cp -f src/scala/NPUCore.scala $NPUDIR
cp -f src/scala/NPUtile.scala $NPUDIR 
cp -f src/scala/SRAM.scala $NPUDIR
cp -f src/scala/loadStoreController.scala $NPUDIR

mkdir $NPUDIR/fudian

cp -f src/scala/fudian/ArgParser.scala $NPUDIR/fudian
cp -f src/scala/fudian/FADD.scala $NPUDIR/fudian
cp -f src/scala/fudian/FCMA.scala $NPUDIR/fudian
cp -f src/scala/fudian/FCMP.scala $NPUDIR/fudian
cp -f src/scala/fudian/FDIV.scala $NPUDIR/fudian
cp -f src/scala/fudian/FMUL.scala $NPUDIR/fudian
cp -f src/scala/fudian/FPToFP.scala $NPUDIR/fudian
cp -f src/scala/fudian/FPToInt.scala $NPUDIR/fudian
cp -f src/scala/fudian/Generator.scala $NPUDIR/fudian
cp -f src/scala/fudian/IntToFP.scala $NPUDIR/fudian
cp -f src/scala/fudian/RoundingUnit.scala $NPUDIR/fudian
cp -f src/scala/fudian/package.scala $NPUDIR/fudian
cp -f src/scala/fudian/utils/CLZ.scala $NPUDIR/fudian/utils
cp -f src/scala/fudian/utils/LZA.scala $NPUDIR/fudian/utils
cp -f src/scala/fudian/utils/ShiftRightJam.scala $NPUDIR/fudian/utils
# cp -f src/resources/vsrc/DMAPathController.v $VSRCDIR
cp -f src/resources/vsrc/DMAPathController_6p.v $VSRCDIR
cp -f src/resources/vsrc/NPUCore.v $VSRCDIR
cp -f src/resources/vsrc/SRAM.v $VSRCDIR
cp -f src/resources/vsrc/loadStoreController.v $VSRCDIR
cp -f src/tcl/ip.tcl $TCLDIR
cp -f build.sbt chipyard
