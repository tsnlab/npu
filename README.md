# CONNX NPU - with RISC-V processor built on chipyard

1. build chipyard 
2. copy files in fpga folder to the directory that has same name of folder in chipyard
3. copy build.sbt file into chipyard

## build project
1. cd chipyard/
2. source env.sh/
3. cd fpga
4. make SUB_PROJECT=nexysvideo bitstream
