PROGRAM=main

compile:
	riscv64-unknown-elf-gcc -fno-common -fno-builtin-printf -specs=htif_nano.specs -c $(PROGRAM).c
	riscv64-unknown-elf-gcc -static  -specs=htif_nano.specs $(PROGRAM).o -o $(PROGRAM).riscv

sim:
	spike $(PROGRAM).riscv

run:
	../chipyard/generators/testchipip/uart_tsi/uart_tsi +tty=/dev/ttyUSB2 +selfcheck $(PROGRAM).riscv