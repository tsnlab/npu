/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: sram 32kx32 dual port sram module
File name	: sram_32kx32.v
Module name	: sram_32kx32
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is 42kx32 dual prot sram module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module sram_32kx32 (
		//----| write signals
		input	wire		clka,		// write clock
		input	wire		ena,		// write enable
		input	wire		wea,		// write werite enable
		input	wire	[14:0]	addra,		// write address
		input	wire	[31:0]	dina,		// write data in

		//----| read signals
		input	wire		clkb,		// read clock
		input	wire		enb,		// read enable
		input	wire	[14:0]	addrb,		// read address
		output	reg	[31:0]	doutb		// read data out
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| behaviral modelling |---------------------------------------------------
reg	[31:0]	ram[32*1024-1:0];

//---- write operation
always @(posedge clka) if(ena && wea) ram[addra] <= dina;

//---- read operation
reg	[31:0]	outb;
always @(posedge clkb) if(enb) doutb <= ram[addrb];

endmodule
//==============================================================================


