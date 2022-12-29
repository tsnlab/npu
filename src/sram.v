/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: sram 16kx64 dual port sram module
File name	: sram_16kx64.v
Module name	: sram_16kx64
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
module sram_16kx64 (
		//----| write signals
		input	wire		clka,		// write clock
		input	wire		ena,		// write enable
		input	wire		wea,		// write werite enable
		input	wire	[13:0]	addra,		// write address
		input	wire	[63:0]	dina,		// write data in

		//----| read signals
		input	wire		clkb,		// read clock
		input	wire		enb,		// read enable
		input	wire	[13:0]	addrb,		// read address
		output	reg	[63:0]	doutb		// read data out
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| behaviral modelling |---------------------------------------------------
reg	[63:0]	ram[16*1024-1:0];

//---- write operation
always @(posedge clka) if(ena && wea) ram[addra] <= dina;

//---- read operation
reg	[63:0]	outb;
always @(posedge clkb) if(enb) doutb <= ram[addrb];

endmodule
//==============================================================================


