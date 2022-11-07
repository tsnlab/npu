/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu core module
File name	: npc.v
Module name	: npc
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the core of npu  module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module fpu	(
		//----| systems
		input	wire		rstn,
		input	wire		clk,

		//----| inputs
		input	wire	[1:0]	opc,	// 0: add, 1: sub, 2: mul, 3: div
		input	wire	[31:0]	a,
		input	wire	[31:0]	b,
		input	wire		i_v,
		input	wire		o_r,

		//----| outputs
		output	wire		i_r,
		output	wire		o_v,
		output	wire	[31:0]	y
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| interpreter |-----------------------------------------------------------
wire	[16:0]	sram_wadr;
wire	[31:0]	sram_wdat;
wire	[16:0]	sram_radr;
wire	[31:0]	sram_rdat;

//----| adder & subtractor |----------------------------------------------------
wire	[31:0]	b_		= opc == 1 ? {~b[31], b[30:0]} : b;
wire	[31:0]	y_add;
FADD		fadd(
		.clock		( clk		),
		.reset		( ~rstn		),
		.io_a		( a		),
		.io_b		( b_		),
		.io_rm		( 3'b000	),
		.io_result	( y_add		),
		.io_fflags	(		)
		);

//----| multiplier |------------------------------------------------------------
wire	[31:0]	y_mul;
FMUL		fmul(
		.clock		( clk		),
		.reset		( ~rstn		),
		.io_a		( a		),
		.io_b		( b		),
		.io_rm		( 3'b000	),
		.io_result	( y_mul		),
		.io_fflags	(		)
		);

//----| divider |---------------------------------------------------------------
wire	[31:0]	y_div;
FDIV		fdiv(
		.clock		( clk		),
		.reset		( ~rstn		),
		.io_a		( a		),
		.io_b		( b		),
		.io_rm		( 3'b000	),
		.io_result	( y_div		),
		.io_fflags	(		),
		.io_specialIO_in_valid	( i_v	),
		.io_specialIO_out_ready	( o_r	),
		.io_specialIO_isSqrt	( 1'b0	),
		.io_specialIO_kill	( 1'b0	),
		.io_specialIO_out_valid	( o_v	),
		.io_specialIO_in_ready	( i_r	)
		);

//----| output maping |---------------------------------------------------------
assign		y		= opc == 2 ? y_mul :
				  opc == 3 ? y_div :
					     y_add ;

endmodule
//==============================================================================
