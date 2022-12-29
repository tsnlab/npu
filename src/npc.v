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
module npc	(
		//----| system interface
		input	wire		rstn,
		input	wire		clk,

		//----| slave interface
		input	wire		slv_stt,	// slave start
		output	wire		slv_fin,	// slave finish
		input	wire	[31:0]	slv_ofs,	// slave offset
		input	wire	[31:0]	slv_siz,	// slave size
		output	wire		slv_bsy,	// slave busy

		//----| npc interface
		output	wire		npc_req,
		input	wire		npc_gnt,
		output	wire		npc_rwn,
		output	wire	[31:0]	npc_adr,
		output	wire	[31:0]	npc_len,
		output	wire	[63:0]	npc_wdt,
		input	wire	[63:0]	npc_rdt,
		input	wire		npc_ack
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| interpreter |-----------------------------------------------------------
wire	[16:0]	sram_wadr;
wire	[31:0]	sram_wdat;
wire	[16:0]	sram_radr;
wire	[31:0]	sram_rdat;

//----| interpreter |-----------------------------------------------------------
wire	[1:0]	fpu_opc;
wire	[31:0]	fpu_a;
wire	[31:0]	fpu_b;
wire	[31:0]	fpu_y;
wire		sram_ena;
wire		sram_wea;
wire	[13:0]	sram_addra;
wire	[63:0]	sram_dina;
wire		sram_enb;
wire	[13:0]	sram_addrb;
wire	[63:0]	sram_doutb;
intp		intp		(
		//----| system interface
		.rstn		( rstn		),
		.clk		( clk		),

		//----| slave interface
		.slv_stt	( slv_stt	),
		.slv_fin	( slv_fin	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy	),

		//----| npc interface
		.npc_req	( npc_req	),
		.npc_gnt	( npc_gnt	),
		.npc_rwn	( npc_rwn	),
		.npc_adr	( npc_adr	),
		.npc_len	( npc_len	),
		.npc_wdt	( npc_wdt	),
		.npc_rdt	( npc_rdt	),
		.npc_ack	( npc_ack	),
		
		//----| npc interface
		.fpu_opc	( fpu_opc	),
		.fpu_a		( fpu_a		),
		.fpu_b		( fpu_b		),
		.fpu_y		( fpu_y		),
		.fpu_iv		( fpu_iv	),
		.fpu_or		( fpu_or	),
		.fpu_ir		( fpu_ir	),
		.fpu_ov		( fpu_ov	),

		//----| sram  interface
		.sram_ena	( sram_ena	),
		.sram_wea	( sram_wea	),
		.sram_addra	( sram_addra	),
		.sram_dina	( sram_dina	),
		.sram_enb	( sram_enb	),
		.sram_addrb	( sram_addrb	),
		.sram_doutb	( sram_doutb	)
		);

//----| fpu |-------------------------------------------------------------------
fpu		fpu		(
		//----| systems
		.rstn		( rstn		),
		.clk		( clk		),

		//----| inputs
		.opc		( fpu_opc	),
		.a		( fpu_a		),
		.b		( fpu_b		),
		.i_v		( fpu_iv	),
		.o_r		( fpu_or	),

		//----| outputs
		.i_r		( fpu_ir	),
		.o_v		( fpu_ov	),
		.y		( fpu_y		)
		);

//----| sram |------------------------------------------------------------------
sram_16kx64	sram		(
		//----| write signals
		.clka		( clk		),
		.ena		( sram_ena	),
		.wea		( sram_wea	),
		.addra		( sram_addra	),
		.dina		( sram_dina	),

		//----| read signals
		.clkb		( clk		),
		.enb		( sram_enb	),
		.addrb		( sram_addrb	),
		.doutb		( sram_doutb	)
		);

endmodule
//==============================================================================
