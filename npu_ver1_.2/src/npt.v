/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu top module
File name	: npt.v
Module name	: npt
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the top of npu  module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module npt	(
		//----| axi master interface
		// axi system signals
		input	wire		m_axi_arstn,
		input	wire		m_axi_aclk,
		// write addresss channel signals
		output	wire	[5:0]	m_axi_awid,
		output	wire	[31:0]	m_axi_awaddr,
		output	wire	[7:0]	m_axi_awlen,
		output	wire	[2:0]	m_axi_awsize,
		output	wire	[1:0]	m_axi_awburst,
		output	wire		m_axi_awlock,
		output	wire	[3:0]	m_axi_awcache,
		output	wire	[2:0]	m_axi_awprot,
		output	wire	[3:0]	m_axi_awqos,
		output	wire	[3:0]	m_axi_awregion,
		// input	wire	[0:0]	m_axi_awuser,
		output	wire		m_axi_awvalid,
		input	wire		m_axi_awready,
		// write data channel signals
		output	wire	[31:0]	m_axi_wdata,
		output	wire	[3:0]	m_axi_wstrb,
		output	wire		m_axi_wlast,
		// input	wire	[0:0]	m_axi_wuser,
		output	wire		m_axi_wvalid,
		input	wire		m_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m_axi_bid,
		input	wire	[1:0]	m_axi_bresp,
		// output	wire	[0:0]	m_axi_buser,
		input	wire		m_axi_bvalid,
		output	wire		m_axi_bready,
		// write read address channel signals
		output	wire	[5:0]	m_axi_arid,
		output	wire	[31:0]	m_axi_araddr,
		output	wire	[7:0]	m_axi_arlen,
		output	wire	[2:0]	m_axi_arsize,
		output	wire	[1:0]	m_axi_arburst,
		output	wire		m_axi_arlock,
		output	wire	[3:0]	m_axi_arcache,
		output	wire	[2:0]	m_axi_arprot,
		output	wire	[3:0]	m_axi_arqos,
		output	wire	[3:0]	m_axi_arregion,
		// input	wire	[0:0]	m_axi_aruser,
		output	wire		m_axi_arvalid,
		input	wire		m_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m_axi_rid,
		input	wire	[31:0]	m_axi_rdata,
		input	wire	[1:0]	m_axi_rresp,
		input	wire		m_axi_rlast,
		// output	wire	[0:0]	m_axi_ruser,
		input	wire		m_axi_rvalid,
		output	wire		m_axi_rready,

		//----| axi slave interface
		// axi system signals
		input	wire		s_axi_arstn,
		input	wire		s_axi_aclk,
		// write addresss channel signals
		input	wire	[11:0]	s_axi_awid,
		input	wire	[31:0]	s_axi_awaddr,
		input	wire	[7:0]	s_axi_awlen,
		input	wire	[2:0]	s_axi_awsize,
		input	wire	[1:0]	s_axi_awburst,
		input	wire		s_axi_awlock,
		input	wire	[3:0]	s_axi_awcache,
		input	wire	[2:0]	s_axi_awprot,
		input	wire	[3:0]	s_axi_awqos,
		input	wire	[3:0]	s_axi_awregion,
		input	wire		s_axi_awvalid,
		output	wire		s_axi_awready,
		// write data channel signals
		input	wire	[31:0]	s_axi_wdata,
		input	wire	[3:0]	s_axi_wstrb,
		input	wire		s_axi_wlast,
		input	wire		s_axi_wvalid,
		output	wire		s_axi_wready,
		// write response channel signals
		output	wire	[11:0]	s_axi_bid,
		output	wire	[1:0]	s_axi_bresp,
		output	wire		s_axi_bvalid,
		input	wire		s_axi_bready,
		// write read address channel signals
		input	wire	[11:0]	s_axi_arid,
		input	wire	[31:0]	s_axi_araddr,
		input	wire	[7:0]	s_axi_arlen,
		input	wire	[2:0]	s_axi_arsize,
		input	wire	[1:0]	s_axi_arburst,
		input	wire		s_axi_arlock,
		input	wire	[3:0]	s_axi_arcache,
		input	wire	[2:0]	s_axi_arprot,
		input	wire	[3:0]	s_axi_arqos,
		input	wire	[3:0]	s_axi_arregion,
		input	wire		s_axi_arvalid,
		output	wire		s_axi_arready,
		// write read data channel signals
		output	wire	[11:0]	s_axi_rid,
		output	wire	[31:0]	s_axi_rdata,
		output	wire	[1:0]	s_axi_rresp,
		output	wire		s_axi_rlast,
		output	wire		s_axi_rvalid,
		input	wire		s_axi_rready,

		//----| interrupt signals
		output	wire	[3:0]	irq,

		//----| led interface
		output  wire	[3:0]	led		// led out
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| np slave module |-------------------------------------------------------
wire	[31:0]	slv_ofs;
wire	[31:0]	slv_siz;
wire	[3:0]	slv_stt;
wire	[3:0]	slv_fin;
wire	[3:0]	slv_bsy;
nps		nps			(
		//----| axi slave interface
		// axi system signals
		.s_axi_arstn		( s_axi_arstn		),
		.s_axi_aclk		( s_axi_aclk		),

		// write addresss channel signals
		.s_axi_awid		( s_axi_awid		),
		.s_axi_awaddr		( s_axi_awaddr		),
		.s_axi_awlen		( s_axi_awlen		),
		.s_axi_awsize		( s_axi_awsize		),
		.s_axi_awburst		( s_axi_awburst		),
		.s_axi_awlock		( s_axi_awlock		),
		.s_axi_awcache		( s_axi_awcache		),
		.s_axi_awprot		( s_axi_awprot		),
		.s_axi_awqos		( s_axi_awqos		),
		.s_axi_awregion		( s_axi_awregion	),
		.s_axi_awvalid		( s_axi_awvalid		),
		.s_axi_awready		( s_axi_awready		),
		// write data channel signals
		.s_axi_wdata		( s_axi_wdata		),
		.s_axi_wstrb		( s_axi_wstrb		),
		.s_axi_wlast		( s_axi_wlast		),
		.s_axi_wvalid		( s_axi_wvalid		),
		.s_axi_wready		( s_axi_wready		),
		// write response channel signals
		.s_axi_bid		( s_axi_bid		),
		.s_axi_bresp		( s_axi_bresp		),
		.s_axi_bvalid		( s_axi_bvalid		),
		.s_axi_bready		( s_axi_bready		),
		// write read address channel signals
		.s_axi_arid		( s_axi_arid		),
		.s_axi_araddr		( s_axi_araddr		),
		.s_axi_arlen		( s_axi_arlen		),
		.s_axi_arsize		( s_axi_arsize		),
		.s_axi_arburst		( s_axi_arburst		),
		.s_axi_arlock		( s_axi_arlock		),
		.s_axi_arcache		( s_axi_arcache		),
		.s_axi_arprot		( s_axi_arprot		),
		.s_axi_arqos		( s_axi_arqos		),
		.s_axi_arregion		( s_axi_arregion	),
		.s_axi_arvalid		( s_axi_arvalid		),
		.s_axi_arready		( s_axi_arready		),
		// write read data channel signals
		.s_axi_rid		( s_axi_rid		),
		.s_axi_rdata		( s_axi_rdata		),
		.s_axi_rresp		( s_axi_rresp		),
		.s_axi_rlast		( s_axi_rlast		),
		.s_axi_rvalid		( s_axi_rvalid		),
		.s_axi_rready		( s_axi_rready		),

		//----| interrupt signals
		.irq			( irq			),

		//----| npc interface signals
		.slv_ofs		( slv_ofs		),
		.slv_siz		( slv_siz		),
		.slv_stt		( slv_stt		),
		.slv_fin		( slv_fin		),
		.slv_bsy		( slv_bsy		)
		);

//----| np master module |------------------------------------------------------
wire	[3:0]	npc_req;
wire	[3:0]	npc_gnt;
wire	[3:0]	npc_rwn;
wire	[31:0]	npc_adr[3:0];
wire	[31:0]	npc_len[3:0];
wire	[31:0]	npc_wdt[3:0];
wire	[31:0]	npc_rdt[3:0];
wire	[3:0]	npc_ack;
npm		npm			(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( m_axi_arstn		),
		.m_axi_aclk		( m_axi_aclk		),

		// write addresss channel signals
		.m_axi_awid		( m_axi_awid		),
		.m_axi_awaddr		( m_axi_awaddr		),
		.m_axi_awlen		( m_axi_awlen		),
		.m_axi_awsize		( m_axi_awsize		),
		.m_axi_awburst		( m_axi_awburst		),
		.m_axi_awlock		( m_axi_awlock		),
		.m_axi_awcache		( m_axi_awcache		),
		.m_axi_awprot		( m_axi_awprot		),
		.m_axi_awqos		( m_axi_awqos		),
		.m_axi_awregion		( m_axi_awregion	),
		.m_axi_awvalid		( m_axi_awvalid		),
		.m_axi_awready		( m_axi_awready		),
		// write data channel signals
		.m_axi_wdata		( m_axi_wdata		),
		.m_axi_wstrb		( m_axi_wstrb		),
		.m_axi_wlast		( m_axi_wlast		),
		.m_axi_wvalid		( m_axi_wvalid		),
		.m_axi_wready		( m_axi_wready		),
		// write response channel signals
		.m_axi_bid		( m_axi_bid		),
		.m_axi_bresp		( m_axi_bresp		),
		.m_axi_bvalid		( m_axi_bvalid		),
		.m_axi_bready		( m_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( m_axi_arid		),
		.m_axi_araddr		( m_axi_araddr		),
		.m_axi_arlen		( m_axi_arlen		),
		.m_axi_arsize		( m_axi_arsize		),
		.m_axi_arburst		( m_axi_arburst		),
		.m_axi_arlock		( m_axi_arlock		),
		.m_axi_arcache		( m_axi_arcache		),
		.m_axi_arprot		( m_axi_arprot		),
		.m_axi_arqos		( m_axi_arqos		),
		.m_axi_arregion		( m_axi_arregion	),
		.m_axi_arvalid		( m_axi_arvalid		),
		.m_axi_arready		( m_axi_arready		),
		// write read data channel signals
		.m_axi_rid		( m_axi_rid		),
		.m_axi_rdata		( m_axi_rdata		),
		.m_axi_rresp		( m_axi_rresp		),
		.m_axi_rlast		( m_axi_rlast		),
		.m_axi_rvalid		( m_axi_rvalid		),
		.m_axi_rready		( m_axi_rready		),

		//----| np core #0 interface
		.npc0_req		( npc_req[0]		),
		.npc0_gnt		( npc_gnt[0]		),
		.npc0_rwn		( npc_rwn[0]		),
		.npc0_adr		( npc_adr[0]		),
		.npc0_len		( npc_len[0]		),
		.npc0_wdt		( npc_wdt[0]		),
		.npc0_rdt		( npc_rdt[0]		),
		.npc0_ack		( npc_ack[0]		),

		//----| np core #1 interface
		.npc1_req		( npc_req[1]		),
		.npc1_gnt		( npc_gnt[1]		),
		.npc1_rwn		( npc_rwn[1]		),
		.npc1_adr		( npc_adr[1]		),
		.npc1_len		( npc_len[1]		),
		.npc1_wdt		( npc_wdt[1]		),
		.npc1_rdt		( npc_rdt[1]		),
		.npc1_ack		( npc_ack[1]		),

		//----| np core #2 interface
		.npc2_req		( npc_req[2]		),
		.npc2_gnt		( npc_gnt[2]		),
		.npc2_rwn		( npc_rwn[2]		),
		.npc2_adr		( npc_adr[2]		),
		.npc2_len		( npc_len[2]		),
		.npc2_wdt		( npc_wdt[2]		),
		.npc2_rdt		( npc_rdt[2]		),
		.npc2_ack		( npc_ack[2]		),

		//----| np core #3 interface
		.npc3_req		( npc_req[3]		),
		.npc3_gnt		( npc_gnt[3]		),
		.npc3_rwn		( npc_rwn[3]		),
		.npc3_adr		( npc_adr[3]		),
		.npc3_len		( npc_len[3]		),
		.npc3_wdt		( npc_wdt[3]		),
		.npc3_rdt		( npc_rdt[3]		),
		.npc3_ack		( npc_ack[3]		)
		);

//----| npc #0 |----------------------------------------------------------------
genvar	gi;
for(gi = 0;gi < 4;gi = gi + 1)
npc		npc		(
		//----| system interface
		.rstn		( m_axi_arstn	),
		.clk		( m_axi_aclk	),

		//----| slave interface
		.slv_stt	( slv_stt[gi]	),
		.slv_fin	( slv_fin[gi]	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy[gi]	),

		//----| npc interface
		.npc_req	( npc_req[gi]	),
		.npc_gnt	( npc_gnt[gi]	),
		.npc_rwn	( npc_rwn[gi]	),
		.npc_adr	( npc_adr[gi]	),
		.npc_len	( npc_len[gi]	),
		.npc_wdt	( npc_wdt[gi]	),
		.npc_rdt	( npc_rdt[gi]	),
		.npc_ack	( npc_ack[gi]	)
		);

//----| led |-------------------------------------------------------------------
assign		led[0]		= slv_bsy[0];
assign		led[1]		= slv_bsy[0];
assign		led[2]		= slv_bsy[2];
assign		led[3]		= slv_bsy[3];

endmodule
//==============================================================================
