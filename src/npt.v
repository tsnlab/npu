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
		//----| axi #0 master interface
		// axi system signals
		input	wire		m0_axi_arstn,
		input	wire		m0_axi_aclk,
		// write addresss channel signals
		output	wire	[5:0]	m0_axi_awid,
		output	wire	[31:0]	m0_axi_awaddr,
		output	wire	[7:0]	m0_axi_awlen,
		output	wire	[2:0]	m0_axi_awsize,
		output	wire	[1:0]	m0_axi_awburst,
		output	wire		m0_axi_awlock,
		output	wire	[3:0]	m0_axi_awcache,
		output	wire	[2:0]	m0_axi_awprot,
		output	wire	[3:0]	m0_axi_awqos,
		output	wire	[3:0]	m0_axi_awregion,
		// input	wire	[0:0]	m0_axi_awuser,
		output	wire		m0_axi_awvalid,
		input	wire		m0_axi_awready,
		// write data channel signals
		output	wire	[63:0]	m0_axi_wdata,
		output	wire	[7:0]	m0_axi_wstrb,
		output	wire		m0_axi_wlast,
		// input	wire	[0:0]	m0_axi_wuser,
		output	wire		m0_axi_wvalid,
		input	wire		m0_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m0_axi_bid,
		input	wire	[1:0]	m0_axi_bresp,
		// output	wire	[0:0]	m0_axi_buser,
		input	wire		m0_axi_bvalid,
		output	wire		m0_axi_bready,
		// write read address channel signals
		output	wire	[5:0]	m0_axi_arid,
		output	wire	[31:0]	m0_axi_araddr,
		output	wire	[7:0]	m0_axi_arlen,
		output	wire	[2:0]	m0_axi_arsize,
		output	wire	[1:0]	m0_axi_arburst,
		output	wire		m0_axi_arlock,
		output	wire	[3:0]	m0_axi_arcache,
		output	wire	[2:0]	m0_axi_arprot,
		output	wire	[3:0]	m0_axi_arqos,
		output	wire	[3:0]	m0_axi_arregion,
		// input	wire	[0:0]	m0_axi_aruser,
		output	wire		m0_axi_arvalid,
		input	wire		m0_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m0_axi_rid,
		input	wire	[63:0]	m0_axi_rdata,
		input	wire	[1:0]	m0_axi_rresp,
		input	wire		m0_axi_rlast,
		// output	wire	[0:0]	m0_axi_ruser,
		input	wire		m0_axi_rvalid,
		output	wire		m0_axi_rready,

		//----| axi #1 master interface
		// axi system signals
		input	wire		m1_axi_arstn,
		input	wire		m1_axi_aclk,
		// write addresss channel signals
		output	wire	[5:0]	m1_axi_awid,
		output	wire	[31:0]	m1_axi_awaddr,
		output	wire	[7:0]	m1_axi_awlen,
		output	wire	[2:0]	m1_axi_awsize,
		output	wire	[1:0]	m1_axi_awburst,
		output	wire		m1_axi_awlock,
		output	wire	[3:0]	m1_axi_awcache,
		output	wire	[2:0]	m1_axi_awprot,
		output	wire	[3:0]	m1_axi_awqos,
		output	wire	[3:0]	m1_axi_awregion,
		// input	wire	[0:0]	m1_axi_awuser,
		output	wire		m1_axi_awvalid,
		input	wire		m1_axi_awready,
		// write data channel signals
		output	wire	[63:0]	m1_axi_wdata,
		output	wire	[7:0]	m1_axi_wstrb,
		output	wire		m1_axi_wlast,
		// input	wire	[0:0]	m1_axi_wuser,
		output	wire		m1_axi_wvalid,
		input	wire		m1_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m1_axi_bid,
		input	wire	[1:0]	m1_axi_bresp,
		// output	wire	[0:0]	m1_axi_buser,
		input	wire		m1_axi_bvalid,
		output	wire		m1_axi_bready,
		// write read address channel signals
		output	wire	[5:0]	m1_axi_arid,
		output	wire	[31:0]	m1_axi_araddr,
		output	wire	[7:0]	m1_axi_arlen,
		output	wire	[2:0]	m1_axi_arsize,
		output	wire	[1:0]	m1_axi_arburst,
		output	wire		m1_axi_arlock,
		output	wire	[3:0]	m1_axi_arcache,
		output	wire	[2:0]	m1_axi_arprot,
		output	wire	[3:0]	m1_axi_arqos,
		output	wire	[3:0]	m1_axi_arregion,
		// input	wire	[0:0]	m1_axi_aruser,
		output	wire		m1_axi_arvalid,
		input	wire		m1_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m1_axi_rid,
		input	wire	[63:0]	m1_axi_rdata,
		input	wire	[1:0]	m1_axi_rresp,
		input	wire		m1_axi_rlast,
		// output	wire	[0:0]	m1_axi_ruser,
		input	wire		m1_axi_rvalid,
		output	wire		m1_axi_rready,

		//----| axi #2 master interface
		// axi system signals
		input	wire		m2_axi_arstn,
		input	wire		m2_axi_aclk,
		// write addresss channel signals
		output	wire	[5:0]	m2_axi_awid,
		output	wire	[31:0]	m2_axi_awaddr,
		output	wire	[7:0]	m2_axi_awlen,
		output	wire	[2:0]	m2_axi_awsize,
		output	wire	[1:0]	m2_axi_awburst,
		output	wire		m2_axi_awlock,
		output	wire	[3:0]	m2_axi_awcache,
		output	wire	[2:0]	m2_axi_awprot,
		output	wire	[3:0]	m2_axi_awqos,
		output	wire	[3:0]	m2_axi_awregion,
		// input	wire	[0:0]	m2_axi_awuser,
		output	wire		m2_axi_awvalid,
		input	wire		m2_axi_awready,
		// write data channel signals
		output	wire	[63:0]	m2_axi_wdata,
		output	wire	[7:0]	m2_axi_wstrb,
		output	wire		m2_axi_wlast,
		// input	wire	[0:0]	m2_axi_wuser,
		output	wire		m2_axi_wvalid,
		input	wire		m2_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m2_axi_bid,
		input	wire	[1:0]	m2_axi_bresp,
		// output	wire	[0:0]	m2_axi_buser,
		input	wire		m2_axi_bvalid,
		output	wire		m2_axi_bready,
		// write read address channel signals
		output	wire	[5:0]	m2_axi_arid,
		output	wire	[31:0]	m2_axi_araddr,
		output	wire	[7:0]	m2_axi_arlen,
		output	wire	[2:0]	m2_axi_arsize,
		output	wire	[1:0]	m2_axi_arburst,
		output	wire		m2_axi_arlock,
		output	wire	[3:0]	m2_axi_arcache,
		output	wire	[2:0]	m2_axi_arprot,
		output	wire	[3:0]	m2_axi_arqos,
		output	wire	[3:0]	m2_axi_arregion,
		// input	wire	[0:0]	m2_axi_aruser,
		output	wire		m2_axi_arvalid,
		input	wire		m2_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m2_axi_rid,
		input	wire	[63:0]	m2_axi_rdata,
		input	wire	[1:0]	m2_axi_rresp,
		input	wire		m2_axi_rlast,
		// output	wire	[0:0]	m2_axi_ruser,
		input	wire		m2_axi_rvalid,
		output	wire		m2_axi_rready,

		//----| axi #3 master interface
		// axi system signals
		input	wire		m3_axi_arstn,
		input	wire		m3_axi_aclk,
		// write addresss channel signals
		output	wire	[5:0]	m3_axi_awid,
		output	wire	[31:0]	m3_axi_awaddr,
		output	wire	[7:0]	m3_axi_awlen,
		output	wire	[2:0]	m3_axi_awsize,
		output	wire	[1:0]	m3_axi_awburst,
		output	wire		m3_axi_awlock,
		output	wire	[3:0]	m3_axi_awcache,
		output	wire	[2:0]	m3_axi_awprot,
		output	wire	[3:0]	m3_axi_awqos,
		output	wire	[3:0]	m3_axi_awregion,
		// input	wire	[0:0]	m3_axi_awuser,
		output	wire		m3_axi_awvalid,
		input	wire		m3_axi_awready,
		// write data channel signals
		output	wire	[63:0]	m3_axi_wdata,
		output	wire	[7:0]	m3_axi_wstrb,
		output	wire		m3_axi_wlast,
		// input	wire	[0:0]	m3_axi_wuser,
		output	wire		m3_axi_wvalid,
		input	wire		m3_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m3_axi_bid,
		input	wire	[1:0]	m3_axi_bresp,
		// output	wire	[0:0]	m3_axi_buser,
		input	wire		m3_axi_bvalid,
		output	wire		m3_axi_bready,
		// write read address channel signals
		output	wire	[5:0]	m3_axi_arid,
		output	wire	[31:0]	m3_axi_araddr,
		output	wire	[7:0]	m3_axi_arlen,
		output	wire	[2:0]	m3_axi_arsize,
		output	wire	[1:0]	m3_axi_arburst,
		output	wire		m3_axi_arlock,
		output	wire	[3:0]	m3_axi_arcache,
		output	wire	[2:0]	m3_axi_arprot,
		output	wire	[3:0]	m3_axi_arqos,
		output	wire	[3:0]	m3_axi_arregion,
		// input	wire	[0:0]	m3_axi_aruser,
		output	wire		m3_axi_arvalid,
		input	wire		m3_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m3_axi_rid,
		input	wire	[63:0]	m3_axi_rdata,
		input	wire	[1:0]	m3_axi_rresp,
		input	wire		m3_axi_rlast,
		// output	wire	[0:0]	m3_axi_ruser,
		input	wire		m3_axi_rvalid,
		output	wire		m3_axi_rready,

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

//----| np master #0 |----------------------------------------------------------
wire	[3:0]	npc_req;
wire	[3:0]	npc_gnt;
wire	[3:0]	npc_rwn;
wire	[31:0]	npc_adr[3:0];
wire	[31:0]	npc_len[3:0];
wire	[63:0]	npc_wdt[3:0];
wire	[63:0]	npc_rdt[3:0];
wire	[3:0]	npc_ack;
npm		npm0			(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( m0_axi_arstn		),
		.m_axi_aclk		( m0_axi_aclk		),

		// write addresss channel signals
		.m_axi_awid		( m0_axi_awid		),
		.m_axi_awaddr		( m0_axi_awaddr		),
		.m_axi_awlen		( m0_axi_awlen		),
		.m_axi_awsize		( m0_axi_awsize		),
		.m_axi_awburst		( m0_axi_awburst	),
		.m_axi_awlock		( m0_axi_awlock		),
		.m_axi_awcache		( m0_axi_awcache	),
		.m_axi_awprot		( m0_axi_awprot		),
		.m_axi_awqos		( m0_axi_awqos		),
		.m_axi_awregion		( m0_axi_awregion	),
		.m_axi_awvalid		( m0_axi_awvalid	),
		.m_axi_awready		( m0_axi_awready	),
		// write data channel signals
		.m_axi_wdata		( m0_axi_wdata		),
		.m_axi_wstrb		( m0_axi_wstrb		),
		.m_axi_wlast		( m0_axi_wlast		),
		.m_axi_wvalid		( m0_axi_wvalid		),
		.m_axi_wready		( m0_axi_wready		),
		// write response channel signals
		.m_axi_bid		( m0_axi_bid		),
		.m_axi_bresp		( m0_axi_bresp		),
		.m_axi_bvalid		( m0_axi_bvalid		),
		.m_axi_bready		( m0_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( m0_axi_arid		),
		.m_axi_araddr		( m0_axi_araddr		),
		.m_axi_arlen		( m0_axi_arlen		),
		.m_axi_arsize		( m0_axi_arsize		),
		.m_axi_arburst		( m0_axi_arburst	),
		.m_axi_arlock		( m0_axi_arlock		),
		.m_axi_arcache		( m0_axi_arcache	),
		.m_axi_arprot		( m0_axi_arprot		),
		.m_axi_arqos		( m0_axi_arqos		),
		.m_axi_arregion		( m0_axi_arregion	),
		.m_axi_arvalid		( m0_axi_arvalid	),
		.m_axi_arready		( m0_axi_arready	),
		// write read data channel signals
		.m_axi_rid		( m0_axi_rid		),
		.m_axi_rdata		( m0_axi_rdata		),
		.m_axi_rresp		( m0_axi_rresp		),
		.m_axi_rlast		( m0_axi_rlast		),
		.m_axi_rvalid		( m0_axi_rvalid		),
		.m_axi_rready		( m0_axi_rready		),

		//----| np core interface
		.npc_req		( npc_req[0]		),
		.npc_gnt		( npc_gnt[0]		),
		.npc_rwn		( npc_rwn[0]		),
		.npc_adr		( npc_adr[0]		),
		.npc_len		( npc_len[0]		),
		.npc_wdt		( npc_wdt[0]		),
		.npc_rdt		( npc_rdt[0]		),
		.npc_ack		( npc_ack[0]		)
		);

npm		npm1			(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( m1_axi_arstn		),
		.m_axi_aclk		( m1_axi_aclk		),

		// write addresss channel signals
		.m_axi_awid		( m1_axi_awid		),
		.m_axi_awaddr		( m1_axi_awaddr		),
		.m_axi_awlen		( m1_axi_awlen		),
		.m_axi_awsize		( m1_axi_awsize		),
		.m_axi_awburst		( m1_axi_awburst	),
		.m_axi_awlock		( m1_axi_awlock		),
		.m_axi_awcache		( m1_axi_awcache	),
		.m_axi_awprot		( m1_axi_awprot		),
		.m_axi_awqos		( m1_axi_awqos		),
		.m_axi_awregion		( m1_axi_awregion	),
		.m_axi_awvalid		( m1_axi_awvalid	),
		.m_axi_awready		( m1_axi_awready	),
		// write data channel signals
		.m_axi_wdata		( m1_axi_wdata		),
		.m_axi_wstrb		( m1_axi_wstrb		),
		.m_axi_wlast		( m1_axi_wlast		),
		.m_axi_wvalid		( m1_axi_wvalid		),
		.m_axi_wready		( m1_axi_wready		),
		// write response channel signals
		.m_axi_bid		( m1_axi_bid		),
		.m_axi_bresp		( m1_axi_bresp		),
		.m_axi_bvalid		( m1_axi_bvalid		),
		.m_axi_bready		( m1_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( m1_axi_arid		),
		.m_axi_araddr		( m1_axi_araddr		),
		.m_axi_arlen		( m1_axi_arlen		),
		.m_axi_arsize		( m1_axi_arsize		),
		.m_axi_arburst		( m1_axi_arburst	),
		.m_axi_arlock		( m1_axi_arlock		),
		.m_axi_arcache		( m1_axi_arcache	),
		.m_axi_arprot		( m1_axi_arprot		),
		.m_axi_arqos		( m1_axi_arqos		),
		.m_axi_arregion		( m1_axi_arregion	),
		.m_axi_arvalid		( m1_axi_arvalid	),
		.m_axi_arready		( m1_axi_arready	),
		// write read data channel signals
		.m_axi_rid		( m1_axi_rid		),
		.m_axi_rdata		( m1_axi_rdata		),
		.m_axi_rresp		( m1_axi_rresp		),
		.m_axi_rlast		( m1_axi_rlast		),
		.m_axi_rvalid		( m1_axi_rvalid		),
		.m_axi_rready		( m1_axi_rready		),

		//----| np core interface
		.npc_req		( npc_req[1]		),
		.npc_gnt		( npc_gnt[1]		),
		.npc_rwn		( npc_rwn[1]		),
		.npc_adr		( npc_adr[1]		),
		.npc_len		( npc_len[1]		),
		.npc_wdt		( npc_wdt[1]		),
		.npc_rdt		( npc_rdt[1]		),
		.npc_ack		( npc_ack[1]		)
		);

npm		npm2			(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( m2_axi_arstn		),
		.m_axi_aclk		( m2_axi_aclk		),

		// write addresss channel signals
		.m_axi_awid		( m2_axi_awid		),
		.m_axi_awaddr		( m2_axi_awaddr		),
		.m_axi_awlen		( m2_axi_awlen		),
		.m_axi_awsize		( m2_axi_awsize		),
		.m_axi_awburst		( m2_axi_awburst	),
		.m_axi_awlock		( m2_axi_awlock		),
		.m_axi_awcache		( m2_axi_awcache	),
		.m_axi_awprot		( m2_axi_awprot		),
		.m_axi_awqos		( m2_axi_awqos		),
		.m_axi_awregion		( m2_axi_awregion	),
		.m_axi_awvalid		( m2_axi_awvalid	),
		.m_axi_awready		( m2_axi_awready	),
		// write data channel signals
		.m_axi_wdata		( m2_axi_wdata		),
		.m_axi_wstrb		( m2_axi_wstrb		),
		.m_axi_wlast		( m2_axi_wlast		),
		.m_axi_wvalid		( m2_axi_wvalid		),
		.m_axi_wready		( m2_axi_wready		),
		// write response channel signals
		.m_axi_bid		( m2_axi_bid		),
		.m_axi_bresp		( m2_axi_bresp		),
		.m_axi_bvalid		( m2_axi_bvalid		),
		.m_axi_bready		( m2_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( m2_axi_arid		),
		.m_axi_araddr		( m2_axi_araddr		),
		.m_axi_arlen		( m2_axi_arlen		),
		.m_axi_arsize		( m2_axi_arsize		),
		.m_axi_arburst		( m2_axi_arburst	),
		.m_axi_arlock		( m2_axi_arlock		),
		.m_axi_arcache		( m2_axi_arcache	),
		.m_axi_arprot		( m2_axi_arprot		),
		.m_axi_arqos		( m2_axi_arqos		),
		.m_axi_arregion		( m2_axi_arregion	),
		.m_axi_arvalid		( m2_axi_arvalid	),
		.m_axi_arready		( m2_axi_arready	),
		// write read data channel signals
		.m_axi_rid		( m2_axi_rid		),
		.m_axi_rdata		( m2_axi_rdata		),
		.m_axi_rresp		( m2_axi_rresp		),
		.m_axi_rlast		( m2_axi_rlast		),
		.m_axi_rvalid		( m2_axi_rvalid		),
		.m_axi_rready		( m2_axi_rready		),

		//----| np core interface
		.npc_req		( npc_req[2]		),
		.npc_gnt		( npc_gnt[2]		),
		.npc_rwn		( npc_rwn[2]		),
		.npc_adr		( npc_adr[2]		),
		.npc_len		( npc_len[2]		),
		.npc_wdt		( npc_wdt[2]		),
		.npc_rdt		( npc_rdt[2]		),
		.npc_ack		( npc_ack[2]		)
		);

npm		npm3			(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( m3_axi_arstn		),
		.m_axi_aclk		( m3_axi_aclk		),

		// write addresss channel signals
		.m_axi_awid		( m3_axi_awid		),
		.m_axi_awaddr		( m3_axi_awaddr		),
		.m_axi_awlen		( m3_axi_awlen		),
		.m_axi_awsize		( m3_axi_awsize		),
		.m_axi_awburst		( m3_axi_awburst	),
		.m_axi_awlock		( m3_axi_awlock		),
		.m_axi_awcache		( m3_axi_awcache	),
		.m_axi_awprot		( m3_axi_awprot		),
		.m_axi_awqos		( m3_axi_awqos		),
		.m_axi_awregion		( m3_axi_awregion	),
		.m_axi_awvalid		( m3_axi_awvalid	),
		.m_axi_awready		( m3_axi_awready	),
		// write data channel signals
		.m_axi_wdata		( m3_axi_wdata		),
		.m_axi_wstrb		( m3_axi_wstrb		),
		.m_axi_wlast		( m3_axi_wlast		),
		.m_axi_wvalid		( m3_axi_wvalid		),
		.m_axi_wready		( m3_axi_wready		),
		// write response channel signals
		.m_axi_bid		( m3_axi_bid		),
		.m_axi_bresp		( m3_axi_bresp		),
		.m_axi_bvalid		( m3_axi_bvalid		),
		.m_axi_bready		( m3_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( m3_axi_arid		),
		.m_axi_araddr		( m3_axi_araddr		),
		.m_axi_arlen		( m3_axi_arlen		),
		.m_axi_arsize		( m3_axi_arsize		),
		.m_axi_arburst		( m3_axi_arburst	),
		.m_axi_arlock		( m3_axi_arlock		),
		.m_axi_arcache		( m3_axi_arcache	),
		.m_axi_arprot		( m3_axi_arprot		),
		.m_axi_arqos		( m3_axi_arqos		),
		.m_axi_arregion		( m3_axi_arregion	),
		.m_axi_arvalid		( m3_axi_arvalid	),
		.m_axi_arready		( m3_axi_arready	),
		// write read data channel signals
		.m_axi_rid		( m3_axi_rid		),
		.m_axi_rdata		( m3_axi_rdata		),
		.m_axi_rresp		( m3_axi_rresp		),
		.m_axi_rlast		( m3_axi_rlast		),
		.m_axi_rvalid		( m3_axi_rvalid		),
		.m_axi_rready		( m3_axi_rready		),

		//----| np core interface
		.npc_req		( npc_req[3]		),
		.npc_gnt		( npc_gnt[3]		),
		.npc_rwn		( npc_rwn[3]		),
		.npc_adr		( npc_adr[3]		),
		.npc_len		( npc_len[3]		),
		.npc_wdt		( npc_wdt[3]		),
		.npc_rdt		( npc_rdt[3]		),
		.npc_ack		( npc_ack[3]		)
		);

//----| npc #0 |----------------------------------------------------------------
npc		npc0		(
		//----| system interface
		.rstn		( m0_axi_arstn	),
		.clk		( m0_axi_aclk	),

		//----| slave interface
		.slv_stt	( slv_stt[0]	),
		.slv_fin	( slv_fin[0]	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy[0]	),

		//----| npc interface
		.npc_req	( npc_req[0]	),
		.npc_gnt	( npc_gnt[0]	),
		.npc_rwn	( npc_rwn[0]	),
		.npc_adr	( npc_adr[0]	),
		.npc_len	( npc_len[0]	),
		.npc_wdt	( npc_wdt[0]	),
		.npc_rdt	( npc_rdt[0]	),
		.npc_ack	( npc_ack[0]	)
		);

//----| npc #1 |----------------------------------------------------------------
npc		npc1		(
		//----| system interface
		.rstn		( m1_axi_arstn	),
		.clk		( m1_axi_aclk	),

		//----| slave interface
		.slv_stt	( slv_stt[1]	),
		.slv_fin	( slv_fin[1]	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy[1]	),

		//----| npc interface
		.npc_req	( npc_req[1]	),
		.npc_gnt	( npc_gnt[1]	),
		.npc_rwn	( npc_rwn[1]	),
		.npc_adr	( npc_adr[1]	),
		.npc_len	( npc_len[1]	),
		.npc_wdt	( npc_wdt[1]	),
		.npc_rdt	( npc_rdt[1]	),
		.npc_ack	( npc_ack[1]	)
		);

//----| npc #2 |----------------------------------------------------------------
npc		npc2		(
		//----| system interface
		.rstn		( m2_axi_arstn	),
		.clk		( m2_axi_aclk	),

		//----| slave interface
		.slv_stt	( slv_stt[2]	),
		.slv_fin	( slv_fin[2]	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy[2]	),

		//----| npc interface
		.npc_req	( npc_req[2]	),
		.npc_gnt	( npc_gnt[2]	),
		.npc_rwn	( npc_rwn[2]	),
		.npc_adr	( npc_adr[2]	),
		.npc_len	( npc_len[2]	),
		.npc_wdt	( npc_wdt[2]	),
		.npc_rdt	( npc_rdt[2]	),
		.npc_ack	( npc_ack[2]	)
		);

//----| npc #3 |----------------------------------------------------------------
npc		npc3		(
		//----| system interface
		.rstn		( m3_axi_arstn	),
		.clk		( m3_axi_aclk	),

		//----| slave interface
		.slv_stt	( slv_stt[3]	),
		.slv_fin	( slv_fin[3]	),
		.slv_ofs	( slv_ofs	),
		.slv_siz	( slv_siz	),
		.slv_bsy	( slv_bsy[3]	),

		//----| npc interface
		.npc_req	( npc_req[3]	),
		.npc_gnt	( npc_gnt[3]	),
		.npc_rwn	( npc_rwn[3]	),
		.npc_adr	( npc_adr[3]	),
		.npc_len	( npc_len[3]	),
		.npc_wdt	( npc_wdt[3]	),
		.npc_rdt	( npc_rdt[3]	),
		.npc_ack	( npc_ack[3]	)
		);

//----| led |-------------------------------------------------------------------
assign		led[0]		= slv_bsy[0];
assign		led[1]		= slv_bsy[0];
assign		led[2]		= slv_bsy[2];
assign		led[3]		= slv_bsy[3];

endmodule
//==============================================================================
