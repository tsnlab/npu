/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu module
File name	: npu.v
Module name	: npt
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the npu  module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module npu	(
		//----| led interface
		output  wire	[3:0]	led		// led out
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| processing system |-----------------------------------------------------
wire	[3:0]	irq;
wire	[31:0]	s_axi_araddr;
wire	[1:0]	s_axi_arburst;
wire	[3:0]	s_axi_arcache;
wire	[11:0]	s_axi_arid;
wire	[7:0]	s_axi_arlen;
wire	[0:0]	s_axi_arlock;
wire	[2:0]	s_axi_arprot;
wire	[3:0]	s_axi_arqos;
wire		s_axi_arready;
wire	[3:0]	s_axi_arregion;
wire	[2:0]	s_axi_arsize;
wire		s_axi_arvalid;
wire	[31:0]	s_axi_awaddr;
wire	[1:0]	s_axi_awburst;
wire	[3:0]	s_axi_awcache;
wire	[11:0]	s_axi_awid;
wire	[7:0]	s_axi_awlen;
wire	[0:0]	s_axi_awlock;
wire	[2:0]	s_axi_awprot;
wire	[3:0]	s_axi_awqos;
wire		s_axi_awready;
wire	[3:0]	s_axi_awregion;
wire	[2:0]	s_axi_awsize;
wire		s_axi_awvalid;
wire	[11:0]	s_axi_bid;
wire		s_axi_bready;
wire	[1:0]	s_axi_bresp;
wire		s_axi_bvalid;
wire	[31:0]	s_axi_rdata;
wire	[11:0]	s_axi_rid;
wire		s_axi_rlast;
wire		s_axi_rready;
wire	[1:0]	s_axi_rresp;
wire		s_axi_rvalid;
wire	[31:0]	s_axi_wdata;
wire		s_axi_wlast;
wire		s_axi_wready;
wire	[3:0]	s_axi_wstrb;
wire		s_axi_wvalid;
wire	[31:0]	m0_axi_araddr;
wire	[1:0]	m0_axi_arburst;
wire	[3:0]	m0_axi_arcache;
wire	[5:0]	m0_axi_arid;
wire	[7:0]	m0_axi_arlen;
wire	[0:0]	m0_axi_arlock;
wire	[2:0]	m0_axi_arprot;
wire	[3:0]	m0_axi_arqos;
wire		m0_axi_arready;
wire	[3:0]	m0_axi_arregion;
wire	[2:0]	m0_axi_arsize;
wire		m0_axi_arvalid;
wire	[31:0]	m0_axi_awaddr;
wire	[1:0]	m0_axi_awburst;
wire	[3:0]	m0_axi_awcache;
wire	[5:0]	m0_axi_awid;
wire	[7:0]	m0_axi_awlen;
wire	[0:0]	m0_axi_awlock;
wire	[2:0]	m0_axi_awprot;
wire	[3:0]	m0_axi_awqos;
wire		m0_axi_awready;
wire	[3:0]	m0_axi_awregion;
wire	[2:0]	m0_axi_awsize;
wire		m0_axi_awvalid;
wire	[5:0]	m0_axi_bid;
wire		m0_axi_bready;
wire	[1:0]	m0_axi_bresp;
wire		m0_axi_bvalid;
wire	[63:0]	m0_axi_rdata;
wire	[5:0]	m0_axi_rid;
wire		m0_axi_rlast;
wire		m0_axi_rready;
wire	[1:0]	m0_axi_rresp;
wire		m0_axi_rvalid;
wire	[63:0]	m0_axi_wdata;
wire		m0_axi_wlast;
wire		m0_axi_wready;
wire	[7:0]	m0_axi_wstrb;
wire		m0_axi_wvalid;
wire	[31:0]	m1_axi_araddr;
wire	[1:0]	m1_axi_arburst;
wire	[3:0]	m1_axi_arcache;
wire	[5:0]	m1_axi_arid;
wire	[7:0]	m1_axi_arlen;
wire	[0:0]	m1_axi_arlock;
wire	[2:0]	m1_axi_arprot;
wire	[3:0]	m1_axi_arqos;
wire		m1_axi_arready;
wire	[3:0]	m1_axi_arregion;
wire	[2:0]	m1_axi_arsize;
wire		m1_axi_arvalid;
wire	[31:0]	m1_axi_awaddr;
wire	[1:0]	m1_axi_awburst;
wire	[3:0]	m1_axi_awcache;
wire	[5:0]	m1_axi_awid;
wire	[7:0]	m1_axi_awlen;
wire	[0:0]	m1_axi_awlock;
wire	[2:0]	m1_axi_awprot;
wire	[3:0]	m1_axi_awqos;
wire		m1_axi_awready;
wire	[3:0]	m1_axi_awregion;
wire	[2:0]	m1_axi_awsize;
wire		m1_axi_awvalid;
wire	[5:0]	m1_axi_bid;
wire		m1_axi_bready;
wire	[1:0]	m1_axi_bresp;
wire		m1_axi_bvalid;
wire	[63:0]	m1_axi_rdata;
wire	[5:0]	m1_axi_rid;
wire		m1_axi_rlast;
wire		m1_axi_rready;
wire	[1:0]	m1_axi_rresp;
wire		m1_axi_rvalid;
wire	[63:0]	m1_axi_wdata;
wire		m1_axi_wlast;
wire		m1_axi_wready;
wire	[7:0]	m1_axi_wstrb;
wire		m1_axi_wvalid;
wire	[31:0]	m2_axi_araddr;
wire	[1:0]	m2_axi_arburst;
wire	[3:0]	m2_axi_arcache;
wire	[5:0]	m2_axi_arid;
wire	[7:0]	m2_axi_arlen;
wire	[0:0]	m2_axi_arlock;
wire	[2:0]	m2_axi_arprot;
wire	[3:0]	m2_axi_arqos;
wire		m2_axi_arready;
wire	[3:0]	m2_axi_arregion;
wire	[2:0]	m2_axi_arsize;
wire		m2_axi_arvalid;
wire	[31:0]	m2_axi_awaddr;
wire	[1:0]	m2_axi_awburst;
wire	[3:0]	m2_axi_awcache;
wire	[5:0]	m2_axi_awid;
wire	[7:0]	m2_axi_awlen;
wire	[0:0]	m2_axi_awlock;
wire	[2:0]	m2_axi_awprot;
wire	[3:0]	m2_axi_awqos;
wire		m2_axi_awready;
wire	[3:0]	m2_axi_awregion;
wire	[2:0]	m2_axi_awsize;
wire		m2_axi_awvalid;
wire	[5:0]	m2_axi_bid;
wire		m2_axi_bready;
wire	[1:0]	m2_axi_bresp;
wire		m2_axi_bvalid;
wire	[63:0]	m2_axi_rdata;
wire	[5:0]	m2_axi_rid;
wire		m2_axi_rlast;
wire		m2_axi_rready;
wire	[1:0]	m2_axi_rresp;
wire		m2_axi_rvalid;
wire	[63:0]	m2_axi_wdata;
wire		m2_axi_wlast;
wire		m2_axi_wready;
wire	[7:0]	m2_axi_wstrb;
wire		m2_axi_wvalid;
wire	[31:0]	m3_axi_araddr;
wire	[1:0]	m3_axi_arburst;
wire	[3:0]	m3_axi_arcache;
wire	[5:0]	m3_axi_arid;
wire	[7:0]	m3_axi_arlen;
wire	[0:0]	m3_axi_arlock;
wire	[2:0]	m3_axi_arprot;
wire	[3:0]	m3_axi_arqos;
wire		m3_axi_arready;
wire	[3:0]	m3_axi_arregion;
wire	[2:0]	m3_axi_arsize;
wire		m3_axi_arvalid;
wire	[31:0]	m3_axi_awaddr;
wire	[1:0]	m3_axi_awburst;
wire	[3:0]	m3_axi_awcache;
wire	[5:0]	m3_axi_awid;
wire	[7:0]	m3_axi_awlen;
wire	[0:0]	m3_axi_awlock;
wire	[2:0]	m3_axi_awprot;
wire	[3:0]	m3_axi_awqos;
wire		m3_axi_awready;
wire	[3:0]	m3_axi_awregion;
wire	[2:0]	m3_axi_awsize;
wire		m3_axi_awvalid;
wire	[5:0]	m3_axi_bid;
wire		m3_axi_bready;
wire	[1:0]	m3_axi_bresp;
wire		m3_axi_bvalid;
wire	[63:0]	m3_axi_rdata;
wire	[5:0]	m3_axi_rid;
wire		m3_axi_rlast;
wire		m3_axi_rready;
wire	[1:0]	m3_axi_rresp;
wire		m3_axi_rvalid;
wire	[63:0]	m3_axi_wdata;
wire		m3_axi_wlast;
wire		m3_axi_wready;
wire	[7:0]	m3_axi_wstrb;
wire		m3_axi_wvalid;
ps		ps			(
		.DDR_addr		(			),
		.DDR_ba			(			),
		.DDR_cas_n		(			),
		.DDR_ck_n		(			),
		.DDR_ck_p		(			),
		.DDR_cke		(			),
		.DDR_cs_n		(			),
		.DDR_dm			(			),
		.DDR_dq			(			),
		.DDR_dqs_n		(			),
		.DDR_dqs_p		(			),
		.DDR_odt		(			),
		.DDR_ras_n		(			),
		.DDR_reset_n		(			),
		.DDR_we_n		(			),
		.FIXED_IO_ddr_vrn	(			),
		.FIXED_IO_ddr_vrp	(			),
		.FIXED_IO_mio		(			),
		.FIXED_IO_ps_clk	(			),
		.FIXED_IO_ps_porb	(			),
		.FIXED_IO_ps_srstb	(			),
		.orstn			( rstn			),
		.oclk			( aclk			),
		.irq			( irq			),
		.s_axi_araddr		( s_axi_araddr		),
		.s_axi_arburst		( s_axi_arburst		),
		.s_axi_arcache		( s_axi_arcache		),
		.s_axi_arid		( s_axi_arid		),
		.s_axi_arlen		( s_axi_arlen		),
		.s_axi_arlock		( s_axi_arlock		),
		.s_axi_arprot		( s_axi_arprot		),
		.s_axi_arqos		( s_axi_arqos		),
		.s_axi_arready		( s_axi_arready		),
		.s_axi_arregion		( s_axi_arregion	),
		.s_axi_arsize		( s_axi_arsize		),
		.s_axi_arvalid		( s_axi_arvalid		),
		.s_axi_awaddr		( s_axi_awaddr		),
		.s_axi_awburst		( s_axi_awburst		),
		.s_axi_awcache		( s_axi_awcache		),
		.s_axi_awid		( s_axi_awid		),
		.s_axi_awlen		( s_axi_awlen		),
		.s_axi_awlock		( s_axi_awlock		),
		.s_axi_awprot		( s_axi_awprot		),
		.s_axi_awqos		( s_axi_awqos		),
		.s_axi_awready		( s_axi_awready		),
		.s_axi_awregion		( s_axi_awregion	),
		.s_axi_awsize		( s_axi_awsize		),
		.s_axi_awvalid		( s_axi_awvalid		),
		.s_axi_bid		( s_axi_bid		),
		.s_axi_bready		( s_axi_bready		),
		.s_axi_bresp		( s_axi_bresp		),
		.s_axi_bvalid		( s_axi_bvalid		),
		.s_axi_rdata		( s_axi_rdata		),
		.s_axi_rid		( s_axi_rid		),
		.s_axi_rlast		( s_axi_rlast		),
		.s_axi_rready		( s_axi_rready		),
		.s_axi_rresp		( s_axi_rresp		),
		.s_axi_rvalid		( s_axi_rvalid		),
		.s_axi_wdata		( s_axi_wdata		),
		.s_axi_wlast		( s_axi_wlast		),
		.s_axi_wready		( s_axi_wready		),
		.s_axi_wstrb		( s_axi_wstrb		),
		.s_axi_wvalid		( s_axi_wvalid		),
		.m0_axi_araddr		( m0_axi_araddr		),
		.m0_axi_arburst		( m0_axi_arburst	),
		.m0_axi_arcache		( m0_axi_arcache	),
		.m0_axi_arid		( m0_axi_arid		),
		.m0_axi_arlen		( m0_axi_arlen		),
		.m0_axi_arlock		( m0_axi_arlock		),
		.m0_axi_arprot		( m0_axi_arprot		),
		.m0_axi_arqos		( m0_axi_arqos		),
		.m0_axi_arready		( m0_axi_arready	),
		.m0_axi_arregion	( m0_axi_arregion	),
		.m0_axi_arsize		( m0_axi_arsize		),
		.m0_axi_arvalid		( m0_axi_arvalid	),
		.m0_axi_awaddr		( m0_axi_awaddr		),
		.m0_axi_awburst		( m0_axi_awburst	),
		.m0_axi_awcache		( m0_axi_awcache	),
		.m0_axi_awid		( m0_axi_awid		),
		.m0_axi_awlen		( m0_axi_awlen		),
		.m0_axi_awlock		( m0_axi_awlock		),
		.m0_axi_awprot		( m0_axi_awprot		),
		.m0_axi_awqos		( m0_axi_awqos		),
		.m0_axi_awready		( m0_axi_awready	),
		.m0_axi_awregion	( m0_axi_awregion	),
		.m0_axi_awsize		( m0_axi_awsize		),
		.m0_axi_awvalid		( m0_axi_awvalid	),
		.m0_axi_bid		( m0_axi_bid		),
		.m0_axi_bready		( m0_axi_bready		),
		.m0_axi_bresp		( m0_axi_bresp		),
		.m0_axi_bvalid		( m0_axi_bvalid		),
		.m0_axi_rdata		( m0_axi_rdata		),
		.m0_axi_rid		( m0_axi_rid		),
		.m0_axi_rlast		( m0_axi_rlast		),
		.m0_axi_rready		( m0_axi_rready		),
		.m0_axi_rresp		( m0_axi_rresp		),
		.m0_axi_rvalid		( m0_axi_rvalid		),
		.m0_axi_wdata		( m0_axi_wdata		),
		.m0_axi_wlast		( m0_axi_wlast		),
		.m0_axi_wready		( m0_axi_wready		),
		.m0_axi_wstrb		( m0_axi_wstrb		),
		.m0_axi_wvalid		( m0_axi_wvalid		),
		.m1_axi_araddr		( m1_axi_araddr		),
		.m1_axi_arburst		( m1_axi_arburst	),
		.m1_axi_arcache		( m1_axi_arcache	),
		.m1_axi_arid		( m1_axi_arid		),
		.m1_axi_arlen		( m1_axi_arlen		),
		.m1_axi_arlock		( m1_axi_arlock		),
		.m1_axi_arprot		( m1_axi_arprot		),
		.m1_axi_arqos		( m1_axi_arqos		),
		.m1_axi_arready		( m1_axi_arready	),
		.m1_axi_arregion	( m1_axi_arregion	),
		.m1_axi_arsize		( m1_axi_arsize		),
		.m1_axi_arvalid		( m1_axi_arvalid	),
		.m1_axi_awaddr		( m1_axi_awaddr		),
		.m1_axi_awburst		( m1_axi_awburst	),
		.m1_axi_awcache		( m1_axi_awcache	),
		.m1_axi_awid		( m1_axi_awid		),
		.m1_axi_awlen		( m1_axi_awlen		),
		.m1_axi_awlock		( m1_axi_awlock		),
		.m1_axi_awprot		( m1_axi_awprot		),
		.m1_axi_awqos		( m1_axi_awqos		),
		.m1_axi_awready		( m1_axi_awready	),
		.m1_axi_awregion	( m1_axi_awregion	),
		.m1_axi_awsize		( m1_axi_awsize		),
		.m1_axi_awvalid		( m1_axi_awvalid	),
		.m1_axi_bid		( m1_axi_bid		),
		.m1_axi_bready		( m1_axi_bready		),
		.m1_axi_bresp		( m1_axi_bresp		),
		.m1_axi_bvalid		( m1_axi_bvalid		),
		.m1_axi_rdata		( m1_axi_rdata		),
		.m1_axi_rid		( m1_axi_rid		),
		.m1_axi_rlast		( m1_axi_rlast		),
		.m1_axi_rready		( m1_axi_rready		),
		.m1_axi_rresp		( m1_axi_rresp		),
		.m1_axi_rvalid		( m1_axi_rvalid		),
		.m1_axi_wdata		( m1_axi_wdata		),
		.m1_axi_wlast		( m1_axi_wlast		),
		.m1_axi_wready		( m1_axi_wready		),
		.m1_axi_wstrb		( m1_axi_wstrb		),
		.m1_axi_wvalid		( m1_axi_wvalid		),
		.m2_axi_araddr		( m2_axi_araddr		),
		.m2_axi_arburst		( m2_axi_arburst	),
		.m2_axi_arcache		( m2_axi_arcache	),
		.m2_axi_arid		( m2_axi_arid		),
		.m2_axi_arlen		( m2_axi_arlen		),
		.m2_axi_arlock		( m2_axi_arlock		),
		.m2_axi_arprot		( m2_axi_arprot		),
		.m2_axi_arqos		( m2_axi_arqos		),
		.m2_axi_arready		( m2_axi_arready	),
		.m2_axi_arregion	( m2_axi_arregion	),
		.m2_axi_arsize		( m2_axi_arsize		),
		.m2_axi_arvalid		( m2_axi_arvalid	),
		.m2_axi_awaddr		( m2_axi_awaddr		),
		.m2_axi_awburst		( m2_axi_awburst	),
		.m2_axi_awcache		( m2_axi_awcache	),
		.m2_axi_awid		( m2_axi_awid		),
		.m2_axi_awlen		( m2_axi_awlen		),
		.m2_axi_awlock		( m2_axi_awlock		),
		.m2_axi_awprot		( m2_axi_awprot		),
		.m2_axi_awqos		( m2_axi_awqos		),
		.m2_axi_awready		( m2_axi_awready	),
		.m2_axi_awregion	( m2_axi_awregion	),
		.m2_axi_awsize		( m2_axi_awsize		),
		.m2_axi_awvalid		( m2_axi_awvalid	),
		.m2_axi_bid		( m2_axi_bid		),
		.m2_axi_bready		( m2_axi_bready		),
		.m2_axi_bresp		( m2_axi_bresp		),
		.m2_axi_bvalid		( m2_axi_bvalid		),
		.m2_axi_rdata		( m2_axi_rdata		),
		.m2_axi_rid		( m2_axi_rid		),
		.m2_axi_rlast		( m2_axi_rlast		),
		.m2_axi_rready		( m2_axi_rready		),
		.m2_axi_rresp		( m2_axi_rresp		),
		.m2_axi_rvalid		( m2_axi_rvalid		),
		.m2_axi_wdata		( m2_axi_wdata		),
		.m2_axi_wlast		( m2_axi_wlast		),
		.m2_axi_wready		( m2_axi_wready		),
		.m2_axi_wstrb		( m2_axi_wstrb		),
		.m2_axi_wvalid		( m2_axi_wvalid		),
		.m3_axi_araddr		( m3_axi_araddr		),
		.m3_axi_arburst		( m3_axi_arburst	),
		.m3_axi_arcache		( m3_axi_arcache	),
		.m3_axi_arid		( m3_axi_arid		),
		.m3_axi_arlen		( m3_axi_arlen		),
		.m3_axi_arlock		( m3_axi_arlock		),
		.m3_axi_arprot		( m3_axi_arprot		),
		.m3_axi_arqos		( m3_axi_arqos		),
		.m3_axi_arready		( m3_axi_arready	),
		.m3_axi_arregion	( m3_axi_arregion	),
		.m3_axi_arsize		( m3_axi_arsize		),
		.m3_axi_arvalid		( m3_axi_arvalid	),
		.m3_axi_awaddr		( m3_axi_awaddr		),
		.m3_axi_awburst		( m3_axi_awburst	),
		.m3_axi_awcache		( m3_axi_awcache	),
		.m3_axi_awid		( m3_axi_awid		),
		.m3_axi_awlen		( m3_axi_awlen		),
		.m3_axi_awlock		( m3_axi_awlock		),
		.m3_axi_awprot		( m3_axi_awprot		),
		.m3_axi_awqos		( m3_axi_awqos		),
		.m3_axi_awready		( m3_axi_awready	),
		.m3_axi_awregion	( m3_axi_awregion	),
		.m3_axi_awsize		( m3_axi_awsize		),
		.m3_axi_awvalid		( m3_axi_awvalid	),
		.m3_axi_bid		( m3_axi_bid		),
		.m3_axi_bready		( m3_axi_bready		),
		.m3_axi_bresp		( m3_axi_bresp		),
		.m3_axi_bvalid		( m3_axi_bvalid		),
		.m3_axi_rdata		( m3_axi_rdata		),
		.m3_axi_rid		( m3_axi_rid		),
		.m3_axi_rlast		( m3_axi_rlast		),
		.m3_axi_rready		( m3_axi_rready		),
		.m3_axi_rresp		( m3_axi_rresp		),
		.m3_axi_rvalid		( m3_axi_rvalid		),
		.m3_axi_wdata		( m3_axi_wdata		),
		.m3_axi_wlast		( m3_axi_wlast		),
		.m3_axi_wready		( m3_axi_wready		),
		.m3_axi_wstrb		( m3_axi_wstrb		),
		.m3_axi_wvalid		( m3_axi_wvalid		)
		);

//----| npu top |---------------------------------------------------------------
npt		npt		(
		//----| axi #0 master interface
		// system signals
		.m0_axi_arstn		( rstn			),
		.m0_axi_aclk		( aclk			),

		// write addresss channel signals
		.m0_axi_awid		( m0_axi_awid		),
		.m0_axi_awaddr		( m0_axi_awaddr		),
		.m0_axi_awlen		( m0_axi_awlen		),
		.m0_axi_awsize		( m0_axi_awsize		),
		.m0_axi_awburst		( m0_axi_awburst	),
		.m0_axi_awlock		( m0_axi_awlock		),
		.m0_axi_awcache		( m0_axi_awcache	),
		.m0_axi_awprot		( m0_axi_awprot		),
		.m0_axi_awqos		( m0_axi_awqos		),
		.m0_axi_awregion	( m0_axi_awregion	),
		.m0_axi_awvalid		( m0_axi_awvalid	),
		.m0_axi_awready		( m0_axi_awready	),
		// write data channel signals
		.m0_axi_wdata		( m0_axi_wdata		),
		.m0_axi_wstrb		( m0_axi_wstrb		),
		.m0_axi_wlast		( m0_axi_wlast		),
		.m0_axi_wvalid		( m0_axi_wvalid		),
		.m0_axi_wready		( m0_axi_wready		),
		// write response channel signals
		.m0_axi_bid		( m0_axi_bid		),
		.m0_axi_bresp		( m0_axi_bresp		),
		.m0_axi_bvalid		( m0_axi_bvalid		),
		.m0_axi_bready		( m0_axi_bready		),
		// write read address channel signals
		.m0_axi_arid		( m0_axi_arid		),
		.m0_axi_araddr		( m0_axi_araddr		),
		.m0_axi_arlen		( m0_axi_arlen		),
		.m0_axi_arsize		( m0_axi_arsize		),
		.m0_axi_arburst		( m0_axi_arburst	),
		.m0_axi_arlock		( m0_axi_arlock		),
		.m0_axi_arcache		( m0_axi_arcache	),
		.m0_axi_arprot		( m0_axi_arprot		),
		.m0_axi_arqos		( m0_axi_arqos		),
		.m0_axi_arregion	( m0_axi_arregion	),
		.m0_axi_arvalid		( m0_axi_arvalid	),
		.m0_axi_arready		( m0_axi_arready	),
		// write read data channel signals
		.m0_axi_rid		( m0_axi_rid		),
		.m0_axi_rdata		( m0_axi_rdata		),
		.m0_axi_rresp		( m0_axi_rresp		),
		.m0_axi_rlast		( m0_axi_rlast		),
		.m0_axi_rvalid		( m0_axi_rvalid		),
		.m0_axi_rready		( m0_axi_rready		),

		//----| axi #1 master interface
		// system signals
		.m1_axi_arstn		( rstn			),
		.m1_axi_aclk		( aclk			),

		// write addresss channel signals
		.m1_axi_awid		( m1_axi_awid		),
		.m1_axi_awaddr		( m1_axi_awaddr		),
		.m1_axi_awlen		( m1_axi_awlen		),
		.m1_axi_awsize		( m1_axi_awsize		),
		.m1_axi_awburst		( m1_axi_awburst	),
		.m1_axi_awlock		( m1_axi_awlock		),
		.m1_axi_awcache		( m1_axi_awcache	),
		.m1_axi_awprot		( m1_axi_awprot		),
		.m1_axi_awqos		( m1_axi_awqos		),
		.m1_axi_awregion	( m1_axi_awregion	),
		.m1_axi_awvalid		( m1_axi_awvalid	),
		.m1_axi_awready		( m1_axi_awready	),
		// write data channel signals
		.m1_axi_wdata		( m1_axi_wdata		),
		.m1_axi_wstrb		( m1_axi_wstrb		),
		.m1_axi_wlast		( m1_axi_wlast		),
		.m1_axi_wvalid		( m1_axi_wvalid		),
		.m1_axi_wready		( m1_axi_wready		),
		// write response channel signals
		.m1_axi_bid		( m1_axi_bid		),
		.m1_axi_bresp		( m1_axi_bresp		),
		.m1_axi_bvalid		( m1_axi_bvalid		),
		.m1_axi_bready		( m1_axi_bready		),
		// write read address channel signals
		.m1_axi_arid		( m1_axi_arid		),
		.m1_axi_araddr		( m1_axi_araddr		),
		.m1_axi_arlen		( m1_axi_arlen		),
		.m1_axi_arsize		( m1_axi_arsize		),
		.m1_axi_arburst		( m1_axi_arburst	),
		.m1_axi_arlock		( m1_axi_arlock		),
		.m1_axi_arcache		( m1_axi_arcache	),
		.m1_axi_arprot		( m1_axi_arprot		),
		.m1_axi_arqos		( m1_axi_arqos		),
		.m1_axi_arregion	( m1_axi_arregion	),
		.m1_axi_arvalid		( m1_axi_arvalid	),
		.m1_axi_arready		( m1_axi_arready	),
		// write read data channel signals
		.m1_axi_rid		( m1_axi_rid		),
		.m1_axi_rdata		( m1_axi_rdata		),
		.m1_axi_rresp		( m1_axi_rresp		),
		.m1_axi_rlast		( m1_axi_rlast		),
		.m1_axi_rvalid		( m1_axi_rvalid		),
		.m1_axi_rready		( m1_axi_rready		),

		//----| axi #2 master interface
		// system signals
		.m2_axi_arstn		( rstn			),
		.m2_axi_aclk		( aclk			),

		// write addresss channel signals
		.m2_axi_awid		( m2_axi_awid		),
		.m2_axi_awaddr		( m2_axi_awaddr		),
		.m2_axi_awlen		( m2_axi_awlen		),
		.m2_axi_awsize		( m2_axi_awsize		),
		.m2_axi_awburst		( m2_axi_awburst	),
		.m2_axi_awlock		( m2_axi_awlock		),
		.m2_axi_awcache		( m2_axi_awcache	),
		.m2_axi_awprot		( m2_axi_awprot		),
		.m2_axi_awqos		( m2_axi_awqos		),
		.m2_axi_awregion	( m2_axi_awregion	),
		.m2_axi_awvalid		( m2_axi_awvalid	),
		.m2_axi_awready		( m2_axi_awready	),
		// write data channel signals
		.m2_axi_wdata		( m2_axi_wdata		),
		.m2_axi_wstrb		( m2_axi_wstrb		),
		.m2_axi_wlast		( m2_axi_wlast		),
		.m2_axi_wvalid		( m2_axi_wvalid		),
		.m2_axi_wready		( m2_axi_wready		),
		// write response channel signals
		.m2_axi_bid		( m2_axi_bid		),
		.m2_axi_bresp		( m2_axi_bresp		),
		.m2_axi_bvalid		( m2_axi_bvalid		),
		.m2_axi_bready		( m2_axi_bready		),
		// write read address channel signals
		.m2_axi_arid		( m2_axi_arid		),
		.m2_axi_araddr		( m2_axi_araddr		),
		.m2_axi_arlen		( m2_axi_arlen		),
		.m2_axi_arsize		( m2_axi_arsize		),
		.m2_axi_arburst		( m2_axi_arburst	),
		.m2_axi_arlock		( m2_axi_arlock		),
		.m2_axi_arcache		( m2_axi_arcache	),
		.m2_axi_arprot		( m2_axi_arprot		),
		.m2_axi_arqos		( m2_axi_arqos		),
		.m2_axi_arregion	( m2_axi_arregion	),
		.m2_axi_arvalid		( m2_axi_arvalid	),
		.m2_axi_arready		( m2_axi_arready	),
		// write read data channel signals
		.m2_axi_rid		( m2_axi_rid		),
		.m2_axi_rdata		( m2_axi_rdata		),
		.m2_axi_rresp		( m2_axi_rresp		),
		.m2_axi_rlast		( m2_axi_rlast		),
		.m2_axi_rvalid		( m2_axi_rvalid		),
		.m2_axi_rready		( m2_axi_rready		),

		//----| axi #3 master interface
		// system signals
		.m3_axi_arstn		( rstn			),
		.m3_axi_aclk		( aclk			),

		// write addresss channel signals
		.m3_axi_awid		( m3_axi_awid		),
		.m3_axi_awaddr		( m3_axi_awaddr		),
		.m3_axi_awlen		( m3_axi_awlen		),
		.m3_axi_awsize		( m3_axi_awsize		),
		.m3_axi_awburst		( m3_axi_awburst	),
		.m3_axi_awlock		( m3_axi_awlock		),
		.m3_axi_awcache		( m3_axi_awcache	),
		.m3_axi_awprot		( m3_axi_awprot		),
		.m3_axi_awqos		( m3_axi_awqos		),
		.m3_axi_awregion	( m3_axi_awregion	),
		.m3_axi_awvalid		( m3_axi_awvalid	),
		.m3_axi_awready		( m3_axi_awready	),
		// write data channel signals
		.m3_axi_wdata		( m3_axi_wdata		),
		.m3_axi_wstrb		( m3_axi_wstrb		),
		.m3_axi_wlast		( m3_axi_wlast		),
		.m3_axi_wvalid		( m3_axi_wvalid		),
		.m3_axi_wready		( m3_axi_wready		),
		// write response channel signals
		.m3_axi_bid		( m3_axi_bid		),
		.m3_axi_bresp		( m3_axi_bresp		),
		.m3_axi_bvalid		( m3_axi_bvalid		),
		.m3_axi_bready		( m3_axi_bready		),
		// write read address channel signals
		.m3_axi_arid		( m3_axi_arid		),
		.m3_axi_araddr		( m3_axi_araddr		),
		.m3_axi_arlen		( m3_axi_arlen		),
		.m3_axi_arsize		( m3_axi_arsize		),
		.m3_axi_arburst		( m3_axi_arburst	),
		.m3_axi_arlock		( m3_axi_arlock		),
		.m3_axi_arcache		( m3_axi_arcache	),
		.m3_axi_arprot		( m3_axi_arprot		),
		.m3_axi_arqos		( m3_axi_arqos		),
		.m3_axi_arregion	( m3_axi_arregion	),
		.m3_axi_arvalid		( m3_axi_arvalid	),
		.m3_axi_arready		( m3_axi_arready	),
		// write read data channel signals
		.m3_axi_rid		( m3_axi_rid		),
		.m3_axi_rdata		( m3_axi_rdata		),
		.m3_axi_rresp		( m3_axi_rresp		),
		.m3_axi_rlast		( m3_axi_rlast		),
		.m3_axi_rvalid		( m3_axi_rvalid		),
		.m3_axi_rready		( m3_axi_rready		),

		//----| axi slave interface
		// axi system signals
		.s_axi_arstn		( rstn			),
		.s_axi_aclk		( aclk			),

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
		.s_axi_arregion		( s_axi_arregion		),
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

		//----| led interface
		.led			( led			)
		);

endmodule
//==============================================================================
