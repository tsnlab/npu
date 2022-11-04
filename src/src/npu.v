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
wire	[31:0]	m_axi_araddr;
wire	[1:0]	m_axi_arburst;
wire	[3:0]	m_axi_arcache;
wire	[11:0]	m_axi_arid;
wire	[7:0]	m_axi_arlen;
wire	[0:0]	m_axi_arlock;
wire	[2:0]	m_axi_arprot;
wire	[3:0]	m_axi_arqos;
wire		m_axi_arready;
wire	[3:0]	m_axi_arregion;
wire	[2:0]	m_axi_arsize;
wire		m_axi_arvalid;
wire	[31:0]	m_axi_awaddr;
wire	[1:0]	m_axi_awburst;
wire	[3:0]	m_axi_awcache;
wire	[11:0]	m_axi_awid;
wire	[7:0]	m_axi_awlen;
wire	[0:0]	m_axi_awlock;
wire	[2:0]	m_axi_awprot;
wire	[3:0]	m_axi_awqos;
wire		m_axi_awready;
wire	[3:0]	m_axi_awregion;
wire	[2:0]	m_axi_awsize;
wire		m_axi_awvalid;
wire	[11:0]	m_axi_bid;
wire		m_axi_bready;
wire	[1:0]	m_axi_bresp;
wire		m_axi_bvalid;
wire	[31:0]	m_axi_rdata;
wire	[11:0]	m_axi_rid;
wire		m_axi_rlast;
wire		m_axi_rready;
wire	[1:0]	m_axi_rresp;
wire		m_axi_rvalid;
wire	[31:0]	m_axi_wdata;
wire		m_axi_wlast;
wire		m_axi_wready;
wire	[3:0]	m_axi_wstrb;
wire		m_axi_wvalid;
wire	[31:0]	s_axi_araddr;
wire	[1:0]	s_axi_arburst;
wire	[3:0]	s_axi_arcache;
wire	[5:0]	s_axi_arid;
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
wire	[5:0]	s_axi_awid;
wire	[7:0]	s_axi_awlen;
wire	[0:0]	s_axi_awlock;
wire	[2:0]	s_axi_awprot;
wire	[3:0]	s_axi_awqos;
wire		s_axi_awready;
wire	[3:0]	s_axi_awregion;
wire	[2:0]	s_axi_awsize;
wire		s_axi_awvalid;
wire	[5:0]	s_axi_bid;
wire		s_axi_bready;
wire	[1:0]	s_axi_bresp;
wire		s_axi_bvalid;
wire	[31:0]	s_axi_rdata;
wire	[5:0]	s_axi_rid;
wire		s_axi_rlast;
wire		s_axi_rready;
wire	[1:0]	s_axi_rresp;
wire		s_axi_rvalid;
wire	[31:0]	s_axi_wdata;
wire		s_axi_wlast;
wire		s_axi_wready;
wire	[3:0]	s_axi_wstrb;
wire		s_axi_wvalid;
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
		.m_axi_araddr		( m_axi_araddr		),
		.m_axi_arburst		( m_axi_arburst		),
		.m_axi_arcache		( m_axi_arcache		),
		.m_axi_arid		( m_axi_arid		),
		.m_axi_arlen		( m_axi_arlen		),
		.m_axi_arlock		( m_axi_arlock		),
		.m_axi_arprot		( m_axi_arprot		),
		.m_axi_arqos		( m_axi_arqos		),
		.m_axi_arready		( m_axi_arready		),
		.m_axi_arregion		( m_axi_arregion	),
		.m_axi_arsize		( m_axi_arsize		),
		.m_axi_arvalid		( m_axi_arvalid		),
		.m_axi_awaddr		( m_axi_awaddr		),
		.m_axi_awburst		( m_axi_awburst		),
		.m_axi_awcache		( m_axi_awcache		),
		.m_axi_awid		( m_axi_awid		),
		.m_axi_awlen		( m_axi_awlen		),
		.m_axi_awlock		( m_axi_awlock		),
		.m_axi_awprot		( m_axi_awprot		),
		.m_axi_awqos		( m_axi_awqos		),
		.m_axi_awready		( m_axi_awready		),
		.m_axi_awregion		( m_axi_awregion	),
		.m_axi_awsize		( m_axi_awsize		),
		.m_axi_awvalid		( m_axi_awvalid		),
		.m_axi_bid		( m_axi_bid		),
		.m_axi_bready		( m_axi_bready		),
		.m_axi_bresp		( m_axi_bresp		),
		.m_axi_bvalid		( m_axi_bvalid		),
		.m_axi_rdata		( m_axi_rdata		),
		.m_axi_rid		( m_axi_rid		),
		.m_axi_rlast		( m_axi_rlast		),
		.m_axi_rready		( m_axi_rready		),
		.m_axi_rresp		( m_axi_rresp		),
		.m_axi_rvalid		( m_axi_rvalid		),
		.m_axi_wdata		( m_axi_wdata		),
		.m_axi_wlast		( m_axi_wlast		),
		.m_axi_wready		( m_axi_wready		),
		.m_axi_wstrb		( m_axi_wstrb		),
		.m_axi_wvalid		( m_axi_wvalid		),
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
		.s_axi_wvalid		( s_axi_wvalid		)
		);

//----| npu top |---------------------------------------------------------------
npt		npt		(
		//----| axi master interface
		// system signals
		.m_axi_arstn		( rstn			),
		.m_axi_aclk		( aclk			),

		// write addresss channel signals
		.m_axi_awid		( s_axi_awid		),
		.m_axi_awaddr		( s_axi_awaddr		),
		.m_axi_awlen		( s_axi_awlen		),
		.m_axi_awsize		( s_axi_awsize		),
		.m_axi_awburst		( s_axi_awburst		),
		.m_axi_awlock		( s_axi_awlock		),
		.m_axi_awcache		( s_axi_awcache		),
		.m_axi_awprot		( s_axi_awprot		),
		.m_axi_awqos		( s_axi_awqos		),
		.m_axi_awregion		( s_axi_awregion	),
		.m_axi_awvalid		( s_axi_awvalid		),
		.m_axi_awready		( s_axi_awready		),
		// write data channel signals
		.m_axi_wdata		( s_axi_wdata		),
		.m_axi_wstrb		( s_axi_wstrb		),
		.m_axi_wlast		( s_axi_wlast		),
		.m_axi_wvalid		( s_axi_wvalid		),
		.m_axi_wready		( s_axi_wready		),
		// write response channel signals
		.m_axi_bid		( s_axi_bid		),
		.m_axi_bresp		( s_axi_bresp		),
		.m_axi_bvalid		( s_axi_bvalid		),
		.m_axi_bready		( s_axi_bready		),
		// write read address channel signals
		.m_axi_arid		( s_axi_arid		),
		.m_axi_araddr		( s_axi_araddr		),
		.m_axi_arlen		( s_axi_arlen		),
		.m_axi_arsize		( s_axi_arsize		),
		.m_axi_arburst		( s_axi_arburst		),
		.m_axi_arlock		( s_axi_arlock		),
		.m_axi_arcache		( s_axi_arcache		),
		.m_axi_arprot		( s_axi_arprot		),
		.m_axi_arqos		( s_axi_arqos		),
		.m_axi_arregion		( s_axi_arregion		),
		.m_axi_arvalid		( s_axi_arvalid		),
		.m_axi_arready		( s_axi_arready		),
		// write read data channel signals
		.m_axi_rid		( s_axi_rid		),
		.m_axi_rdata		( s_axi_rdata		),
		.m_axi_rresp		( s_axi_rresp		),
		.m_axi_rlast		( s_axi_rlast		),
		.m_axi_rvalid		( s_axi_rvalid		),
		.m_axi_rready		( s_axi_rready		),

		//----| axi slave interface
		// axi system signals
		.s_axi_arstn		( rstn			),
		.s_axi_aclk		( aclk			),

		// write addresss channel signals
		.s_axi_awid		( m_axi_awid		),
		.s_axi_awaddr		( m_axi_awaddr		),
		.s_axi_awlen		( m_axi_awlen		),
		.s_axi_awsize		( m_axi_awsize		),
		.s_axi_awburst		( m_axi_awburst		),
		.s_axi_awlock		( m_axi_awlock		),
		.s_axi_awcache		( m_axi_awcache		),
		.s_axi_awprot		( m_axi_awprot		),
		.s_axi_awqos		( m_axi_awqos		),
		.s_axi_awregion		( m_axi_awregion	),
		.s_axi_awvalid		( m_axi_awvalid		),
		.s_axi_awready		( m_axi_awready		),
		// write data channel signals
		.s_axi_wdata		( m_axi_wdata		),
		.s_axi_wstrb		( m_axi_wstrb		),
		.s_axi_wlast		( m_axi_wlast		),
		.s_axi_wvalid		( m_axi_wvalid		),
		.s_axi_wready		( m_axi_wready		),
		// write response channel signals
		.s_axi_bid		( m_axi_bid		),
		.s_axi_bresp		( m_axi_bresp		),
		.s_axi_bvalid		( m_axi_bvalid		),
		.s_axi_bready		( m_axi_bready		),
		// write read address channel signals
		.s_axi_arid		( m_axi_arid		),
		.s_axi_araddr		( m_axi_araddr		),
		.s_axi_arlen		( m_axi_arlen		),
		.s_axi_arsize		( m_axi_arsize		),
		.s_axi_arburst		( m_axi_arburst		),
		.s_axi_arlock		( m_axi_arlock		),
		.s_axi_arcache		( m_axi_arcache		),
		.s_axi_arprot		( m_axi_arprot		),
		.s_axi_arqos		( m_axi_arqos		),
		.s_axi_arregion		( m_axi_arregion		),
		.s_axi_arvalid		( m_axi_arvalid		),
		.s_axi_arready		( m_axi_arready		),
		// write read data channel signals
		.s_axi_rid		( m_axi_rid		),
		.s_axi_rdata		( m_axi_rdata		),
		.s_axi_rresp		( m_axi_rresp		),
		.s_axi_rlast		( m_axi_rlast		),
		.s_axi_rvalid		( m_axi_rvalid		),
		.s_axi_rready		( m_axi_rready		),

		//----| interrupt signals
		.irq			( irq			),

		//----| led interface
		.led			( led			)
		);

endmodule
//==============================================================================
