/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu master module
File name	: npm.v
Module name	: npm
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the master(arbiter + dma) module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module npm	(
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
		output	wire		m_axi_awvalid,
		input	wire		m_axi_awready,
		// write data channel signals
		output	wire	[63:0]	m_axi_wdata,
		output	wire	[7:0]	m_axi_wstrb,
		output	wire		m_axi_wlast,
		output	wire		m_axi_wvalid,
		input	wire		m_axi_wready,
		// write response channel signals
		input	wire	[5:0]	m_axi_bid,
		input	wire	[1:0]	m_axi_bresp,
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
		output	wire		m_axi_arvalid,
		input	wire		m_axi_arready,
		// write read data channel signals
		input	wire	[5:0]	m_axi_rid,
		input	wire	[63:0]	m_axi_rdata,
		input	wire	[1:0]	m_axi_rresp,
		input	wire		m_axi_rlast,
		input	wire		m_axi_rvalid,
		output	wire		m_axi_rready,

		//----| np core #0 interface
		input	wire		npc_req,
		output	wire		npc_gnt,
		input	wire		npc_rwn,
		input	wire	[31:0]	npc_adr,
		input	wire	[31:0]	npc_len,
		input	wire	[63:0]	npc_wdt,
		output	wire	[63:0]	npc_rdt,
		output	wire		npc_ack
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| global clock & reset |--------------------------------------------------
wire		rstn		= m_axi_arstn;
wire		clk		= m_axi_aclk;

//----| npc arbitration |-------------------------------------------------------
reg		run;
wire		arben		= ~run;
wire		win		= arben & npc_req;

//----| npc run area |----------------------------------------------------------
wire		stt		= win;
wire		fin;
always @(negedge rstn or posedge clk) run <= !rstn ? 0 : win ? 1 : fin ? 0 : run;

//----| npc grant generatin |---------------------------------------------------
reg		gnt;
always @(negedge rstn or posedge clk) gnt <= !rstn ? 0 : win;

//----| npc burst end |---------------------------------------------------------
wire		bend;

//----| npc length offset calculation |-----------------------------------------
reg	[31:0]	adr, adr_nxt;
(* mark_debug = "true" *)reg	[31:0]	len, len_nxt;
wire	[31:0]	len_ofs	= len >= 256 ? 256 : len;

//----| npc r/w, anpcess, length |----------------------------------------------
reg		rwn;
always @(negedge rstn or posedge clk) rwn <= !rstn ? 0 : win ? npc_rwn : rwn;
always @(negedge rstn or posedge clk) adr_nxt <= !rstn ? 0 : adr + len_ofs * 8; // 16;
always @(negedge rstn or posedge clk) adr <= !rstn ? 0 : win ? npc_adr : bend ? adr_nxt : adr;
always @(negedge rstn or posedge clk) len_nxt <= !rstn ? 0 : len - len_ofs;
wire		upd_len;
always @(negedge rstn or posedge clk) len <= !rstn ? 0 : win ? npc_len : upd_len ? len_nxt : len;
wire		last_area	= len >= 1 && len <= 256;

//------------------------------------------------------------------------------
//----| npc operation |---------------------------------------------------------
(* mark_debug = "true" *)reg	[3:0]	sta;
(* mark_debug = "true" *)wire		sta_dat		= sta[1] & (~rwn & m_axi_awvalid & m_axi_awready | rwn & m_axi_arvalid & m_axi_arready);
(* mark_debug = "true" *)wire		sta_rsp		= sta[2] & ~rwn & m_axi_wvalid & m_axi_wready & m_axi_wlast;
(* mark_debug = "true" *)wire		sta_don		= sta[2] & rwn & m_axi_rvalid & m_axi_rready & m_axi_rlast | sta[3] & ~rwn & m_axi_bvalid & m_axi_bready;
(* mark_debug = "true" *)wire		sta_adr		= sta[0] && stt || sta_don && len_nxt != 0;
always @(negedge rstn or posedge clk) sta <= !rstn ? 1 : sta_adr ? 2 : sta_dat ? 4 : sta_rsp ? 8 : sta_don ? 1 : sta;
wire		adr_area	= sta[1];
wire		dat_area	= sta[2];
wire		rsp_area	= sta[3];

//---- data acknowledge
wire		dack		= dat_area & (rwn ? m_axi_rvalid & m_axi_rready : m_axi_wvalid & m_axi_wready);

//----| npc finish condition |--------------------------------------------------
reg	[1:0]	npc_fin_dly;
wire		npc_fin		= rwn ? dack & bend & last_area : sta_don & last_area;
always @(negedge rstn or posedge clk) npc_fin_dly <= !rstn ? 0 : {npc_fin_dly[0], npc_fin};	// room for write latency
assign		fin		= npc_fin_dly[1];

//----| npc burst count |-------------------------------------------------------
reg	[7:0]	bcnt;
always @(negedge rstn or posedge clk) bcnt <= !rstn ? 0 : sta_dat ? 0 : dack ? bcnt + 1 : bcnt;
assign		bend		= dack && bcnt == len_ofs - 1;
assign		upd_len		= rwn ? bend : sta_don;

//----| axi master signal generation |------------------------------------------
assign		m_axi_awid	= 0;
assign		m_axi_awaddr	= adr;		// in byte address
assign		m_axi_awlen	= len_ofs - 1;
assign		m_axi_awsize	= 3'b011;	// 8byte burst size
assign		m_axi_awburst	= 2'b01;	// incremental-address burst
assign		m_axi_awlock	= 0;
assign		m_axi_awcache	= 4'b0010;
assign		m_axi_awprot	= 3'b000;
assign		m_axi_awqos	= 4'b0000;
assign		m_axi_awregion	= 4'b0000;
assign		m_axi_awvalid	= ~rwn & adr_area;
assign		m_axi_wdata	= npc_wdt;
assign		m_axi_wstrb	= 8'b11111111;
assign		m_axi_wlast	= ~rwn & dat_area & bend; 
assign		m_axi_wvalid	= ~rwn & dat_area;
assign		m_axi_bready	= rsp_area;
assign		m_axi_arid	= 0;
assign		m_axi_araddr	= adr;		// in byte address
assign		m_axi_arlen	= len_ofs - 1;
assign		m_axi_arsize	= 3'b011;	// 8byte burst size
assign		m_axi_arburst	= 2'b01;	// incremental-address burst
assign		m_axi_arlock	= 0;
assign		m_axi_arcache	= 4'b0010;
assign		m_axi_arprot	= 3'b000;
assign		m_axi_arqos	= 4'b0000;
assign		m_axi_arregion	= 4'b0000;
assign		m_axi_arvalid	= rwn & adr_area;
assign		m_axi_rready	= rwn & dat_area;

//----| npc signal generation |-------------------------------------------------
assign		npc_gnt		= gnt;
assign		npc_rdt		= m_axi_rdata;
assign		npc_ack		= run & dack;

endmodule
//==============================================================================
