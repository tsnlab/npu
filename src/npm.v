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
		output	wire	[31:0]	m_axi_wdata,
		output	wire	[3:0]	m_axi_wstrb,
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
		input	wire	[31:0]	m_axi_rdata,
		input	wire	[1:0]	m_axi_rresp,
		input	wire		m_axi_rlast,
		input	wire		m_axi_rvalid,
		output	wire		m_axi_rready,

		//----| np core #0 interface
		input	wire		npc0_req,
		output	wire		npc0_gnt,
		input	wire		npc0_rwn,
		input	wire	[31:0]	npc0_adr,
		input	wire	[31:0]	npc0_len,
		input	wire	[31:0]	npc0_wdt,
		output	wire	[31:0]	npc0_rdt,
		output	wire		npc0_ack,

		//----| np core #1 interface
		input	wire		npc1_req,
		output	wire		npc1_gnt,
		input	wire		npc1_rwn,
		input	wire	[31:0]	npc1_adr,
		input	wire	[31:0]	npc1_len,
		input	wire	[31:0]	npc1_wdt,
		output	wire	[31:0]	npc1_rdt,
		output	wire		npc1_ack,

		//----| np core #2 interface
		input	wire		npc2_req,
		output	wire		npc2_gnt,
		input	wire		npc2_rwn,
		input	wire	[31:0]	npc2_adr,
		input	wire	[31:0]	npc2_len,
		input	wire	[31:0]	npc2_wdt,
		output	wire	[31:0]	npc2_rdt,
		output	wire		npc2_ack,

		//----| np core #3 interface
		input	wire		npc3_req,
		output	wire		npc3_gnt,
		input	wire		npc3_rwn,
		input	wire	[31:0]	npc3_adr,
		input	wire	[31:0]	npc3_len,
		input	wire	[31:0]	npc3_wdt,
		output	wire	[31:0]	npc3_rdt,
		output	wire		npc3_ack
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| global clock & reset |--------------------------------------------------
wire		rstn		= m_axi_arstn;
wire		clk		= m_axi_aclk;

//----| npc arbitration |-------------------------------------------------------
reg		run0, run1, run2, run3;
wire		arben		= ~run0 & ~run1 & ~run2 & ~run3;
wire		win0		= arben & npc0_req;
wire		win1		= arben & npc1_req & ~npc0_req;
wire		win2		= arben & npc2_req & ~npc0_req & ~npc1_req;
wire		win3		= arben & npc3_req & ~npc0_req & ~npc1_req & ~npc2_req;
wire		win		= win0 | win1 | win2 | win3;

//----| npc run area |----------------------------------------------------------
wire		stt		= win;
wire		fin;
always @(negedge rstn or posedge clk) run0 <= !rstn ? 0 : win0 ? 1 : fin ? 0 : run0;
always @(negedge rstn or posedge clk) run1 <= !rstn ? 0 : win1 ? 1 : fin ? 0 : run1;
always @(negedge rstn or posedge clk) run2 <= !rstn ? 0 : win2 ? 1 : fin ? 0 : run2;
always @(negedge rstn or posedge clk) run3 <= !rstn ? 0 : win3 ? 1 : fin ? 0 : run3;

//----| npc grant generatin |---------------------------------------------------
reg		gnt0, gnt1, gnt2, gnt3;
always @(negedge rstn or posedge clk) gnt0 <= !rstn ? 0 : win0;
always @(negedge rstn or posedge clk) gnt1 <= !rstn ? 0 : win1;
always @(negedge rstn or posedge clk) gnt2 <= !rstn ? 0 : win2;
always @(negedge rstn or posedge clk) gnt3 <= !rstn ? 0 : win3;

//----| npc burst end |---------------------------------------------------------
wire		bend;

//----| npc length offset calculation |-----------------------------------------
reg	[31:0]	npc_adr, npc_adr_nxt;
(* mark_debug = "true" *)reg	[31:0]	npc_len, npc_len_nxt;
wire	[31:0]	npc_len_ofs	= npc_len >= 256 ? 256 : npc_len;

//----| npc r/w, anpcess, length |----------------------------------------------
reg		npc_rwn;
always @(negedge rstn or posedge clk) npc_rwn <= !rstn ? 0 : win0 ? npc0_rwn : win1 ? npc1_rwn : win2 ? npc2_rwn : win3 ? npc3_rwn : npc_rwn;
always @(negedge rstn or posedge clk) npc_adr_nxt <= !rstn ? 0 : npc_adr + npc_len_ofs * 4; // 16;
always @(negedge rstn or posedge clk) npc_adr <= !rstn ? 0 : win0 ? npc0_adr : win1 ? npc1_adr : win2 ? npc2_adr : win3 ? npc3_adr : bend ? npc_adr_nxt : npc_adr;
always @(negedge rstn or posedge clk) npc_len_nxt <= !rstn ? 0 : npc_len - npc_len_ofs;
wire		upd_len;
always @(negedge rstn or posedge clk) npc_len <= !rstn ? 0 : win0 ? npc0_len : win1 ? npc1_len : win2 ? npc2_len : win3 ? npc3_len : upd_len ? npc_len_nxt : npc_len;
wire		last_area	= npc_len >= 1 && npc_len <= 256;

//------------------------------------------------------------------------------
//----| npc operation |---------------------------------------------------------
(* mark_debug = "true" *)reg	[3:0]	sta;
(* mark_debug = "true" *)wire		sta_dat		= sta[1] & (~npc_rwn & m_axi_awvalid & m_axi_awready | npc_rwn & m_axi_arvalid & m_axi_arready);
(* mark_debug = "true" *)wire		sta_rsp		= sta[2] & ~npc_rwn & m_axi_wvalid & m_axi_wready & m_axi_wlast;
(* mark_debug = "true" *)wire		sta_don		= sta[2] & npc_rwn & m_axi_rvalid & m_axi_rready & m_axi_rlast | sta[3] & ~npc_rwn & m_axi_bvalid & m_axi_bready;
(* mark_debug = "true" *)wire		sta_adr		= sta[0] && stt || sta_don && npc_len_nxt != 0;
always @(negedge rstn or posedge clk) sta <= !rstn ? 1 : sta_adr ? 2 : sta_dat ? 4 : sta_rsp ? 8 : sta_don ? 1 : sta;
wire		adr_area	= sta[1];
wire		dat_area	= sta[2];
wire		rsp_area	= sta[3];

//---- data acknowledge
wire		dack		= dat_area & (npc_rwn ? m_axi_rvalid & m_axi_rready : m_axi_wvalid & m_axi_wready);

//----| npc finish condition |--------------------------------------------------
reg	[1:0]	npc_fin_dly;
wire		npc_fin		= npc_rwn ? dack & bend & last_area : sta_don & last_area;
always @(negedge rstn or posedge clk) npc_fin_dly <= !rstn ? 0 : {npc_fin_dly[0], npc_fin};	// room for write latency
assign		fin		= npc_fin_dly[1];

//----| npc burst count |-------------------------------------------------------
reg	[7:0]	bcnt;
always @(negedge rstn or posedge clk) bcnt <= !rstn ? 0 : sta_dat ? 0 : dack ? bcnt + 1 : bcnt;
assign		bend		= dack && bcnt == npc_len_ofs - 1;
assign		upd_len		= npc_rwn ? bend : sta_don;

//----| axi master signal generation |------------------------------------------
assign		m_axi_awid	= 0;
assign		m_axi_awaddr	= npc_adr;	// in byte address
assign		m_axi_awlen	= npc_len_ofs - 1;
assign		m_axi_awsize	= 3'b010;	// 4byte burst size
assign		m_axi_awburst	= 2'b01;	// incremental-address burst
assign		m_axi_awlock	= 0;
assign		m_axi_awcache	= 4'b0010;
assign		m_axi_awprot	= 3'b000;
assign		m_axi_awqos	= 4'b0000;
assign		m_axi_awregion	= 4'b0000;
assign		m_axi_awvalid	= ~npc_rwn & adr_area;
assign		m_axi_wdata	= run0 ? npc0_wdt : run1 ? npc1_wdt : run2 ? npc2_wdt : npc3_wdt;
assign		m_axi_wstrb	= 4'b1111;
assign		m_axi_wlast	= ~npc_rwn & dat_area & bend; 
assign		m_axi_wvalid	= ~npc_rwn & dat_area;

assign		m_axi_bready	= rsp_area;

assign		m_axi_arid	= 0;
assign		m_axi_araddr	= npc_adr;	// in byte address
assign		m_axi_arlen	= npc_len_ofs - 1;
assign		m_axi_arsize	= 3'b010;	// 4byte burst size
assign		m_axi_arburst	= 2'b01;	// incremental-address burst
assign		m_axi_arlock	= 0;
assign		m_axi_arcache	= 4'b0010;
assign		m_axi_arprot	= 3'b000;
assign		m_axi_arqos	= 4'b0000;
assign		m_axi_arregion	= 4'b0000;
assign		m_axi_arvalid	= npc_rwn & adr_area;

assign		m_axi_rready	= npc_rwn & dat_area;

//----| npc0 signal generation |------------------------------------------------
assign		npc0_gnt	= gnt0;
assign		npc0_lst	= run0 & last_area;
assign		npc0_rdt	= m_axi_rdata;
assign		npc0_ack	= run0 & dack;

//----| npc1 signal generation |------------------------------------------------
assign		npc1_gnt	= gnt1;
assign		npc1_lst	= run1 & last_area;
assign		npc1_rdt	= m_axi_rdata;
assign		npc1_ack	= run1 & dack;

//----| npc2 signal generation |------------------------------------------------
assign		npc2_gnt	= gnt2;
assign		npc2_lst	= run2 & last_area;
assign		npc2_rdt	= m_axi_rdata;
assign		npc2_ack	= run2 & dack;

//----| npc3 signal generation |------------------------------------------------
assign		npc3_gnt	= gnt3;
assign		npc3_lst	= run3 & last_area;
assign		npc3_rdt	= m_axi_rdata;
assign		npc3_ack	= run3 & dack;

endmodule
//==============================================================================
