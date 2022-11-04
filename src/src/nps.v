/*==============================================================================
Copyright (C) 2022, LK1 Inc. All Rights Reserved

Project name	: NPU
Title		: npu slave module
File name	: nps.v
Module name	: nps
Date		: 2022. 9. 20 ~
Version		: Release 1
Author		: ckim@lk1.co.kr
Organization	: R&D Center
Comment		: Synthesizable Verilog-HDL code
Description	: This module is the slave of npu module.

Revison Note	: 
	1.0	2022. 9. 20	Cheol Kim
		Initial creation.

==============================================================================*/

//==============================================================================
module nps	(
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
		output	reg	[3:0]	irq,

		//----| npc interface signals
		output	reg	[31:0]	slv_ofs,
		output	reg	[31:0]	slv_siz,
		output	wire	[3:0]	slv_stt,
		input	wire	[3:0]	slv_fin,
		input	wire	[3:0]	slv_bsy
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| global signals |--------------------------------------------------------
wire		aclk		= s_axi_aclk;
wire		arstn		= s_axi_arstn;

//----| axi slave interface |---------------------------------------------------
//---- awrdy generation
reg	 	awrdy;
reg		wrun;
reg		rrun;
reg	 	wrdy;
wire		wset		= ~awrdy & s_axi_awvalid & ~wrun & ~rrun;
wire		wclr		= s_axi_wlast & wrdy;
always @ (posedge aclk or negedge arstn) awrdy <= !arstn ? 0 : wset ? 1 : ~wclr ? 0 : awrdy;
always @ (posedge aclk or negedge arstn) wrun <= !arstn ? 0 : wset ? 1 :  wclr ? 0 : wrun;

//---- wadr, wlen, wcnt
reg	[31:0]	wadr;
reg	[7:0]	wlen;
reg	[7:0]	wcnt;
reg	[1:0]	wbur;
wire	[31:0]	wwsz		= 4 * wlen; 
wire		wwrp		= (wadr & wwsz) == wwsz;
wire		walat		= ~awrdy & s_axi_awvalid & ~wrun;
wire		wdlat		= wcnt <= wlen && wrdy && s_axi_wvalid;
wire	[31:0]	wadr_incr	= (wadr[31:2] + 1) << 2;
wire	[31:0]	wadr_wrap	= wwrp ? wadr - wwsz : wadr_incr;
wire	[31:0]	wadr_next	= wbur == 0 ? wadr : wbur == 2 ? wadr_wrap : wadr_incr;
always @ (posedge aclk or negedge arstn) wadr <= !arstn ? 0 : walat ? s_axi_awaddr[31:0] : wdlat ? wadr_next : wadr;
always @ (posedge aclk or negedge arstn) wlen <= !arstn ? 0 : walat ? s_axi_awlen : wlen;
always @ (posedge aclk or negedge arstn) wbur <= !arstn ? 0 : walat ? s_axi_awburst : wbur;
always @ (posedge aclk or negedge arstn) wcnt <= !arstn ? 0 : walat ? 0 : wdlat ? wcnt + 1 : wcnt;
wire		wack		= s_axi_wready && s_axi_wvalid;

//-----
wire		wrdy_set	= ~wrdy & s_axi_wvalid & wrun;
wire		wrdy_clr	= s_axi_wlast & wrdy;
always @ (posedge aclk or negedge arstn) wrdy <= !arstn ? 0 : wrdy_set ? 1 : wrdy_clr ? 0 : wrdy;

//-----
reg [1:0]	brsp;
reg		bvld;
wire		bvalid_set	= wrun & wrdy & s_axi_wvalid & ~bvld & s_axi_wlast;
wire		bvalid_clr	= s_axi_bready & bvld;
always @ (posedge aclk or negedge arstn) bvld <= !arstn ? 0 : bvalid_set ? 1 : bvalid_clr ? 0 : bvld;
always @ (posedge aclk or negedge arstn) brsp <= !arstn ? 0 : bvalid_set ? 0 : brsp;

//-----
reg	[7:0]	rlen;
reg	[7:0]	rcnt;
reg		arrdy;
wire		rstt		= ~arrdy && s_axi_arvalid && ~rrun;
wire		rfin;
always @ (posedge aclk or negedge arstn) arrdy <= !arstn ? 0 : rstt ? 1 : ~rfin ? 0 : arrdy;
always @ (posedge aclk or negedge arstn) rrun <= !arstn ? 0 : rstt ? 1 :  rfin ? 0 : rrun;

//-----
reg	[31:0]	radr;
reg	[1:0]	rbur;
wire	[31:0]	rwsz		= 4 * rlen; 
wire		rwrp		= (radr & rwsz) == rwsz;
wire	[31:0] 	radr_incr	= (radr[31:2] + 1) << 2;
wire	[31:0] 	radr_wrap	= rwrp ? radr - rwsz : radr_incr;
wire	[31:0] 	radr_next	= rbur == 0 ? radr : rbur == 2 ? radr_wrap : radr_incr;

wire		rack;
always @ (posedge aclk or negedge arstn) radr <= !arstn ? 0 : rstt ? s_axi_araddr : rack ? radr_next : radr;
always @ (posedge aclk or negedge arstn) rlen <= !arstn ? 0 : rstt ? s_axi_arlen : rlen;
always @ (posedge aclk or negedge arstn) rbur <= !arstn ? 0 : rstt ? s_axi_arburst : rbur;
always @ (posedge aclk or negedge arstn) rcnt <= !arstn ? 0 : rstt ? 0 : rack ? rcnt + 1 : rcnt;
wire		rlst		= rcnt == rlen;

//----- read valid & response generation
reg	[1:0]	rvs;
assign		rack		= rvs == 3 && s_axi_rready;
always @ (posedge aclk or negedge arstn) rvs <= !arstn ? 0 : rrun && rvs != 3 ? rvs + 1 : rack ? 0 : rvs;
reg	[1:0]	rrsp;
always @ (posedge aclk or negedge arstn) rrsp <= !arstn ? 0 : rack ? 0 : rrsp;
assign		rfin		= rack & rlst;

//----| host operation |--------------------------------------------------------
reg	[15:0]	slv_cid;
//---- write
always @ (posedge aclk or negedge arstn) slv_ofs <= !arstn ? 0 : wack && wadr[7:2] == 0 ? s_axi_wdata : slv_ofs;
always @ (posedge aclk or negedge arstn) slv_siz <= !arstn ? 0 : wack && wadr[7:2] == 1 ? s_axi_wdata : slv_siz;
always @ (posedge aclk or negedge arstn) slv_cid <= !arstn ? 0 : wack && wadr[7:2] == 2 ? s_axi_wdata : slv_cid;

//---- read
wire	[31:0]	slv_sts		= slv_bsy;
reg	[31:0]	slv_cyc[3:0];
reg	[31:0]	rdat;
always @ (posedge aclk or negedge arstn) rdat <= !arstn ? 0 : radr[7:2] == 0 ? slv_ofs :
							      radr[7:2] == 1 ? slv_siz :
							      radr[7:2] == 2 ? slv_cid :
							      radr[7:2] == 3 ? slv_sts :
							      radr[7:2] == 4 ? slv_cyc[0] :
							      radr[7:2] == 5 ? slv_cyc[1] :
							      radr[7:2] == 6 ? slv_cyc[2] :
							      radr[7:2] == 7 ? slv_cyc[3] :
							      0;

//----| slave interface signals |-----------------------------------------------
genvar	gi;
for(gi = 0;gi < 4;gi = gi + 1)
begin: fpu_con
assign		slv_stt[gi]	= wack && wadr[7:2] == 2 && s_axi_wdata == gi;
always @ (posedge aclk or negedge arstn) irq[gi] <= !arstn ? 0 : slv_fin[gi];
always @ (posedge aclk or negedge arstn) slv_cyc[gi] <= !arstn ? 0 : slv_stt[gi] ? 0 : slv_bsy[gi] ? slv_cyc[gi] + 1 : slv_cyc[gi];
end	// gi

//----| axi signal mapping |----------------------------------------------------
assign		s_axi_awready	= awrdy;
assign		s_axi_wready	= wrdy;
assign		s_axi_bresp	= brsp;
assign		s_axi_bvalid	= bvld;
assign	#1	s_axi_arready	= arrdy;
assign		s_axi_rdata	= rdat;
assign		s_axi_rresp	= rrsp;
assign		s_axi_rlast	= rlst;
assign		s_axi_rvalid	= rack;
assign		s_axi_bid	= s_axi_awid;
assign		s_axi_rid	= s_axi_arid;

endmodule
//==============================================================================
