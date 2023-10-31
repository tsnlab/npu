module SRAM (
		//----| write signals
		input   wire		clka,		// write clock
		input	wire		ena,		// write enable
		input	wire		wea,		// write werite enable
		input	wire	[11:0]	addra,		// write address
		input	wire	[127:0]	dina,		// write data in

		//----| read signals
		input	wire		clkb,		// read clock
		input	wire		enb,		// read enable
		input	wire	[11:0]	addrb,		// read address
		output	reg	[127:0]	doutb		// read data out
		);
//==============================================================================
//----| parameter & macro definition |------------------------------------------

//----| behaviral modelling |---------------------------------------------------
reg	[127:0]	ram[16*256-1:0];

//---- write operation
always @(posedge clka) if(ena && wea) ram[addra] <= dina;

//---- read operation
reg	[63:0]	outb;
always @(posedge clkb) if(enb) doutb <= ram[addrb];

endmodule