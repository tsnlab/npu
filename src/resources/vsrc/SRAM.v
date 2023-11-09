module SRAM (
		//----| write signals
		// input   wire        rstn,
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

reg	[127:0]	ram[16*256-1:0];

//---- write operation
integer i;
always @(posedge clka) begin
    if(ena && wea) ram[addra] <= dina;
end

always @(posedge clkb) begin
    if(enb) doutb <= ram[addrb];
end

endmodule
