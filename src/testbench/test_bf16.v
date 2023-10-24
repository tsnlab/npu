`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/10/24 20:06:42
// Design Name: 
// Module Name: test_bf16
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/10/24 17:53:26
// Design Name: 
// Module Name: test_bf16
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module test_bf16;

  reg [2:0] io_opc;
  reg [15:0] io_a = 16'h41cc;
  wire [15:0] io_b = 16'h41ac;
  wire [15:0] io_y;
  reg [15:0] expected_result = 16'h0000;

  // Instantiate the BF16Unit module
  BF16Unit uut (
    .io_opc(io_opc),
    .io_a(io_a),
    .io_b(io_b),
    .io_y(io_y)
  );

  initial begin
    // Initialize the inputs
    io_opc = 3'b000; // Set the opcode (you can choose the appropriate opcode)
    #10;
    io_opc = 3'b001; // Set the opcode (you can choose the appropriate opcode)
    #10;
    io_opc = 3'b010; // Set the opcode (you can choose the appropriate opcode)
    #10;


    // Check the output 'io_y'
    if (io_y == expected_result) begin
      $display("Test PASSED: io_y is correct.");
    end else begin
      $display("Test FAILED: io_y is incorrect. Expected: %h, Actual: %h", expected_result, io_y);
    end

    // Finish the simulation
    $finish;
  end

  // Define the expected result based on the opcode and input values
  always @(*) begin
    case (io_opc)
      3'b000: expected_result = 16'h423c; // Example operation, adjust based on opcode
      3'b001: expected_result = 16'h4080; // Another example operation
      3'b010: expected_result = 16'h4409; // Another example operation
      // Add more cases for other opcodes as needed
      default: expected_result = 16'h0000; // Default value for undefined opcode
    endcase
  end

endmodule
