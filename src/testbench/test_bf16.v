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
  
  reg clock = 1'b0;
  reg reset = 1'b0;
  reg [2:0] io_opc;
  reg [15:0] io_a = 16'h41cc;
  wire [15:0] io_b = 16'h41ac;
  reg io_in_valid = 1'b1; 
  reg io_out_ready = 1'b1;
  reg io_isSqrt = 1'b0;
  reg io_kill = 1'b0;
  wire io_in_ready; 
  wire io_out_valid; 
  wire [15:0] io_y;
  reg [15:0] expected_result = 16'h0000;

  // Instantiate the BF16Unit module
  BF16Unit uut (
    .clock(clock),
    .reset(reset),
    .io_opc(io_opc),
    .io_a(io_a),
    .io_b(io_b),
    .io_in_valid(io_in_valid),
    .io_out_ready(io_out_ready),
    .io_y(io_y),
    .io_in_ready(io_in_ready),
    .io_out_valid(io_out_valid),
    .io_isSqrt(io_isSqrt),
    .io_kill(io_kill)
  );

  initial begin
    // Initialize the inputs
    reset = 1;
    #10;
    reset = 0;
    #10;
    io_opc = 3'b000; // Set the opcode (you can choose the appropriate opcode)
    #10;
    io_opc = 3'b001; // Set the opcode (you can choose the appropriate opcode)
    #10;
    io_opc = 3'b010; // Set the opcode (you can choose the appropriate opcode)
    #10;
    io_opc = 3'b011; // Set the opcode (you can choose the appropriate opcode)
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

always begin
  #5 clock = ~clock; // Create a 50MHz signal (divided from 100MHz)
end
  // Define the expected result based on the opcode and input values
  always @(*) begin
    case (io_opc)
      3'b000: expected_result = 16'h423c; // Example operation, adjust based on opcode
      3'b001: expected_result = 16'h4080; // Another example operation
      3'b010: expected_result = 16'h4409; // Another example operation
      3'b011: expected_result = 16'h4409; // Another example operation
      // Add more cases for other opcodes as needed
      default: expected_result = 16'h0000; // Default value for undefined opcode
    endcase
  end

endmodule