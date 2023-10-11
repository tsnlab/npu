
module npuCore
(
    input clock,
    input reset,

    input rocc_if_dram_offset,
    input rocc_if_size,
    input rocc_if_funct,
    input rocc_if_cmd_vld,
    output rocc_if_fin,
    output rocc_if_busy,

    output bf16_opc,
    output bf16_a,
    output bf16_b,
    input bf16_y,
    output bf16_iv,
    output bf16_or,
    input bf16_ov,
    input bf16_ir,

    output int32_opc,
    output int32_a,
    output int32_b,
    input int32_y,
    output int32_iv,
    output int32_or,
    input int32_ov,
    input int32_ir

)
    always@(posedge clock or posedge reset) begin
        if reset begin
            bf16_a <= 0;
            bf16_b <= 0;
            bf16_iv <= 0;
            rocc_if_fin <= 0;
            rocc_if_busy <= 0;
            bf16_or <= 0;
            int32_opc <= 0;
            int32_a <= 0;
            int32_b <= 0;
            int32_iv <= 0;
            int32_or <= 0;
        end else begin
            bf16_a <= rocc_if_size;
            bf16_b <= rocc_if_funct;
            bf16_iv <= rocc_if_cmd_vld;
            rocc_if_fin <= bf16_y;
            rocc_if_busy <= bf16_ov;
            bf16_or <= bf16_ir;
            int32_opc <= int32_y;
            int32_a <= int32_ov;
            int32_b <= int32_ir;
            int32_iv <= int32_ov;
            int32_or <= int32_ir;
        end
    end    
    // assign bf16_opc = rocc_if_dram_offset;
    // assign bf16_a = rocc_if_size;
    // assign bf16_b = rocc_if_funct;
    // assign bf16_iv = rocc_if_cmd_vld;
    // assign rocc_if_fin = bf16_y;
    // assign rocc_if_busy = bf16_ov;
    // assign bf16_or = bf16_ir;
    // assign int32_opc = int32_y;
    // assign int32_a = int32_ov;
    // assign int32_b = int32_ir;
    // assign int32_iv = int32_ov;
    // assign int32_or = int32_ir;
    
endmodule