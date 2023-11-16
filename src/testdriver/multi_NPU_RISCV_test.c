#include "rocc.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

//#define DATA_SIZE 2048            // Data Size
#define DATA_SIZE 16            // Data Size
#define TEST_OP_TYPE "vadd.bf16"  // Op Type for Test, Use "vadd.bf16", "vsub.bf16", "vmul.bf16", "vdiv.bf16"
#define NUMBER_OF_CORES 4         // Number f Cores used at the same time

#define SYS_CLK 12500000

#define KERNEL_WITH_LOAD_STORE 0
#define NPU_REG_ID_OFFSET 3

#define LOAD_STORE_TEST 1
#define __DEBUG_MODE__

#ifdef __DEBUG_MODE__
#define trace_pc_position()  printf("%s - %d \n", __func__, __LINE__);
#else
#define trace_pc_position()
#endif

typedef union {
    float f;
    uint32_t i;
} FloatUnion;

typedef struct {
    uint16_t mantissa : 7;
    uint16_t exponent : 8;
    uint16_t sign : 1;
} BF16;

static inline void npu_regSet(int idx, unsigned long data)
{
    ROCC_INSTRUCTION_SS(3, data, idx, 0);
}

static inline unsigned long npu_regGet(int idx)
{
    unsigned long value;
    ROCC_INSTRUCTION_DSS(3, value, 0, idx, 1);
    return value;
}

static inline void npu_exec()
{
    ROCC_INSTRUCTION(3, 2);
}

static inline void npu_load()
{
    asm volatile ("fence");
    ROCC_INSTRUCTION(3, 3);
}

static inline void npu_store()
{
    asm volatile ("fence");
    ROCC_INSTRUCTION(3, 4);
}

uint64_t get_time() {
    uint64_t tmp;

    asm volatile("csrr %0,time":"=r"(tmp));
    return tmp;
}

BF16 float_to_bf16(float value) {

    FloatUnion fu;
    fu.f = value;
    BF16 bf16;

    // Extract the sign bit.
    bf16.sign = (fu.i >> 31) & 0x1;

    // Extract the biased exponent (8 bits).
    int biased_exponent = (fu.i >> 23) & 0xFF;

    if (biased_exponent == 0) {
        bf16.exponent = 0;
        bf16.mantissa = (fu.i >> 16) & 0x7F;
    } else {
        bf16.exponent = biased_exponent - 127;

        bf16.mantissa = (fu.i >> 16) & 0x7F;
    }

    return bf16;
}

float bf16_to_float(BF16 bf16) {

    FloatUnion fu;
    int biased_exponent;

    fu.i = (bf16.sign << 31);

    if (bf16.exponent == 0) {
        fu.i |= ((bf16.mantissa & 0x7F) << 16);
    } else {
        biased_exponent = bf16.exponent + 127;
        fu.i |= (biased_exponent << 23);

        fu.i |= ((bf16.mantissa & 0x7F) << 16);
    }

    return fu.f;
}

BF16 bf16_add(BF16 a, BF16 b) {

    float result = bf16_to_float(a) + bf16_to_float(b);
    return float_to_bf16(result);
}

BF16 bf16_subtract(BF16 a, BF16 b) {

    float result = bf16_to_float(a) - bf16_to_float(b);
    return float_to_bf16(result);
}

BF16 bf16_multiply(BF16 a, BF16 b) {

    float result = bf16_to_float(a) * bf16_to_float(b);
    return float_to_bf16(result);
}

BF16 bf16_divide(BF16 a, BF16 b) {

    float result = bf16_to_float(a) / bf16_to_float(b);
    return float_to_bf16(result);
}

static void resize_converted_data_size_kernel(uint8_t* kernel, int size) {

    // Seperate Size Bytes Low and High
    uint8_t byte_low;
    uint8_t byte_high;

    // Devide Size
    byte_low = size & 0xff;
    byte_high = size >> 8;

#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[8] = byte_low;
    kernel[9] = byte_high;
    kernel[24] = byte_low;
    kernel[25] = byte_high;
    kernel[60] = byte_low;
    kernel[61] = byte_high;
#endif
}

static void resize_op_iteration_kernel(uint8_t* kernel, int size) {

    // Seperate Size Bytes Low and High
    uint8_t byte_low;
    uint8_t byte_high;

    // Devide Size
    byte_low = size & 0xff;
    byte_high = size >> 8;

#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[44] = byte_low;
    kernel[45] = byte_high;
#else
    kernel[12] = byte_low;
    kernel[13] = byte_high;
#endif
}

static void kernel_op_change(uint8_t* kernel, char* op) {

    uint8_t opcode;
    if (op == "vadd.bf16") {
        opcode = 0x09;
    } else if (op == "vsub.bf16") {
        opcode = 0x0a;
    } else if (op == "vmul.bf16"){
        opcode = 0x0b;
    } else if (op == "vdiv.bf16") {
        opcode = 0x0c;
    } else {
        printf("Wrong OP!!!\n");
    }
#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[51] = opcode;
#else
    kernel[19] = opcode;
#endif
}

static void riscv_calculate(BF16* output, BF16* input_A, BF16* input_B, char* op, int size) {

    for (int i = 0; i < size; i++) {
        if (op == "vadd.bf16") {
            output[i] = bf16_add(input_A[i], input_B[i]);
        } else if (op == "vsub.bf16") {
            output[i] = bf16_subtract(input_A[i], input_B[i]);
        } else if (op == "vmul.bf16"){
            output[i] = bf16_multiply(input_A[i], input_B[i]);
        } else if (op == "vdiv.bf16") {
            output[i] = bf16_divide(input_A[i], input_B[i]);
        }
    }
}

static void floatToString(float floatValue, char* strValue, int maxLength) {

    int intPart = (int)floatValue;
    int decimalPart = (int)((floatValue - intPart) * 1000); // Assuming 3 decimal places

    if (maxLength < 7) {
        // Buffer is too small to store anything meaningful
        return;
    }

    snprintf(strValue, maxLength, "%d.%03d", intPart, decimalPart);
}

static int compare_riscv_and_npu(int npu, char *op, BF16* out_risc_bf16, BF16* out_npu_bf16, int count) {

    // Check How Many Are Correct
    int check = 0;
    int error_cnt = 0;
    float out_risc_flt;
    float out_npu_flt;
    char riscvStrValue[50]; 
    char npuStrValue[50]; 
    char diffStrValue[50]; 
    for (int i = 0; i < count; i++) {
        out_risc_flt = bf16_to_float(out_risc_bf16[i]);
        out_npu_flt = bf16_to_float(out_npu_bf16[i]);
        if (out_risc_flt != out_npu_flt) {
            error_cnt += 1;
#ifdef __DEBUG_MODE__
            memset(riscvStrValue, 0, 50);
            memset(npuStrValue, 0, 50);
            memset(diffStrValue, 0, 50);
            floatToString(out_risc_flt, riscvStrValue, sizeof(riscvStrValue));
            floatToString(out_npu_flt, npuStrValue, sizeof(npuStrValue));
            floatToString(out_risc_flt - out_npu_flt, diffStrValue, sizeof(diffStrValue));
            printf("[Test Case %d] %s - FAIL\nRISCV data[%d]: %s\nNPU%d data[%d]: %s\nRISCV data[%d] - NPU data[%d] = %s\n",
                count, op, i, riscvStrValue, npu, i, npuStrValue, i, i, diffStrValue);
#endif
        } else {
            check += 1;
        }
    }

    printf("\n[Test Case %s, NPU%d] FAIL(Equal: %d, Not equal: %d)\n", op, npu, check, error_cnt);

    return check;
}

/* riscv issues store command to npu */
static void store_command_to_npu(int npu, long unsigned int l_addr, long unsigned int r_addr, int size) {

    trace_pc_position()
    npu_regSet((npu * NPU_REG_ID_OFFSET + 1), (long unsigned int)r_addr);
    trace_pc_position()
    npu_regSet((npu * NPU_REG_ID_OFFSET + 2), size);
    trace_pc_position()
    npu_regSet((npu * NPU_REG_ID_OFFSET + 3), (long unsigned int)l_addr);
    trace_pc_position()
    npu_load();
    trace_pc_position()

}

/* riscv issues store command to npu */
static void load_command_to_npu(int npu, long unsigned int r_addr, long unsigned int l_addr, int size) {

    npu_regSet((npu * NPU_REG_ID_OFFSET + 1), (long unsigned int)r_addr);
    npu_regSet((npu * NPU_REG_ID_OFFSET + 2), size);
    npu_regSet((npu * NPU_REG_ID_OFFSET + 3), (long unsigned int)l_addr);
    npu_store();
}

int main() {
    // 4096 Size Input data
    BF16 input_A[DATA_SIZE];
    BF16 input_B[DATA_SIZE];
    
    float f_val_A;
    float f_val_B;

    BF16 output_riscv_add[DATA_SIZE];
    BF16 output_riscv_sub[DATA_SIZE];
    BF16 output_riscv_mul[DATA_SIZE];
    BF16 output_riscv_div[DATA_SIZE];

    BF16 output_npu_0[DATA_SIZE]; // NPU 0 Output
    BF16 output_npu_1[DATA_SIZE]; // NPU 1 Output
    BF16 output_npu_2[DATA_SIZE]; // NPU 2 Output
    BF16 output_npu_3[DATA_SIZE]; // NPU 3 Output

    printf("\n========Init========\n\n");
    printf("Multi NAU Test\n");
    printf("[%s Test] Using %d Cores\n", TEST_OP_TYPE, NUMBER_OF_CORES);
    if(KERNEL_WITH_LOAD_STORE == 0) {
        printf("    Kernel without load/store functions\n\n");
    } else {
        printf("    Kernel with load/store functions\n\n");
    }

    memset(output_riscv_add, 0, sizeof(output_riscv_add));
    memset(output_riscv_sub, 0, sizeof(output_riscv_sub));
    memset(output_riscv_mul, 0, sizeof(output_riscv_mul));
    memset(output_riscv_div, 0, sizeof(output_riscv_div));
    
    trace_pc_position()
    memset(output_npu_0, 0, sizeof(output_npu_0));
    memset(output_npu_1, 0, sizeof(output_npu_1));
    memset(output_npu_2, 0, sizeof(output_npu_2));
    memset(output_npu_3, 0, sizeof(output_npu_3));

    trace_pc_position()
    // Random Data Input
    for (int temp_count = 0; temp_count < DATA_SIZE; temp_count++) {
        f_val_A = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        f_val_B = (float)rand() / RAND_MAX * 2000.0 - 1000.0;

        input_A[temp_count] = float_to_bf16(f_val_A);
        input_B[temp_count] = float_to_bf16(f_val_B);
    }

    printf("\ninput_A & input_B are filled with random data.\n");

    trace_pc_position()
    riscv_calculate(output_riscv_add, input_A, input_B, "vadd.bf16", DATA_SIZE);
    riscv_calculate(output_riscv_sub, input_A, input_B, "vsub.bf16", DATA_SIZE);
    riscv_calculate(output_riscv_mul, input_A, input_B, "vmul.bf16", DATA_SIZE);
    riscv_calculate(output_riscv_div, input_A, input_B, "vdiv.bf16", DATA_SIZE);

    printf("\nThe result values of risc-v for each function were calculated using input_A & input_B.\n");

    trace_pc_position()
#if 1 // *** FAILED *** (tohost = 7)
    // Memory Input
    memcpy((BF16*)0x200000, input_A, sizeof(BF16) * DATA_SIZE);
    memcpy((BF16*)0x201000, input_B, sizeof(BF16) * DATA_SIZE);
#endif

    trace_pc_position()
    // Kernel: need to align by 8bytes                              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59    60    61    62    63    64    65    66    67    68    69    70    71    72    73    74    75    76    77    78    79    80    81    82    83    84    85    86    87    88    89    90    91    92    93    94    95
    // Kernel: need to align by 8bytes                              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19  
    //                                                             20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39  
    //                                                             40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59  
    //                                                             60    61    62    63    64    65    66    67    68    69    70    71    72    73    74    75    76    77    78    79  
    //                                                             80    81    82    83    84    85    86    87    88    89    90    91    92    93    94    95    96    97    98    99
#if KERNEL_WITH_LOAD_STORE
    // With load/store, vadd.bf16
    __attribute__ ((aligned (8))) volatile uint8_t kernel_0[] = {0x20, 0x00, 0x10, 0x02, 0x00, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x04, 0x10, 0x02,
                                                                 0x20, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02,
                                                                 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09, 0x40, 0x40, 0x10, 0x02, 0x20, 0x08, 0x20, 0x02,
                                                                 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x08, 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_1[] = {0x20, 0x00, 0x10, 0x02, 0x00, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x04, 0x10, 0x02,
                                                                 0x20, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02,
                                                                 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09, 0x60, 0x40, 0x10, 0x02, 0x20, 0x08, 0x20, 0x02,
                                                                 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x08, 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_2[] = {0x20, 0x00, 0x10, 0x02, 0x00, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x04, 0x10, 0x02,
                                                                 0x20, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02,
                                                                 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09, 0x80, 0x40, 0x10, 0x02, 0x20, 0x08, 0x20, 0x02,
                                                                 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x08, 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_3[] = {0x20, 0x00, 0x10, 0x02, 0x00, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x04, 0x10, 0x02,
                                                                 0x20, 0x40, 0x20, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x07, 0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02,
                                                                 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09, 0xa0, 0x40, 0x10, 0x02, 0x20, 0x08, 0x20, 0x02,
                                                                 0x00, 0x04, 0x30, 0x02, 0x00, 0x30, 0x12, 0x08, 0x00, 0x00, 0x00, 0xff};
#else
    // Without load/store, vadd.bf16
    __attribute__ ((aligned (8))) volatile uint8_t kernel_0[] = {0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09,
                                                                 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_1[] = {0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09,
                                                                 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_2[] = {0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09,
                                                                 0x00, 0x00, 0x00, 0xff};
    __attribute__ ((aligned (8))) volatile uint8_t kernel_3[] = {0x20, 0x00, 0x10, 0x02, 0x20, 0x04, 0x20, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x08, 0x40, 0x02, 0x00, 0x24, 0x31, 0x09,
                                                                 0x00, 0x00, 0x00, 0xff};
#endif

    trace_pc_position()
    resize_op_iteration_kernel(kernel_0, DATA_SIZE);
    resize_op_iteration_kernel(kernel_1, DATA_SIZE);
    resize_op_iteration_kernel(kernel_2, DATA_SIZE);
    resize_op_iteration_kernel(kernel_3, DATA_SIZE);
    
    trace_pc_position()
    // Change Kernel's Opcode
    kernel_op_change(kernel_0, TEST_OP_TYPE);
    kernel_op_change(kernel_1, TEST_OP_TYPE);
    kernel_op_change(kernel_2, TEST_OP_TYPE);
    kernel_op_change(kernel_3, TEST_OP_TYPE);

#if KERNEL_WITH_LOAD_STORE // with-load-store
    resize_converted_data_size_kernel(kernel_0, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_1, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_2, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_3, (int)((DATA_SIZE * 2 + 3)/4));
#endif

    printf("\nKernel images for each NPU have been prepared.\n\n");

    trace_pc_position()
    // Load kernel code at address 0 of npu
    store_command_to_npu(0, (long unsigned int)0x00, (long unsigned int)kernel_0, (int)sizeof(kernel_0));
    store_command_to_npu(1, (long unsigned int)0x00, (long unsigned int)kernel_1, (int)sizeof(kernel_1));
    store_command_to_npu(2, (long unsigned int)0x00, (long unsigned int)kernel_2, (int)sizeof(kernel_2));
    store_command_to_npu(3, (long unsigned int)0x00, (long unsigned int)kernel_3, (int)sizeof(kernel_3));

    printf("Kernel images are stored in each NPU.\n\n");

    trace_pc_position()
#if !KERNEL_WITH_LOAD_STORE // without-load-store
    // Load input_A at address 0x80 of npu
    store_command_to_npu(0, (long unsigned int)0x80, (long unsigned int)0x200000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(1, (long unsigned int)0x80, (long unsigned int)0x200000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(2, (long unsigned int)0x80, (long unsigned int)0x200000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(3, (long unsigned int)0x80, (long unsigned int)0x200000, (int)(sizeof(BF16) * DATA_SIZE));

    printf("input_A is stored in all NPUs.\n\n");

    trace_pc_position()
    // Load input_B at address 0x1080 of npu
    store_command_to_npu(0, (long unsigned int)0x1080, (long unsigned int)0x201000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(1, (long unsigned int)0x1080, (long unsigned int)0x201000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(2, (long unsigned int)0x1080, (long unsigned int)0x201000, (int)(sizeof(BF16) * DATA_SIZE));
    store_command_to_npu(3, (long unsigned int)0x1080, (long unsigned int)0x201000, (int)(sizeof(BF16) * DATA_SIZE));

    printf("input_B is stored in all NPUs.\n\n");
#endif

    trace_pc_position()
    // sys-clk time_start, time_end;
    //uint32_t time_start, time_end;
    uint64_t time_start, time_end;

    printf("\nRuns all NPUs.\n");
#if 0
    // NPU Set
    volatile uint32_t *npu_base = (uint32_t*)0x43C00000; // kernel offset
    //Core 0
    npu_base[0] = (uint32_t)kernel_0;// the address of kernel in main memory
    npu_base[1] = sizeof(kernel_0); // the size of kernel, also need to align by 8bytes
    
    //XTime_GetTime(&time_start); // Start Measuring Time when core 0 is Start
    time_start = get_time();
    npu_base[2] = 0; // Core Id
    //Core 1
    npu_base[0] = (uint32_t)kernel_1;
    npu_base[1] = sizeof(kernel_1);
    npu_base[2] = 1;
    //Core 2
    npu_base[0] = (uint32_t)kernel_2;
    npu_base[1] = sizeof(kernel_2);
    npu_base[2] = 2;
    //Core 3
    npu_base[0] = (uint32_t)kernel_3;
    npu_base[1] = sizeof(kernel_3);
    npu_base[2] = 3;

    while(npu_base[3] & 0b1111) { // wait until operation is done (not busy)
        //Xil_DCacheInvalidateRange(&npu_base[3], (uint32_t)sizeof(npu_base[3]));
        //invalidate_data_cache(&npu_base[3], (uint32_t)sizeof(npu_base[3]));
    }; 
    // XTime_GetTime(&time_end); // Get End Time
#endif
    trace_pc_position()
    //time_start = get_time();

    trace_pc_position()
    //time_end = get_time();
    
    printf("\nAll NPUs have completed calculations.\n\n");

    trace_pc_position()

#if !KERNEL_WITH_LOAD_STORE // without-load-store
#if LOAD_STORE_TEST
    // Load input_A at address 0x202000 of riscv
    load_command_to_npu(0, (long unsigned int)0x202000, (long unsigned int)0x0080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(1, (long unsigned int)0x203000, (long unsigned int)0x0080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(2, (long unsigned int)0x204000, (long unsigned int)0x0080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(3, (long unsigned int)0x205000, (long unsigned int)0x0080, (int)(sizeof(BF16) * DATA_SIZE));
#else
    // Load output_C at address 0x202000 of riscv
    load_command_to_npu(0, (long unsigned int)0x202000, (long unsigned int)0x2080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(1, (long unsigned int)0x203000, (long unsigned int)0x2080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(2, (long unsigned int)0x204000, (long unsigned int)0x2080, (int)(sizeof(BF16) * DATA_SIZE));
    load_command_to_npu(3, (long unsigned int)0x205000, (long unsigned int)0x2080, (int)(sizeof(BF16) * DATA_SIZE));
#endif

    printf("The calculated result values were loaded into external memory.\n\n");
#endif

    trace_pc_position()
#if 1 // *** FAILED *** (tohost = 5)
    // Memory Output
    memcpy(output_npu_0, 0x202000, sizeof(BF16) * DATA_SIZE);
    memcpy(output_npu_1, 0x203000, sizeof(BF16) * DATA_SIZE);
    memcpy(output_npu_2, 0x204000, sizeof(BF16) * DATA_SIZE);
    memcpy(output_npu_3, 0x205000, sizeof(BF16) * DATA_SIZE);

    printf("Copy the result values loaded in external memory to local variables of risc-v.\n\n");
#endif

    trace_pc_position()

    printf("\nCompare the results calculated by risc-v and the results calculated by NPUs.\n");
    // Check RISCV's Outpus & NPUs's Outputs Are Same, Input RISCV's OP Output Array in 1st Parameter
    int check0 = 0;
    int check1 = 0;
    int check2 = 0;
    int check3 = 0;
    if (TEST_OP_TYPE == "vadd.bf16") {
        check0 = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_add, output_npu_0, DATA_SIZE);
        check1 = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_add, output_npu_1, DATA_SIZE);
        check2 = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_add, output_npu_2, DATA_SIZE);
        check3 = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_add, output_npu_3, DATA_SIZE);
    } else if (TEST_OP_TYPE == "vsub.bf16") {
        check0 = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_sub, output_npu_0, DATA_SIZE);
        check1 = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_sub, output_npu_1, DATA_SIZE);
        check2 = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_sub, output_npu_2, DATA_SIZE);
        check3 = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_sub, output_npu_3, DATA_SIZE);
    } else if (TEST_OP_TYPE == "vmul.bf16"){
        check0 = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_mul, output_npu_0, DATA_SIZE);
        check1 = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_mul, output_npu_1, DATA_SIZE);
        check2 = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_mul, output_npu_2, DATA_SIZE);
        check3 = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_mul, output_npu_3, DATA_SIZE);
    } else if (TEST_OP_TYPE == "vdiv.bf16") {
        check0 = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_div, output_npu_0, DATA_SIZE);
        check1 = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_div, output_npu_1, DATA_SIZE);
        check2 = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_div, output_npu_2, DATA_SIZE);
        check3 = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_div, output_npu_3, DATA_SIZE);
    }

    trace_pc_position()
    // If All Pass, Print
    if (check0 == DATA_SIZE) {
        printf("[NPU 0 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check1 == DATA_SIZE) {
        printf("[NPU 1 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check2 == DATA_SIZE) {
        printf("[NPU 2 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check3 == DATA_SIZE) {
        printf("[NPU 3 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }

    char elapsedTimeStrValue[50]; 
    floatToString(2.0 * (time_end - time_start) / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));

    printf("\nRISC-V Time: %s us.\n", elapsedTimeStrValue);

#if 0
    // Test Case N's Total Cycles, NPU: 125MHz
    printf("[#0 NPU] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 0], npu_base[4 + 0] * 8.00 / 1000.00);
    printf("[#1 NPU] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 1], npu_base[4 + 1] * 8.00 / 1000.00);
    printf("[#2 NPU] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 2], npu_base[4 + 2] * 8.00 / 1000.00);
    printf("[#3 NPU] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 3], npu_base[4 + 3] * 8.00 / 1000.00);
#endif

    printf("\n========Finish========\n\n");
    
    return 0;
}
