#include "rocc.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define MAX_DATA_SIZE 512        // Data Size
#define DATA_SIZE 512            // Data Size
#define MAX_NUMBER_OF_CORES 6     // Max. Number of Cores used at the same time
#define NUMBER_OF_CORES 6         // Number of Cores used at the same time

#define LOOP_COUNT 1024         // Number of loop 

#define SYS_CLK 26000000 // RISC-V: 26MHz 26,000,000

#define KERNEL_WITH_LOAD_STORE 0
#define NPU_REG_ID_OFFSET 3

#define INPUT_A_SRAM_BASE_ADDRESS    0x80 // 128            //  0x40 // 0x200
#define INPUT_B_SRAM_BASE_ADDRESS  (INPUT_A_SRAM_BASE_ADDRESS + MAX_DATA_SIZE * 2) // 0x480 128 + 1024     //  0xC0 // 0x1200
#define RESULT_SRAM_BASE_ADDRESS   (INPUT_A_SRAM_BASE_ADDRESS + MAX_DATA_SIZE * 2 * 2) // 0x880 128 + 1024 * 2 *2 // 0x140 // 0x2200

#define NPU_LOAD_STORE_MICRO_DELAY 10000

#define NPU_COMPLETE_EXEC_INTERRUPT 1
#define NPU_COMPLETE_EXEC_REG (NUMBER_OF_CORES * 3 + 1) // 19
#define NPU_COMPLETE_INTERRUPT_RST (NPU_COMPLETE_EXEC_REG + 1) // 20
#define NPU_COMPLETE_EXEC_TIMEOUT 10000000.00
#define EPSILON 0.01

#define DUMMY_DATA_SIZE 16

#if 1
#define DDR_M    1 // 128 // DDR_ADDR_MAGNIFICATION
#define SRAM_M   1 // 4 // SRAM_ADDR_MAGNIFICATION
#define SIZE_M  16 // 4 // SIZE_MAGNIFICATION
#else
#define DDR_M   128 // DDR_ADDR_MAGNIFICATION
#define SRAM_M    4 // SRAM_ADDR_MAGNIFICATION
#define SIZE_M    4 // SIZE_MAGNIFICATION
#endif

#define MAX_LOAD_STORE_CHUNK_SIZE 128 // must be >= 128

//#define _NPU_LOAD_STORE_TEST_MODE_
//#define __DEBUG_MODE__

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

// 4096 Size Input data
__attribute__ ((aligned (128))) volatile BF16 input_A[DATA_SIZE];
__attribute__ ((aligned (128))) volatile BF16 input_B[DATA_SIZE];

__attribute__ ((aligned (128))) volatile BF16 output_npu_0[DATA_SIZE]; // NPU 0 Output
__attribute__ ((aligned (128))) volatile BF16 output_npu_1[DATA_SIZE]; // NPU 1 Output
__attribute__ ((aligned (128))) volatile BF16 output_npu_2[DATA_SIZE]; // NPU 2 Output
__attribute__ ((aligned (128))) volatile BF16 output_npu_3[DATA_SIZE]; // NPU 3 Output
__attribute__ ((aligned (128))) volatile BF16 output_npu_4[DATA_SIZE]; // NPU 4 Output
__attribute__ ((aligned (128))) volatile BF16 output_npu_5[DATA_SIZE]; // NPU 5 Output

__attribute__ ((aligned (64))) volatile BF16 output_riscv_add[DATA_SIZE];
__attribute__ ((aligned (64))) volatile BF16 output_riscv_sub[DATA_SIZE];
__attribute__ ((aligned (64))) volatile BF16 output_riscv_mul[DATA_SIZE];
__attribute__ ((aligned (64))) volatile BF16 output_riscv_div[DATA_SIZE];

__attribute__ ((aligned (128))) volatile uint8_t dummy_output_0[DUMMY_DATA_SIZE];
__attribute__ ((aligned (128))) volatile uint8_t dummy_output_1[DUMMY_DATA_SIZE];
__attribute__ ((aligned (128))) volatile uint8_t dummy_output_2[DUMMY_DATA_SIZE];
__attribute__ ((aligned (128))) volatile uint8_t dummy_output_3[DUMMY_DATA_SIZE];
__attribute__ ((aligned (128))) volatile uint8_t dummy_output_4[DUMMY_DATA_SIZE];
__attribute__ ((aligned (128))) volatile uint8_t dummy_output_5[DUMMY_DATA_SIZE];

  // Kernel: need to align by 8bytes                              0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19
  //   0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19
  //  20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39
  //  40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59
  //  60    61    62    63    64    65    66    67    68    69    70    71    72    73    74    75    76    77    78    79
  //  80    81    82    83    84    85    86    87    88    89    90    91    92    93    94    95    96    97    98    99
#if KERNEL_WITH_LOAD_STORE
// With load/store, vadd.bf16
__attribute__ ((aligned (128))) volatile uint8_t kernel_0[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
__attribute__ ((aligned (128))) volatile uint8_t kernel_1[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
__attribute__ ((aligned (128))) volatile uint8_t kernel_2[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
__attribute__ ((aligned (128))) volatile uint8_t kernel_3[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
__attribute__ ((aligned (128))) volatile uint8_t kernel_4[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
__attribute__ ((aligned (128))) volatile uint8_t kernel_5[] = {
    0x00, 0x40, 0x20, 0x03, 0x20, 0x00, 0x10, 0x02, 0x00, 0x04, 0x30, 0x02, 0x00, 0x00, 0x20, 0x04, 0x20, 0x04, 0x10, 0x02,
    0x00, 0x30, 0x12, 0x07, 0x00, 0x00, 0x20, 0x04, 0x20, 0x40, 0x20, 0x03, 0x00, 0x30, 0x12, 0x07, 0x00, 0x04, 0x30, 0x02,
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x40, 0x40, 0x10, 0x03,
    0x00, 0x24, 0x31, 0x09, 0x20, 0x08, 0x20, 0x02, 0x00, 0x00, 0x10, 0x04, 0x00, 0x30, 0x12, 0x08, 0x00, 0x04, 0x30, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff};
#else
    // Without load/store, vadd.bf16
#if 1 // ifneq %e %f -12
__attribute__ ((aligned (128))) volatile uint8_t kernel_0[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
__attribute__ ((aligned (128))) volatile uint8_t kernel_1[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
__attribute__ ((aligned (128))) volatile uint8_t kernel_2[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
__attribute__ ((aligned (128))) volatile uint8_t kernel_3[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
__attribute__ ((aligned (128))) volatile uint8_t kernel_4[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
__attribute__ ((aligned (128))) volatile uint8_t kernel_5[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x60, 0x02,
    0x00, 0x04, 0x50, 0x02, 0x01, 0x00, 0x60, 0x0d, 0x00, 0x24, 0x31, 0x09, 0x00, 0x00, 0x00, 0xff, 0xf8, 0xff, 0x56, 0x11};
#else
__attribute__ ((aligned (128))) volatile uint8_t kernel_0[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x24, 0x31, 0x09};
__attribute__ ((aligned (128))) volatile uint8_t kernel_1[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x24, 0x31, 0x09};
__attribute__ ((aligned (128))) volatile uint8_t kernel_2[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x24, 0x31, 0x09};
__attribute__ ((aligned (128))) volatile uint8_t kernel_3[] = {
    0x20, 0x04, 0x20, 0x02, 0x20, 0x00, 0x10, 0x02, 0x00, 0x08, 0x40, 0x02, 0x20, 0x08, 0x30, 0x02, 0x00, 0x00, 0x00, 0xff,
    0x00, 0x24, 0x31, 0x09};
#endif
#endif

uint64_t elapsedCsrrsCycle = 0;
unsigned long g_interrupt_mask = 0xF;
char* TEST_OP_TYPE = "vadd.bf16";  // Op Type for Test, Use "vadd.bf16", "vsub.bf16", "vmul.bf16", "vdiv.bf16"

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

/* riscv issues store command to npu */
static void load_command_to_npu(int npu, long unsigned int l_addr, long unsigned int r_addr, int size) {

    npu_regSet((npu * NPU_REG_ID_OFFSET + 1), (long unsigned int)((r_addr + DDR_M - 1) / DDR_M));
    npu_regSet((npu * NPU_REG_ID_OFFSET + 2), (int)((size + SIZE_M - 1) / SIZE_M));
    npu_regSet((npu * NPU_REG_ID_OFFSET + 3), (long unsigned int)((l_addr + SRAM_M - 1) / SRAM_M));
}

/* riscv issues store command to npu */
static void store_command_to_npu(int npu, long unsigned int r_addr, long unsigned int l_addr, int size) {

    npu_regSet((npu * NPU_REG_ID_OFFSET + 1), (long unsigned int)((r_addr + DDR_M - 1) / DDR_M));
    npu_regSet((npu * NPU_REG_ID_OFFSET + 2), (int)((size + SIZE_M - 1) / SIZE_M));
    npu_regSet((npu * NPU_REG_ID_OFFSET + 3), (long unsigned int)((l_addr + SRAM_M - 1) / SRAM_M));
}

static inline uint64_t get_time() {
    uint64_t tmp;

    asm volatile("csrrs %0, cycle, x0":"=r"(tmp));
    return tmp;
}

void delay_in_usec(int us) {

    uint64_t cycle_start;
    uint64_t cycle_end;
    int elapsedTime;

    cycle_start = get_time();
    cycle_end = get_time();
    elapsedTime = (cycle_end - cycle_start) / (SYS_CLK / 1000000); 
    while(elapsedTime < us) {
        cycle_end = get_time();
        elapsedTime = (cycle_end - cycle_start) / (SYS_CLK / 1000000); 
    }
}

BF16 swap_bf16_bytes(BF16 value) {
#if 1
    BF16 swap_value;
    char *origin;
    char *swap;

    origin = &value;
    swap = &swap_value;
    swap[0] = origin[1];
    swap[1] = origin[0];

    return swap_value;
#else
    return value;
#endif
}

BF16 float_to_bf16(float value) {

    uint32_t f32_value_as_uint32;
    uint16_t bf16_value;

    // Assuming little-endian architecture
    memcpy(&f32_value_as_uint32, &value, sizeof(float));

    // Extract the 16 most significant bits
    bf16_value = (uint16_t)(f32_value_as_uint32 >> 16);

    BF16 bf16_result;
    memcpy(&bf16_result, &bf16_value, sizeof(BF16));

    return swap_bf16_bytes(bf16_result);
#if 0
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
#endif
}

float bf16_to_float(BF16 bf16) {

#if 1
    BF16 swap_bf16;
    uint32_t f32_value_as_uint32;
    uint16_t zero_padding = 0;

    swap_bf16 = swap_bf16_bytes(bf16);

    // Assuming little-endian architecture
    memcpy(&f32_value_as_uint32, &zero_padding, sizeof(uint16_t));
    memcpy(((uint8_t*)&f32_value_as_uint32) + sizeof(uint16_t), &swap_bf16, sizeof(uint16_t));

    float f32_result;
    memcpy(&f32_result, &f32_value_as_uint32, sizeof(float));

    return f32_result;
#else
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
#endif
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
    kernel[36] = byte_low;
    kernel[37] = byte_high;
    kernel[76] = byte_low;
    kernel[77] = byte_high;
#endif
}

static void kernel_loop_count_change(uint8_t* kernel, int count) {

    // Seperate Size Bytes Low and High
    uint8_t byte_low;
    uint8_t byte_high;

    // Devide Size
    byte_low = count & 0xff;
    byte_high = count >> 8;

#if KERNEL_WITH_LOAD_STORE // with-load-store
#else
    kernel[20] = byte_low;
    kernel[21] = byte_high;
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
    kernel[48] = byte_low;
    kernel[49] = byte_high;
#else
    kernel[8] = byte_low;
    kernel[9] = byte_high;
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
    kernel[63] = opcode;
#else
    kernel[31] = opcode;
#endif
}

static void kernel_input_a_sram_addr_change(uint8_t* kernel, unsigned long addr) {

    uint8_t byte_low;
    uint8_t byte_high;

    addr = (addr + 3)/4;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
#else
    kernel[4] = byte_low;
    kernel[5] = byte_high;
#endif
}

static void kernel_input_b_sram_addr_change(uint8_t* kernel, unsigned long addr) {

    uint8_t byte_low;
    uint8_t byte_high;

    addr = (addr + 3)/4;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
#else
    kernel[0] = byte_low;
    kernel[1] = byte_high;
#endif
}

static void kernel_output_c_sram_addr_change(uint8_t* kernel, unsigned long addr) {

    uint8_t byte_low;
    uint8_t byte_high;

    addr = (addr + 3)/4;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
#else
    kernel[12] = byte_low;
    kernel[13] = byte_high;
#endif
}

static void kernel_input_a_addr_change(uint8_t* kernel, BF16* data) {

    unsigned long addr;
    uint8_t byte_low;
    uint8_t byte_high;

    addr = (unsigned long)data;
    addr = (addr + 127)/128;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[0] = byte_low;
    kernel[1] = byte_high;

    byte_low = (addr >> 16) & 0xff;
    byte_high = (addr >> 24) & 0xFF;

    kernel[12] = byte_low;
    kernel[13] = byte_high;
#endif
}

static void kernel_input_b_addr_change(uint8_t* kernel, BF16* data) {

    unsigned long addr;
    uint8_t byte_low;
    uint8_t byte_high;

    addr = (unsigned long)data;
    addr = (addr + 127)/128;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[28] = byte_low;
    kernel[29] = byte_high;

    byte_low = (addr >> 16) & 0xff;
    byte_high = (addr >> 24) & 0xFF;

    kernel[24] = byte_low;
    kernel[25] = byte_high;
#endif
}

static void kernel_input_c_addr_change(uint8_t* kernel, BF16* data) {

    unsigned long addr;
    uint8_t byte_low;
    uint8_t byte_high;

    addr = (unsigned long)data;
    addr = (addr + 127)/128;

    // Devide Size
    byte_low = addr & 0xff;
    byte_high = (addr >> 8) & 0xFF;

#if KERNEL_WITH_LOAD_STORE // with-load-store
    kernel[56] = byte_low;
    kernel[57] = byte_high;

    byte_low = (addr >> 16) & 0xff;
    byte_high = (addr >> 24) & 0xFF;

    kernel[68] = byte_low;
    kernel[69] = byte_high;
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
    float diff;
    char riscvStrValue[50]; 
    char npuStrValue[50]; 
    char diffStrValue[50]; 

    for (int i = 0; i < count; i++) {
        out_risc_flt = bf16_to_float(out_risc_bf16[i]);
        out_npu_flt = bf16_to_float(out_npu_bf16[i]);
        if(out_risc_flt > out_npu_flt) {
            diff = out_risc_flt - out_npu_flt;
        } else {
            diff = out_npu_flt - out_risc_flt;
        }
        if (diff > EPSILON) {
            error_cnt += 1;
#if 1 // def __DEBUG_MODE__
            memset(riscvStrValue, 0, 50);
            memset(npuStrValue, 0, 50);
            memset(diffStrValue, 0, 50);
            floatToString(out_risc_flt, riscvStrValue, sizeof(riscvStrValue));
            floatToString(out_npu_flt, npuStrValue, sizeof(npuStrValue));
            floatToString(out_risc_flt - out_npu_flt, diffStrValue, sizeof(diffStrValue));
#if 0
            printf("[Test Case %d] %s - FAIL\nRISCV data[%d]: %s\nNPU%d data[%d]: %s\nRISCV data[%d] - NPU data[%d] = %s\n",
                count, op, i, riscvStrValue, npu, i, npuStrValue, i, i, diffStrValue);
#endif
            char * r_data;
            char * n_data;
            char * a_data;
            char * b_data;
            a_data = (char *)&input_A[i];
            b_data = (char *)&input_B[i];
            r_data = (char *)&out_risc_bf16[i];
            n_data = (char *)&out_npu_bf16[i];
            printf("%s[%d/%d] - FAIL, A 0x%02x%02x B 0x%02x%02x = RISC-V 0x%02x%02x - NPU 0x%02x%02x\n",
                op, i, count, a_data[1], a_data[0], b_data[1], b_data[0], r_data[1], r_data[0], n_data[1], n_data[0]);
#endif
        } else {
            check += 1;
#if 1
            char * r_data;
            char * n_data;
            char * a_data;
            char * b_data;
            a_data = (char *)&input_A[i];
            b_data = (char *)&input_B[i];
            r_data = (char *)&out_risc_bf16[i];
            n_data = (char *)&out_npu_bf16[i];
            printf("%s[%d/%d] - SUCCESS, A 0x%02x%02x B 0x%02x%02x = RISC-V 0x%02x%02x - NPU 0x%02x%02x\n",
                op, i, count, a_data[1], a_data[0], b_data[1], b_data[0], r_data[1], r_data[0], n_data[1], n_data[0]);
#endif
        }
    }

    printf("[Test Case %s, NPU%d] Result(matched: %d, unmatched: %d)\n", op, npu, check, error_cnt);

    return check;
}

void init_variavles() {

    float f_val_A;
    float f_val_B;

    memset(output_riscv_add, 0, sizeof(output_riscv_add));
    memset(output_riscv_sub, 0, sizeof(output_riscv_sub));
    memset(output_riscv_mul, 0, sizeof(output_riscv_mul));
    memset(output_riscv_div, 0, sizeof(output_riscv_div));
    
    memset(output_npu_0, 0, sizeof(output_npu_0));
    memset(output_npu_1, 0, sizeof(output_npu_1));
    memset(output_npu_2, 0, sizeof(output_npu_2));
    memset(output_npu_3, 0, sizeof(output_npu_3));
    memset(output_npu_4, 0, sizeof(output_npu_4));
    memset(output_npu_5, 0, sizeof(output_npu_5));

    // Random Data Input
    for (int temp_count = 0; temp_count < DATA_SIZE; temp_count++) {

#if 0
//host.data[0x200000] = jnp.array([(v + 1) * 1.1 for v in range(2048)], dtype=jnp.bfloat16).tobytes()
//host.data[0x201000] = jnp.array([(v + 1) * 0.1 for v in range(2048)], dtype=jnp.bfloat16).tobytes()
        uint16_t a_val;
        uint16_t b_val;

        a_val = (temp_count + 1) * 1.1;
        b_val = (temp_count + 1) * 0.1;

        memcpy(&input_A[temp_count], &a_val, sizeof(uint16_t));
        memcpy(&input_B[temp_count], &b_val, sizeof(uint16_t));
#else
        //f_val_A = (temp_count + 1) * 1.1; // (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        //f_val_B = (temp_count + 1) * 0.1; //(float)rand() / RAND_MAX * 2000.0 - 1000.0;
        f_val_A = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        f_val_B = (float)rand() / RAND_MAX * 2000.0 - 1000.0;

        input_A[temp_count] = float_to_bf16(f_val_A);
        input_B[temp_count] = float_to_bf16(f_val_B);
#endif
    }
}

void riscv_calculate_result() {

    uint64_t cycle_start;
    uint64_t cycle_end;
    char elapsedTimeStrValue[50]; 

    cycle_start = get_time();
    riscv_calculate(output_riscv_add, input_A, input_B, "vadd.bf16", DATA_SIZE);
    cycle_end = get_time();
    memset(elapsedTimeStrValue, 0, 50);
    floatToString((cycle_end - cycle_start) / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));
    printf("Time spent by RISC-V calculating vadd.bf16: %s us.\n", elapsedTimeStrValue);

    cycle_start = get_time();
    riscv_calculate(output_riscv_sub, input_A, input_B, "vsub.bf16", DATA_SIZE);
    cycle_end = get_time();
    memset(elapsedTimeStrValue, 0, 50);
    floatToString((cycle_end - cycle_start) / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));
    printf("Time spent by RISC-V calculating vsub.bf16: %s us.\n", elapsedTimeStrValue);

    cycle_start = get_time();
    riscv_calculate(output_riscv_mul, input_A, input_B, "vmul.bf16", DATA_SIZE);
    cycle_end = get_time();
    memset(elapsedTimeStrValue, 0, 50);
    floatToString((cycle_end - cycle_start) / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));
    printf("Time spent by RISC-V calculating vmul.bf16: %s us.\n", elapsedTimeStrValue);

    cycle_start = get_time();
    riscv_calculate(output_riscv_div, input_A, input_B, "vdiv.bf16", DATA_SIZE);
    cycle_end = get_time();
    memset(elapsedTimeStrValue, 0, 50);
    floatToString((cycle_end - cycle_start) / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));
    printf("Time spent by RISC-V calculating vdiv.bf16: %s us.\n", elapsedTimeStrValue);
}

void adjust_kernel() {

    resize_op_iteration_kernel(kernel_0, DATA_SIZE);
    resize_op_iteration_kernel(kernel_1, DATA_SIZE);
    resize_op_iteration_kernel(kernel_2, DATA_SIZE);
    resize_op_iteration_kernel(kernel_3, DATA_SIZE);
    resize_op_iteration_kernel(kernel_4, DATA_SIZE);
    resize_op_iteration_kernel(kernel_5, DATA_SIZE);

    kernel_loop_count_change(kernel_0, LOOP_COUNT);
    kernel_loop_count_change(kernel_1, LOOP_COUNT);
    kernel_loop_count_change(kernel_2, LOOP_COUNT);
    kernel_loop_count_change(kernel_3, LOOP_COUNT);
    kernel_loop_count_change(kernel_4, LOOP_COUNT);
    kernel_loop_count_change(kernel_5, LOOP_COUNT);

    // Change Kernel's Opcode
    kernel_op_change(kernel_0, TEST_OP_TYPE);
    kernel_op_change(kernel_1, TEST_OP_TYPE);
    kernel_op_change(kernel_2, TEST_OP_TYPE);
    kernel_op_change(kernel_3, TEST_OP_TYPE);
    kernel_op_change(kernel_4, TEST_OP_TYPE);
    kernel_op_change(kernel_5, TEST_OP_TYPE);

    kernel_input_a_sram_addr_change(kernel_0, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));
    kernel_input_a_sram_addr_change(kernel_1, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));
    kernel_input_a_sram_addr_change(kernel_2, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));
    kernel_input_a_sram_addr_change(kernel_3, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));
    kernel_input_a_sram_addr_change(kernel_4, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));
    kernel_input_a_sram_addr_change(kernel_5, (unsigned long )(INPUT_A_SRAM_BASE_ADDRESS / 1));

    kernel_input_b_sram_addr_change(kernel_0, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));
    kernel_input_b_sram_addr_change(kernel_1, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));
    kernel_input_b_sram_addr_change(kernel_2, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));
    kernel_input_b_sram_addr_change(kernel_3, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));
    kernel_input_b_sram_addr_change(kernel_4, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));
    kernel_input_b_sram_addr_change(kernel_5, (unsigned long )(INPUT_B_SRAM_BASE_ADDRESS / 1));

    kernel_output_c_sram_addr_change(kernel_0, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));
    kernel_output_c_sram_addr_change(kernel_1, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));
    kernel_output_c_sram_addr_change(kernel_2, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));
    kernel_output_c_sram_addr_change(kernel_3, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));
    kernel_output_c_sram_addr_change(kernel_4, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));
    kernel_output_c_sram_addr_change(kernel_5, (unsigned long )(RESULT_SRAM_BASE_ADDRESS / 1));

#if KERNEL_WITH_LOAD_STORE // with-load-store
    // Change Kernel's input_A address
    kernel_input_a_addr_change(kernel_0, input_A);
    kernel_input_a_addr_change(kernel_1, input_A);
    kernel_input_a_addr_change(kernel_2, input_A);
    kernel_input_a_addr_change(kernel_3, input_A);
    kernel_input_a_addr_change(kernel_4, input_A);
    kernel_input_a_addr_change(kernel_5, input_A);

    // Change Kernel's input_A address
    kernel_input_b_addr_change(kernel_0, input_B);
    kernel_input_b_addr_change(kernel_1, input_B);
    kernel_input_b_addr_change(kernel_2, input_B);
    kernel_input_b_addr_change(kernel_3, input_B);
    kernel_input_b_addr_change(kernel_4, input_B);
    kernel_input_b_addr_change(kernel_5, input_B);

    // Change Kernel's input_A address
    kernel_input_c_addr_change(kernel_0, output_npu_0);
    kernel_input_c_addr_change(kernel_1, output_npu_1);
    kernel_input_c_addr_change(kernel_2, output_npu_2);
    kernel_input_c_addr_change(kernel_3, output_npu_3);
    kernel_input_c_addr_change(kernel_4, output_npu_3);
    kernel_input_c_addr_change(kernel_5, output_npu_3);

    resize_converted_data_size_kernel(kernel_0, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_1, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_2, (int)((DATA_SIZE * 2 + 3)/4));
    resize_converted_data_size_kernel(kernel_3, (int)((DATA_SIZE * 2 + 3)/4));
#endif
}

void dump_data(char * data, int size) {
    for(int id = 0; id < size; id++) {
        printf("0x%02x, ", data[id] & 0xFF);
        if(((id+1) % 16) == 0) {
            printf("\n");
        }
    }
    printf("\n\n");
}

void dummy_store() {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size =  DUMMY_DATA_SIZE;

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = 0x00 + len;

        ddr_a = (long unsigned int)dummy_output_0 + len;
        store_command_to_npu(0, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)dummy_output_1 + len;
        store_command_to_npu(1, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)dummy_output_2 + len;
        store_command_to_npu(2, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)dummy_output_3 + len;
        store_command_to_npu(3, ddr_a, sram_a, loadSize);
        if(NUMBER_OF_CORES >= 5) {
            ddr_a = (long unsigned int)dummy_output_4 + len;
            store_command_to_npu(4, ddr_a, sram_a, loadSize);
        }
        if(NUMBER_OF_CORES >= 6) {
            ddr_a = (long unsigned int)dummy_output_5 + len;
            store_command_to_npu(5, ddr_a, sram_a, loadSize);
        }

        npu_store();

        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
    }
}

void load_kernel_into_npu(int npus) {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size =  (int)sizeof(kernel_0);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = 0x00 + len;

        if(npus & 0x1) {
            ddr_a = (long unsigned int)kernel_0 + len;
            load_command_to_npu(0, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x2) {
            ddr_a = (long unsigned int)kernel_1 + len;
            load_command_to_npu(1, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x4) {
            ddr_a = (long unsigned int)kernel_2 + len;
            load_command_to_npu(2, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x8) {
            ddr_a = (long unsigned int)kernel_3 + len;
            load_command_to_npu(3, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x10) {
            ddr_a = (long unsigned int)kernel_4 + len;
            load_command_to_npu(4, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x20) {
            ddr_a = (long unsigned int)kernel_5 + len;
            load_command_to_npu(5, sram_a, ddr_a, loadSize);
        }

        npu_load();

        //printf("%s - Offset: 0x%x, loadSize: %d\n", __func__, len, loadSize);
        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
//        dummy_store();
    }
}

void load_input_A_into_npu(int npus) {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size = (int)(sizeof(BF16) * DATA_SIZE);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = INPUT_A_SRAM_BASE_ADDRESS + len;

        ddr_a = (long unsigned int)input_A + len;
        if(npus & 0x1) {
            load_command_to_npu(0, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x2) {
            load_command_to_npu(1, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x4) {
            load_command_to_npu(2, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x8) {
            load_command_to_npu(3, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x10) {
            load_command_to_npu(4, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x20) {
            load_command_to_npu(5, sram_a, ddr_a, loadSize);
        }

        npu_load();

        //printf("%s - Offset: 0x%x, loadSize: %d\n", __func__, len, loadSize);
        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
//        dummy_store();
    }
}

void load_input_B_into_npu(int npus) {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size = (int)(sizeof(BF16) * DATA_SIZE);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = INPUT_B_SRAM_BASE_ADDRESS + len;

        ddr_a = (long unsigned int)input_B + len;
        if(npus & 0x1) {
            load_command_to_npu(0, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x2) {
            load_command_to_npu(1, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x4) {
            load_command_to_npu(2, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x8) {
            load_command_to_npu(3, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x10) {
            load_command_to_npu(4, sram_a, ddr_a, loadSize);
        }
        if(npus & 0x20) {
            load_command_to_npu(5, sram_a, ddr_a, loadSize);
        }

        npu_load();

        //printf("%s - Offset: 0x%x, loadSize: %d\n", __func__, len, loadSize);
        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
//        dummy_store();
    }
}

void load_kernel_data_into_npu() {

    int npus;

    if(NUMBER_OF_CORES == 5) {
        npus = 0x1F;
    } else if(NUMBER_OF_CORES == 6) {
        npus = 0x3F;
    } else {
        npus = 0x0F;
    }

    // Load kernel code at address 0 of npu
    load_kernel_into_npu(npus);
    printf("Kernel images are stored in each NPU.\n\n");

#if !KERNEL_WITH_LOAD_STORE // without-load-store

    load_input_A_into_npu(npus);
    printf("input_A is stored in all NPUs.\n");

    load_input_B_into_npu(npus);
    printf("input_B is stored in all NPUs.\n\n");
#endif
}

void store_result_into_ddr() {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size =  (int)(sizeof(BF16) * DATA_SIZE);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = RESULT_SRAM_BASE_ADDRESS + len;

        ddr_a = (long unsigned int)output_npu_0 + len;
        store_command_to_npu(0, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_1 + len;
        store_command_to_npu(1, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_2 + len;
        store_command_to_npu(2, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_3 + len;
        store_command_to_npu(3, ddr_a, sram_a, loadSize);
        if(NUMBER_OF_CORES >= 5) {
            ddr_a = (long unsigned int)output_npu_4 + len;
            store_command_to_npu(4, ddr_a, sram_a, loadSize);
        }
        if(NUMBER_OF_CORES >= 6) {
            ddr_a = (long unsigned int)output_npu_5 + len;
            store_command_to_npu(5, ddr_a, sram_a, loadSize);
        }

        npu_store();

        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
    }
}

void store_kernel_into_ddr() {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size =  (int)sizeof(kernel_0);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = 0x00 + len;

        ddr_a = (long unsigned int)output_npu_0 + len;
        store_command_to_npu(0, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_1 + len;
        store_command_to_npu(1, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_2 + len;
        store_command_to_npu(2, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_3 + len;
        store_command_to_npu(3, ddr_a, sram_a, loadSize);
        if(NUMBER_OF_CORES >= 5) {
            ddr_a = (long unsigned int)output_npu_4 + len;
            store_command_to_npu(4, ddr_a, sram_a, loadSize);
        }
        if(NUMBER_OF_CORES >= 6) {
            ddr_a = (long unsigned int)output_npu_5 + len;
            store_command_to_npu(5, ddr_a, sram_a, loadSize);
        }

        npu_store();

        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
    }
}

void store_input_A_into_ddr() {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size = (int)(sizeof(BF16) * DATA_SIZE);
    
    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = INPUT_A_SRAM_BASE_ADDRESS + len;

        ddr_a = (long unsigned int)output_npu_0 + len;
        store_command_to_npu(0, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_1 + len;
        store_command_to_npu(1, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_2 + len;
        store_command_to_npu(2, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_3 + len;
        store_command_to_npu(3, ddr_a, sram_a, loadSize);
        if(NUMBER_OF_CORES >= 5) {
            ddr_a = (long unsigned int)output_npu_4 + len;
            store_command_to_npu(4, ddr_a, sram_a, loadSize);
        }
        if(NUMBER_OF_CORES >= 6) {
            ddr_a = (long unsigned int)output_npu_5 + len;
            store_command_to_npu(5, ddr_a, sram_a, loadSize);
        }

        npu_store();

        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
    }
}

void store_input_B_into_ddr() {

    long unsigned int sram_a;
    long unsigned int ddr_a;
    int remaining;
    int loadSize;
    int size;

    size = (int)(sizeof(BF16) * DATA_SIZE);

    for (int len = 0; len < size; len += MAX_LOAD_STORE_CHUNK_SIZE) {
        remaining = size - len;
        loadSize = remaining < MAX_LOAD_STORE_CHUNK_SIZE ? remaining : MAX_LOAD_STORE_CHUNK_SIZE;
        sram_a = INPUT_B_SRAM_BASE_ADDRESS + len;

        ddr_a = (long unsigned int)output_npu_0 + len;
        store_command_to_npu(0, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_1 + len;
        store_command_to_npu(1, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_2 + len;
        store_command_to_npu(2, ddr_a, sram_a, loadSize);
        ddr_a = (long unsigned int)output_npu_3 + len;
        store_command_to_npu(3, ddr_a, sram_a, loadSize);
        if(NUMBER_OF_CORES >= 5) {
            ddr_a = (long unsigned int)output_npu_4 + len;
            store_command_to_npu(4, ddr_a, sram_a, loadSize);
        }
        if(NUMBER_OF_CORES >= 6) {
            ddr_a = (long unsigned int)output_npu_5 + len;
            store_command_to_npu(5, ddr_a, sram_a, loadSize);
        }

        npu_store();

        delay_in_usec(NPU_LOAD_STORE_MICRO_DELAY);
    }
}

#ifdef _NPU_LOAD_STORE_TEST_MODE_

void compare_load_store_data(int npu, char *org, char *npu_ls, int size) {

    int mismatch = 0;
    int idx;

    for(idx = 0;idx < size;idx++) {
        if(org[idx] != npu_ls[idx]) {
            printf("NPU%d Mismatch - origin[%4d]: 0x%02x, output_npu[%4d]: 0x%02x\n",
                        npu, idx, org[idx], idx, npu_ls[idx]);
            mismatch = 1;
        }
#if 0 // def __DEBUG_MODE__
        else {
            printf("NPU%d Match - origin[%4d]: 0x%02x, output_npu[%4d]: 0x%02x\n",
                        npu, idx, org[idx], idx, npu_ls[idx]);
        }
#endif
    }

    if(mismatch) {
        printf("[NPU%d] Fail to load & store\n", npu);
    } else {
        printf("[NPU%d] Success to load & store\n", npu);
    }
}

void load_store_test(int id) {

    char *org;
    char *npu_ls;
    int size;

    printf("\n>>> %s(%d)\n\n", __func__, id);

    if(id == 0) {
        load_kernel_into_npu(g_interrupt_mask);
    } else if(id == 1) {
        load_input_A_into_npu(g_interrupt_mask);
    } else if(id == 2) {
        load_input_B_into_npu(g_interrupt_mask);
    } else {
        return;
    }
    printf("\ncomplete npu_load\n\n");

    if(id == 0) {

        store_kernel_into_ddr();

        org = (char *)kernel_0;
        npu_ls = (char *)output_npu_0;
        compare_load_store_data(0, org, npu_ls, (int)sizeof(kernel_0));

        org = (char *)kernel_1;
        npu_ls = (char *)output_npu_1;
        compare_load_store_data(1, org, npu_ls, (int)sizeof(kernel_1));

        org = (char *)kernel_2;
        npu_ls = (char *)output_npu_2;
        compare_load_store_data(2, org, npu_ls, (int)sizeof(kernel_2));

        org = (char *)kernel_3;
        npu_ls = (char *)output_npu_3;
        compare_load_store_data(3, org, npu_ls, (int)sizeof(kernel_3));
        if(NUMBER_OF_CORES >= 5) {
            org = (char *)kernel_4;
            npu_ls = (char *)output_npu_4;
            compare_load_store_data(4, org, npu_ls, (int)sizeof(kernel_4));
        }
        if(NUMBER_OF_CORES >= 6) {
            org = (char *)kernel_5;
            npu_ls = (char *)output_npu_5;
            compare_load_store_data(5, org, npu_ls, (int)sizeof(kernel_5));
        }
    } else if(id == 1) {

        size = (int)(sizeof(BF16) * DATA_SIZE);
        store_input_A_into_ddr();

        org = (char *)input_A;
        compare_load_store_data(0, org, (char *)output_npu_0, size);
        compare_load_store_data(1, org, (char *)output_npu_1, size);
        compare_load_store_data(2, org, (char *)output_npu_2, size);
        compare_load_store_data(3, org, (char *)output_npu_3, size);
        if(NUMBER_OF_CORES >= 5) {
            compare_load_store_data(4, org, (char *)output_npu_4, size);
        }
        if(NUMBER_OF_CORES >= 6) {
            compare_load_store_data(5, org, (char *)output_npu_5, size);
        }
    } else if(id == 2) {

        size = (int)(sizeof(BF16) * DATA_SIZE);
        store_input_B_into_ddr();

        org = (char *)input_B;
        compare_load_store_data(0, org, (char *)output_npu_0, size);
        compare_load_store_data(1, org, (char *)output_npu_1, size);
        compare_load_store_data(2, org, (char *)output_npu_2, size);
        compare_load_store_data(3, org, (char *)output_npu_3, size);
        if(NUMBER_OF_CORES >= 5) {
            compare_load_store_data(4, org, (char *)output_npu_4, size);
        }
        if(NUMBER_OF_CORES >= 6) {
            compare_load_store_data(5, org, (char *)output_npu_5, size);
        }
    }
}

#endif

static inline void init_complete_exec() {

    unsigned long value = 0x0;

    npu_regSet(NPU_COMPLETE_INTERRUPT_RST, (long unsigned int)0x01); // reset - active high
    npu_regSet(NPU_COMPLETE_EXEC_REG, (long unsigned int)0x00);
    npu_regSet(NPU_COMPLETE_INTERRUPT_RST, (long unsigned int)0x00);

    value = npu_regGet(NPU_COMPLETE_EXEC_REG);
    if(value != 0) {
        printf("%s - Fail, value: %lx\r", value);
    }
}

static uint64_t check_complete_exec(uint64_t start) {

    unsigned long value = 0x0;
    uint64_t end;

    end = get_time();
#if NPU_COMPLETE_EXEC_INTERRUPT
    value = npu_regGet(NPU_COMPLETE_EXEC_REG);
#endif

    while(((value & g_interrupt_mask) != g_interrupt_mask) && 
    //while(((value) == 0) && 
          ((float)((end - start) / (SYS_CLK / 1000000)) < NPU_COMPLETE_EXEC_TIMEOUT)) {
        end = get_time();
#if NPU_COMPLETE_EXEC_INTERRUPT
        value = npu_regGet(NPU_COMPLETE_EXEC_REG);
#else
        printf("end: %016x\r", end);
        fflush(stdout);
#endif
    }
    printf("\nsrart: %016x\n", start);
    printf("end: %016x\n", end);
    printf("complete_exec_state: %lx\n", value);

    return end;
}

void get_average_csrrs_cycle() {

    uint64_t cycle_start;
    uint64_t cycle_end;

    cycle_start = get_time();
    for(int count = 0; count < 10; count++) {
        cycle_end = get_time();
    }
    elapsedCsrrsCycle = (cycle_end - cycle_start ) / 10;
    printf("elapsedCsrrsCycle: 0x%lx\n", elapsedCsrrsCycle);
}

int main_function() {
    
    uint64_t cycle_start;
    uint64_t cycle_end;
    int check[NUMBER_OF_CORES];
    char elapsedTimeStrValue[50]; 
    uint64_t elapsedCycle;
    uint64_t flops;
    float MFLOPS;
    char megaFlopsStrValue[50]; 

    printf("\n\n[%s Test] Using %d Cores\n", TEST_OP_TYPE, NUMBER_OF_CORES);

    memset(check, 0, NUMBER_OF_CORES * sizeof(int));

    adjust_kernel();
    printf("\nKernel images for each NPU have been prepared.\n");

#ifdef __DEBUG_MODE__
    printf("[kernel_0]\n");
    dump_data((char *)kernel_0, (int)sizeof(kernel_0));
    printf("[kernel_1]\n");
    dump_data((char *)kernel_1, (int)sizeof(kernel_1));
    printf("[kernel_2]\n");
    dump_data((char *)kernel_2, (int)sizeof(kernel_2));
    printf("[kernel_3]\n");
    dump_data((char *)kernel_3, (int)sizeof(kernel_3));
    if(NUMBER_OF_CORES >= 5) {
        g_interrupt_mask = 0x1F;
        printf("[kernel_4]\n");
        dump_data((char *)kernel_4, (int)sizeof(kernel_4));
    }
    if(NUMBER_OF_CORES >= 6) {
        g_interrupt_mask = 0x3F;
        printf("[kernel_5]\n");
        dump_data((char *)kernel_5, (int)sizeof(kernel_5));
    }
#endif

    load_kernel_data_into_npu();

    init_complete_exec();

    printf("\nRuns all NPUs.\n");
    cycle_start = get_time();

    npu_exec();

    cycle_end = check_complete_exec(cycle_start);
    if(((cycle_end - cycle_start) / (SYS_CLK / 1000000)) < NPU_COMPLETE_EXEC_TIMEOUT) {
        printf("\nAll NPUs have completed calculations.\n\n");
    } else {
        printf("\nTimeout - Waiting for all npu execs to finish.\n\n");
        return 0;
    }

#if !KERNEL_WITH_LOAD_STORE // without-load-store
    store_result_into_ddr();
    printf("The calculated result values were loaded into external memory.\n\n");
#endif

#ifdef __DEBUG_MODE__
    printf("[output_npu_0]\n");
    dump_data((char *)output_npu_0, (int)(sizeof(BF16) * DATA_SIZE));
    printf("[output_npu_1]\n");
    dump_data((char *)output_npu_1, (int)(sizeof(BF16) * DATA_SIZE));
    printf("[output_npu_2]\n");
    dump_data((char *)output_npu_2, (int)(sizeof(BF16) * DATA_SIZE));
    printf("[output_npu_3]\n");
    dump_data((char *)output_npu_3, (int)(sizeof(BF16) * DATA_SIZE));
    if(NUMBER_OF_CORES >= 5) {
        printf("[output_npu_4]\n");
        dump_data((char *)output_npu_4, (int)(sizeof(BF16) * DATA_SIZE));
    }
    if(NUMBER_OF_CORES >= 6) {
        printf("[output_npu_5]\n");
        dump_data((char *)output_npu_5, (int)(sizeof(BF16) * DATA_SIZE));
    }
#endif

    printf("\nCompare the results calculated by risc-v and the results calculated by NPUs.\n");
    // Check RISCV's Outpus & NPUs's Outputs Are Same, Input RISCV's OP Output Array in 1st Parameter
    if (TEST_OP_TYPE == "vadd.bf16") {
        check[0] = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_add, output_npu_0, DATA_SIZE);
        check[1] = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_add, output_npu_1, DATA_SIZE);
        check[2] = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_add, output_npu_2, DATA_SIZE);
        check[3] = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_add, output_npu_3, DATA_SIZE);
        if(NUMBER_OF_CORES >= 5) {
            check[4] = compare_riscv_and_npu(4, TEST_OP_TYPE, output_riscv_add, output_npu_4, DATA_SIZE);
        }
        if(NUMBER_OF_CORES >= 6) {
            check[5] = compare_riscv_and_npu(5, TEST_OP_TYPE, output_riscv_add, output_npu_5, DATA_SIZE);
        }
    } else if (TEST_OP_TYPE == "vsub.bf16") {
        check[0] = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_sub, output_npu_0, DATA_SIZE);
        check[1] = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_sub, output_npu_1, DATA_SIZE);
        check[2] = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_sub, output_npu_2, DATA_SIZE);
        check[3] = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_sub, output_npu_3, DATA_SIZE);
        if(NUMBER_OF_CORES >= 5) {
            check[4] = compare_riscv_and_npu(4, TEST_OP_TYPE, output_riscv_sub, output_npu_4, DATA_SIZE);
        }
        if(NUMBER_OF_CORES >= 6) {
            check[5] = compare_riscv_and_npu(5, TEST_OP_TYPE, output_riscv_sub, output_npu_5, DATA_SIZE);
        }
    } else if (TEST_OP_TYPE == "vmul.bf16"){
        check[0] = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_mul, output_npu_0, DATA_SIZE);
        check[1] = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_mul, output_npu_1, DATA_SIZE);
        check[2] = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_mul, output_npu_2, DATA_SIZE);
        check[3] = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_mul, output_npu_3, DATA_SIZE);
        if(NUMBER_OF_CORES >= 5) {
            check[4] = compare_riscv_and_npu(4, TEST_OP_TYPE, output_riscv_mul, output_npu_4, DATA_SIZE);
        }
        if(NUMBER_OF_CORES >= 6) {
            check[5] = compare_riscv_and_npu(5, TEST_OP_TYPE, output_riscv_mul, output_npu_5, DATA_SIZE);
        }
    } else if (TEST_OP_TYPE == "vdiv.bf16") {
        check[0] = compare_riscv_and_npu(0, TEST_OP_TYPE, output_riscv_div, output_npu_0, DATA_SIZE);
        check[1] = compare_riscv_and_npu(1, TEST_OP_TYPE, output_riscv_div, output_npu_1, DATA_SIZE);
        check[2] = compare_riscv_and_npu(2, TEST_OP_TYPE, output_riscv_div, output_npu_2, DATA_SIZE);
        check[3] = compare_riscv_and_npu(3, TEST_OP_TYPE, output_riscv_div, output_npu_3, DATA_SIZE);
        if(NUMBER_OF_CORES >= 5) {
            check[4] = compare_riscv_and_npu(4, TEST_OP_TYPE, output_riscv_div, output_npu_4, DATA_SIZE);
        }
        if(NUMBER_OF_CORES >= 6) {
            check[5] = compare_riscv_and_npu(5, TEST_OP_TYPE, output_riscv_div, output_npu_5, DATA_SIZE);
        }
    }
    printf("\n");

    // If All Pass, Print
    if (check[0] == DATA_SIZE) {
        printf("[NPU 0 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check[1] == DATA_SIZE) {
        printf("[NPU 1 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check[2] == DATA_SIZE) {
        printf("[NPU 2 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check[3] == DATA_SIZE) {
        printf("[NPU 3 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if(NUMBER_OF_CORES >= 5) {
        if (check[4] == DATA_SIZE) {
            printf("[NPU 4 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
        }
    }
    if(NUMBER_OF_CORES >= 6) {
        if (check[5] == DATA_SIZE) {
            printf("[NPU 5 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
        }
    }

    // 1 cycle = 25ns
    elapsedCycle = cycle_end - cycle_start - elapsedCsrrsCycle;

#ifdef __DEBUG_MODE__
    floatToString(elapsedCycle / (SYS_CLK / 1000000), 
                 elapsedTimeStrValue, sizeof(elapsedTimeStrValue));

    printf("\nRISC-V Time: %s us.\n", elapsedTimeStrValue);
#endif

    flops = NUMBER_OF_CORES * DATA_SIZE * LOOP_COUNT;

    printf("flops: %ld, elapsedCycle: %ld(1 cycle = 25ns)\n", flops, elapsedCycle);
    MFLOPS = (flops * 1000) / (elapsedCycle * 25);
    floatToString(MFLOPS, megaFlopsStrValue, sizeof(megaFlopsStrValue));
    printf(" =  %s MFLOPS.\n\n", megaFlopsStrValue);

    return 0;
}

int main() {
    
    uint64_t cycle_start;
    uint64_t cycle_end;
    int check[NUMBER_OF_CORES];
    char elapsedTimeStrValue[50]; 

    printf("\n========Init========\n\n");
    printf("Multi NAU Test\n");
    if(KERNEL_WITH_LOAD_STORE == 0) {
        printf("    Kernel without load/store functions\n\n");
    } else {
        printf("    Kernel with load/store functions\n\n");
    }

    memset(check, 0, NUMBER_OF_CORES * sizeof(int));

    init_variavles();
    printf("\ninput_A & input_B are filled with random data.\n");

#ifdef __DEBUG_MODE__
    printf("[input_A]\n");
    dump_data((char *)input_A, (int)(sizeof(BF16) * DATA_SIZE));
    printf("[input_B]\n");
    dump_data((char *)input_B, (int)(sizeof(BF16) * DATA_SIZE));
#endif

    riscv_calculate_result();
    printf("\nThe result values of risc-v for each function were calculated using input_A & input_B.\n");

    if(NUMBER_OF_CORES >= 5) {
        g_interrupt_mask = 0x1F;
    }
    if(NUMBER_OF_CORES >= 6) {
        g_interrupt_mask = 0x3F;
    }

    get_average_csrrs_cycle();

#ifdef _NPU_LOAD_STORE_TEST_MODE_

    load_store_test(0);
    load_store_test(1);
    load_store_test(2);

    return 0;
#endif

#if 1
    TEST_OP_TYPE = "vadd.bf16";
    main_function();
#endif

#if 0
    TEST_OP_TYPE = "vsub.bf16";
    main_function();
#endif

#if 0
    TEST_OP_TYPE = "vmul.bf16";
    main_function();
#endif

#if 0
    TEST_OP_TYPE = "vdiv.bf16";
    main_function();
#endif

    printf("\n========Finish========\n\n");
    
    return 0;
}
