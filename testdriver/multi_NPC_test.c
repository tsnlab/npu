#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "xparameters.h"
#include "xil_io.h"
#include "xtime_l.h"

#define DATA_SIZE 2048      // Data Size
#define TEST_OP_TYPE "add"  // Op Type for Test, Use "add", "sub", "mul", "div"
#define NUMBER_OF_CORES 4   // Number f Cores used at the same time

static void resize_kernel(uint8_t* kernel, int size) {
    // Seperate Size Bytes Low and High
    uint8_t byte_low;
    uint8_t byte_high;
    
    // Devide Size
    byte_low = size & 0xff;
    byte_high = size >> 8;

    // Change Number to Add in Kernel
    kernel[17] = byte_low;
    kernel[18] = byte_high;
    kernel[37] = byte_low;
    kernel[38] = byte_high;
    kernel[65] = byte_low;
    kernel[66] = byte_high;
    kernel[85] = byte_low;
    kernel[86] = byte_high;
}

static void kernel_op_change(uint8_t* kernel, char* op) {
    uint8_t opcode;
    if (op == "add") {
        opcode = 0x05;
    } else if (op == "sub") {
        opcode = 0x06;
    } else if (op == "mul"){
        opcode = 0x07;
    } else if (op == "div") {
        opcode = 0x08;
    } else {
        printf("Wrong OP!!!\n");
    }
    kernel[64] = opcode;
}

static void ps_calculate(float* output, float* input_A, float* input_B, char* op, int size) {
    for (int i = 0; i < size; i++) {
        if (op == "add") {
            output[i] = input_A[i] + input_B[i];
        } else if (op == "sub") {
            output[i] = input_A[i] - input_B[i];
        } else if (op == "mul"){
            output[i] = input_A[i] * input_B[i];
        } else if (op == "div") {
            output[i] = input_A[i] / input_B[i];
        }
    }
}

static int compare_ps_and_pl(float* output_ps, float* output_pl, int count) {
    // Check How Many Are Correct
    int check = 0;
    for (int i = 0; i < count; i++) {
        if (output_ps[i] != output_pl[i]) {
            printf("[Test Case %d] FAIL\nPS data[%d]: %f\nPL data[%d]: %f\nPS data[%d] - PL data[%d] = %f\n", count, i, output_ps[i], i, output_pl[i], i, i, output_ps[i] - output_pl[i]);
        } else {
            check += 1;
        }
    }
    return check;
}

int main() {
    printf("\n========Init========\n\n");
    printf("Multi Core Test\n");
    printf("[%s Test] Using %d Cores\n\n", TEST_OP_TYPE, NUMBER_OF_CORES);
    // 8192 Size Input data
    float input_A[DATA_SIZE];
    float input_B[DATA_SIZE];
    
    float output_ps_add[DATA_SIZE] = {0, }; // PS add Output
    float output_ps_sub[DATA_SIZE] = {0, }; // PS sub Output
    float output_ps_mul[DATA_SIZE] = {0, }; // PS mul Output
    float output_ps_div[DATA_SIZE] = {0, }; // PS div Output
    
    float output_core_0[DATA_SIZE] = {0, }; // PL core 0 Output
    float output_core_1[DATA_SIZE] = {0, }; // PL core 1 Output
    float output_core_2[DATA_SIZE] = {0, }; // PL core 2 Output
    float output_core_3[DATA_SIZE] = {0, }; // PL core 3 Output

    // Random Data Input
    for (int temp_count = 0; temp_count < DATA_SIZE; temp_count++) {
        input_A[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        input_B[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
    }

    ps_calculate(output_ps_add, input_A, input_B, "add", DATA_SIZE);
    ps_calculate(output_ps_sub, input_A, input_B, "sub", DATA_SIZE);
    ps_calculate(output_ps_mul, input_A, input_B, "mul", DATA_SIZE);
    ps_calculate(output_ps_div, input_A, input_B, "div", DATA_SIZE);

    // Memory Input
    memcpy((float*)0x200000, input_A, sizeof(float) * DATA_SIZE);
    memcpy((float*)0x202000, input_B, sizeof(float) * DATA_SIZE);

    // Kernel: DATA_SIZE 0, add OP	
    uint8_t kernel_0[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x00, 0x00, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x20, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x03, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x01, 0x03, 0x00, 0x00, 0x02, 0x03, 0x00, 0x42, 0x05, 0x00, 0x00, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x40, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x42, 0x04, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00};
    uint8_t kernel_1[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x20, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x01, 0x03, 0x00, 0x00, 0x02, 0x03, 0x00, 0x42, 0x05, 0x00, 0x02, 0x00, 0x01, 0x01, 0x30, 0x00, 0x02, 0x01, 0x00, 0x40, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x42, 0x04, 0x00, 0x02, 0x00, 0x09, 0x00, 0x00, 0x00};
    uint8_t kernel_2[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x20, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x01, 0x03, 0x00, 0x00, 0x02, 0x03, 0x00, 0x42, 0x05, 0x00, 0x02, 0x00, 0x01, 0x01, 0x40, 0x00, 0x02, 0x01, 0x00, 0x40, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x42, 0x04, 0x00, 0x02, 0x00, 0x09, 0x00, 0x00, 0x00};
    uint8_t kernel_3[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x20, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x03, 0x00, 0x02, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x01, 0x03, 0x00, 0x00, 0x02, 0x03, 0x00, 0x42, 0x05, 0x00, 0x02, 0x00, 0x01, 0x01, 0x50, 0x00, 0x02, 0x01, 0x00, 0x40, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x42, 0x04, 0x00, 0x02, 0x00, 0x09, 0x00, 0x00, 0x00};
    resize_kernel(kernel_0, DATA_SIZE);
    resize_kernel(kernel_1, DATA_SIZE);
    resize_kernel(kernel_2, DATA_SIZE);
    resize_kernel(kernel_3, DATA_SIZE);
    
    // Change Kernel's Opcode
    kernel_op_change(kernel_0, TEST_OP_TYPE);
    kernel_op_change(kernel_1, TEST_OP_TYPE);
    kernel_op_change(kernel_2, TEST_OP_TYPE);
    kernel_op_change(kernel_3, TEST_OP_TYPE);

    // Input's Memory Cache Flush
    Xil_DCacheFlushRange(0x200000, (uint32_t)sizeof(float) * DATA_SIZE);
    Xil_DCacheFlushRange(0x202000, (uint32_t)sizeof(float) * DATA_SIZE);

    // Kernel Cache Flush
    Xil_DCacheFlushRange(kernel_0, sizeof(kernel_0));
    Xil_DCacheFlushRange(kernel_1, sizeof(kernel_1));
    Xil_DCacheFlushRange(kernel_2, sizeof(kernel_2));
    Xil_DCacheFlushRange(kernel_3, sizeof(kernel_3));

    // To Get PS's time
    XTime time_start, time_end;

    // NPU Set
    uint32_t *npu_base = (uint32_t*)0x43C00000; // kernel offset
    //Core 0
    npu_base[0] = (uint32_t)kernel_0;// the address of kernel in main memory
    npu_base[1] = sizeof(kernel_0); // the size of kernel
    
    XTime_GetTime(&time_start); // Start Measuring Time when core 0 is Start
    npu_base[2] = 0; // Core Id
    //Core 1
    npu_base[0] = (uint32_t)kernel_1;// the address of kernel in main memory
    npu_base[1] = sizeof(kernel_1);
    npu_base[2] = 1;
    //Core 2
    npu_base[0] = (uint32_t)kernel_2;// the address of kernel in main memory
    npu_base[1] = sizeof(kernel_2);
    npu_base[2] = 2;
    //Core 3
    npu_base[0] = (uint32_t)kernel_3;// the address of kernel in main memory
    npu_base[1] = sizeof(kernel_3);
    npu_base[2] = 3;

    while(npu_base[3] & 0b1111) { // wait until operation is done (not busy)
        Xil_DCacheInvalidateRange(&npu_base[3], (uint32_t)sizeof(npu_base[3]));
    }; 
    XTime_GetTime(&time_end); // Get End Time
    
    // Output's Memory Cache Invalidate
    Xil_DCacheInvalidateRange(0x204000, (uint32_t)sizeof(float) * DATA_SIZE);
    Xil_DCacheInvalidateRange(0x304000, (uint32_t)sizeof(float) * DATA_SIZE);
    Xil_DCacheInvalidateRange(0x404000, (uint32_t)sizeof(float) * DATA_SIZE);
    Xil_DCacheInvalidateRange(0x504000, (uint32_t)sizeof(float) * DATA_SIZE);
    
    // Memory Output
    memcpy(output_core_0, (float*)0x204000, sizeof(float) * DATA_SIZE);
    memcpy(output_core_1, (float*)0x304000, sizeof(float) * DATA_SIZE);
    memcpy(output_core_2, (float*)0x404000, sizeof(float) * DATA_SIZE);
    memcpy(output_core_3, (float*)0x504000, sizeof(float) * DATA_SIZE);

    // Check PS's Outpus & PL's Outputs Are Same, Input PS's OP Output Array in 1st Parameter
    // If Operator is not "add", should replace output_ps_add with output_ps_{other operator}
    int check0 = compare_ps_and_pl(output_ps_add, output_core_0, DATA_SIZE);
    int check1 = compare_ps_and_pl(output_ps_add, output_core_1, DATA_SIZE);
    int check2 = compare_ps_and_pl(output_ps_add, output_core_2, DATA_SIZE);
    int check3 = compare_ps_and_pl(output_ps_add, output_core_3, DATA_SIZE);
    // If All Pass, Print
    if (check0 == DATA_SIZE) {
        printf("[Core 0 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check1 == DATA_SIZE) {
        printf("[Core 1 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check2 == DATA_SIZE) {
        printf("[Core 2 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }
    if (check3 == DATA_SIZE) {
        printf("[Core 3 Test Case %d] %s All Pass\n", DATA_SIZE, TEST_OP_TYPE);
    }

    // NPU Times, XTime Counter increases by one at every two processor cycles, PS: 667MHz
    printf("\nPS Time: %.3f us.\n", 2.0 * (time_end - time_start) / (XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ / 1000000));

    // Test Case N's Total Cycles, PL: 100MHz
    printf("[#0 Core] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 0], npu_base[4 + 0] / 100.00);
    printf("[#1 Core] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 1], npu_base[4 + 1] / 100.00);
    printf("[#2 Core] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 2], npu_base[4 + 2] / 100.00);
    printf("[#3 Core] Total cycles: %d\tConvert Times: %.3fus\n", npu_base[4 + 3], npu_base[4 + 3] / 100.00);

    printf("\n========Finish========\n");
    
    return 0;
}