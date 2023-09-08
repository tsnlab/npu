#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "xparameters.h"
#include "xil_io.h"
#include "xtime_l.h"

#define DATA_SIZE 10880		// Data Size
#define TEST_OP_TYPE "add"    // Op Type for Test, Use "add", "sub", "mul", "div"
#define TEST_CORE_ID 0        // Core Id for Test, 0 to 3

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

// Convert Op Type ad Op Code
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

// Calculate by Op Type
static void ps_calculate(float* output, float* input_A, float* input_B, char* op) {
    for (int i = 0; i < DATA_SIZE; i++) {
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

// Compare PS Output and PL Output
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
    printf("========Init========\n\n");
    printf("#%d Core %s %d Test\n\n", TEST_CORE_ID, TEST_OP_TYPE, DATA_SIZE);
    // 8192 Size Input data
    float input_A[DATA_SIZE];
    float input_B[DATA_SIZE];
    float output_ps[DATA_SIZE] = {0, }; // PS Output
    float output_pl[DATA_SIZE] = {0, }; // PL Output

    // Random Data Input
    for (int temp_count = 0; temp_count < DATA_SIZE; temp_count++) {
        input_A[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        input_B[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0; //
    }

    memcpy((float*)0x200000, input_A, sizeof(float) * DATA_SIZE);
    memcpy((float*)0x20AA00, input_B, sizeof(float) * DATA_SIZE);

    // PS Add
    ps_calculate(output_ps, input_A, input_B, TEST_OP_TYPE);

    // Kernel: need to align by 8bytes	
    __attribute__ ((aligned (8))) uint8_t kernel[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x80, 0x2a, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0xaa, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0xac, 0x03, 0x80, 0x2a, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0xac, 0x01, 0x03, 0x01, 0x00, 0x02, 0x03, 0x00, 0x56, 0x05, 0x80, 0x2a, 0x00, 0x01, 0x01, 0x21, 0x00, 0x02, 0x01, 0x00, 0x54, 0x01, 0x02, 0x01, 0x00, 0x02, 0x02, 0x00, 0x56, 0x04, 0x80, 0x2a, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    // Change Kernel's Opcode by 
    kernel_op_change(kernel, TEST_OP_TYPE);

    // Test loop
    for (int count = 2; count <= DATA_SIZE; count += 2) {
        
        resize_kernel(kernel, count);

        // Input's Memory Cache Flush
        Xil_DCacheFlushRange(0x200000, (uint32_t)sizeof(float) * count);
        Xil_DCacheFlushRange(0x20AA00, (uint32_t)sizeof(float) * count);

        // Kernel Cache Flush
        Xil_DCacheFlushRange(kernel, sizeof(kernel));

        // NPU Set
        volatile uint32_t *npu_base = (uint32_t*)0x43C00000; // kernel offset
        npu_base[0] = (uint32_t)kernel;// the address of kernel in main memory
        npu_base[1] = sizeof(kernel); // the size of kernel, also need to align by 8bytes

        // To Get PS's time
        XTime time_start, time_end;
        XTime_GetTime(&time_start);

        // Start NPU operation
        npu_base[2] = TEST_CORE_ID; // Core Id

        while(npu_base[3] & (1 << TEST_CORE_ID)) {
            Xil_DCacheInvalidateRange(&npu_base[3], (uint32_t)sizeof(npu_base[3]));
        }; // wait until operation is done (not busy)

        XTime_GetTime(&time_end);

        // Output's Memory Cache Invalidate
        Xil_DCacheInvalidateRange(0x215400, (uint32_t)sizeof(float) * count); //
        
        memcpy(output_pl, (float*)0x215400, sizeof(float) * count);

        // Check PS's Outpus & PL's Outputs Are Same
        int check = compare_ps_and_pl(output_ps, output_pl, count);

        // If All Pass, Print
        if (check == count) {
            printf("[Test Case %d] All Pass\n", count);
        }

        // NPU Times, PS: 667MHz & PL: 125MHz, npu_base[4 ~ 7]: Core #0 ~ #3's Run Cycle registers
        printf("PS Time: %.3fus & PL Time: %.3fus\n", 2.0 * (time_end - time_start) / (XPAR_CPU_CORTEXA9_CORE_CLOCK_FREQ_HZ / 1000000), npu_base[4 + TEST_CORE_ID] * 8.00 / 1000.00);
    }
    printf("========Finish========\n");
    
    return 0;
}
