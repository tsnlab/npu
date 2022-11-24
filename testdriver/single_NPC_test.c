#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "xparameters.h"
#include "xil_io.h"
#include "xtime_l.h"

#define DATA_SIZE 2048      // Data Size
#define END_SIZE 2048       // Count Size for Test
#define TESTOPTYPE "add"    // Op Type for Test, Use "add", "sub", "mul", "div"
#define TESTCOREID 1        // Core Id for Test

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
static int compare_ps_and_pl(float* output_Ps, float* output_Pl, int count) {
    // Check How Many Are Correct
    int check = 0;
    for (int i = 0; i < count; i++) {
            if (output_Ps[i] != output_Pl[i]) {
                printf("[Test Case %d] FAIL\nPS data[%d]: %f\nPL data[%d]: %f\nPS data[%d] - PL data[%d] = %f\n", count, i, output_Ps[i], i, output_Pl[i], i, i, output_Ps[i] - output_Pl[i]);
            } else {
                check += 1;
            }
        }
    return check;
}

int main() {
	printf("========Init========\n\n");
	printf("#%d Core %s Test\n\n", TESTCOREID, TESTOPTYPE);
	// 8192 Size Input data
	float input_A[DATA_SIZE];
	float input_B[DATA_SIZE];
	float output_Ps[DATA_SIZE] = {0, }; // PS Output
    float output_Pl[DATA_SIZE] = {0, }; // PL Output

    // Random Data Input
    for (int temp_count = 0; temp_count < DATA_SIZE; temp_count++){
        input_A[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
        input_B[temp_count] = (float)rand() / RAND_MAX * 2000.0 - 1000.0; //
    }

    // Memory Input
	memcpy((float*)0x200000, input_A, sizeof(float) * DATA_SIZE);
	memcpy((float*)0x202000, input_B, sizeof(float) * DATA_SIZE);
    
    // PS Add
    ps_calculate(output_Ps, input_A, input_B, TESTOPTYPE);


    // Kernel: DATA_SIZE 0, add OP	
	uint8_t kernel[] = {0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x02, 0x03, 0x00, 0x00, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x20, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x03, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x22, 0x01, 0x03, 0x00, 0x00, 0x02, 0x03, 0x00, 0x42, 0x05, 0x00, 0x00, 0x00, 0x01, 0x01, 0x20, 0x00, 0x02, 0x01, 0x00, 0x40, 0x01, 0x02, 0x00, 0x00, 0x02, 0x02, 0x00, 0x42, 0x04, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00};
    
    // Change Kernel's Opcode by 
    kernel_op_change(kernel, TESTOPTYPE);

    // Seperate Count Bytes Low and High
    uint8_t byte_low;
    uint8_t byte_high;

    // Calculate by increasing by 1 from 1 to 2048
    for (int count = 1; count <= END_SIZE; count++) {
        // Devide count
        byte_low = count & 0xff;
        byte_high = count >> 8;

        // Change Number to Add in Kernel
        kernel[17] = byte_low;
        kernel[18] = byte_high;
        kernel[37] = byte_low;
        kernel[38] = byte_high;
        kernel[65] = byte_low;
        kernel[66] = byte_high;
        kernel[85] = byte_low;
        kernel[86] = byte_high;
        
        // Input's Memory Cache Flush
        Xil_DCacheFlushRange(0x200000, (uint32_t)sizeof(float) * count);
        Xil_DCacheFlushRange(0x202000, (uint32_t)sizeof(float) * count);

        // Kernel Cache Flush
        Xil_DCacheFlushRange(kernel, sizeof(kernel));

        // NPU Set
        uint32_t *npu_base = (uint32_t*)0x43C00000; // kernel offset
        npu_base[0] = (uint32_t)kernel;// the address of kernel in main memory
        npu_base[1] = sizeof(kernel); // the size of kernel
        
        // To Get PS's time
        XTime tStart, tEnd;
        XTime_GetTime(&tStart);

        // Start NPU operation
        npu_base[2] = TESTCOREID; // core id = 0
        while(npu_base[3] & (1 << TESTCOREID)) {
            Xil_DCacheInvalidateRange(&npu_base[3], (uint32_t)sizeof(npu_base[3]));
        }; // wait until operation is done (not busy)

        XTime_GetTime(&tEnd);
        
        // Output's Memory Cache Invalidate
        Xil_DCacheInvalidateRange(0x204000, (uint32_t)sizeof(float) * count); //
        
        // Memory Output
        memcpy(output_Pl, (float*)0x204000, sizeof(float) * count);

        // Check PS's Outpus & PL's Outputs Are Same
        int check = compare_ps_and_pl(output_Ps, output_Pl, count);

        // If All Pass, Print
        if (check == count) {
            printf("[Test Case %d] All Pass\n", count);
        }

        // NPU Times, XTime Counter increases by one at every two processor cycles, PS: 667MHz
        printf("Time: %.3f us.\n", 2.0 * (tEnd - tStart) * (1.0 / 677.0));

        // Test Case N's Total Cycles, PL: 100MHz
        printf("Total cycles: %d\tConvert Times: %.3fus\n\n", npu_base[4 + TESTCOREID], npu_base[4 + TESTCOREID] / 100.00);
    }
    printf("========Finish========\n");
    
    return 0;
}
