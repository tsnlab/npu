#include "rocc.h"
#include <stdio.h>
#include <stdlib.h>

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

// static inline void npu_load()
// {
// 	asm volatile ("fence");
// 	ROCC_INSTRUCTION(3, 3);
// }

static inline void npu_load(int addr, unsigned long data)
{
	ROCC_INSTRUCTION_SS(3, data, addr, 3);
}

// static inline void npu_store()
// {
// 	// asm volatile ("fence");
// 	ROCC_INSTRUCTION(3, 4);
// }

static inline unsigned long npu_store(int addr)
{
	unsigned long value;
    // asm volatile ("nop");
	ROCC_INSTRUCTION_DSS(3, value, 0, addr, 4);
	return value;
}


unsigned long data1[16] = {0x0101L, 0x0202L, 0x0303L, 0x0404L, 0x0505L, 0x0606L, 0x0707L, 0x0808L, 0x0909L, 0x0a0aL, 0x0b0bL, 0x0c0cL, 0x0d0dL, 0x0e0eL, 0x0f0fL, 0x2020L};
unsigned long data2[16] = {0x0101L, 0x0202L, 0x0303L, 0x0404L, 0x0505L, 0x0606L, 0x0707L, 0x0808L, 0x0909L, 0x0a0aL, 0x0b0bL, 0x0c0cL, 0x0d0dL, 0x0e0eL, 0x0f0fL, 0x2020L};
unsigned long data3[16] = {0x0101L, 0x0202L, 0x0303L, 0x0404L, 0x0505L, 0x0606L, 0x0707L, 0x0808L, 0x0909L, 0x0a0aL, 0x0b0bL, 0x0c0cL, 0x0d0dL, 0x0e0eL, 0x0f0fL, 0x2020L};
unsigned long data4[16] = {0x0101L, 0x0202L, 0x0303L, 0x0404L, 0x0505L, 0x0606L, 0x0707L, 0x0808L, 0x0909L, 0x0a0aL, 0x0b0bL, 0x0c0cL, 0x0d0dL, 0x0e0eL, 0x0f0fL, 0x2020L};
// unsigned long data2[4] = {0x1234L, 0x5678L, 0x12345678L, 0x56781234L};
// unsigned long data3[4] = {0x123400003330L, 0x567822220000L, 0x54813515L, 0x816121L};
// unsigned long data4[4] = {0x1234000000000000L, 0x5678000000000000L, 0x93939L, 0x17839L};
unsigned long data5[16] = {0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L};
unsigned long data6[16] = {0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L};
unsigned long data7[16] = {0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L};
unsigned long data8[16] = {0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L, 0x0002L};


int main(void)
{
	unsigned long input = 317;
	unsigned long result = 0;

    int cnt = 0;

    // printf("regSet start. input = %lx \n", input); 
	// npu_regSet(1, input);
    // result = npu_regGet(1);
    // printf("regGet finished. result = %lx \n", result); 

    printf("data5[0] is %lx\n", data5[0]);
    printf("data5[1] is %lx\n", data5[1]);
	npu_load(0x0000, data1[0]);
	npu_load(0x0001, data1[1]);
	data5[0] = npu_store(0x0000);
	data5[1] = npu_store(0x0001);
	// npu_regSet(4, (long unsigned int)data2);
	// npu_regSet(7, (long unsigned int)data3);
	// npu_regSet(10, (long unsigned int)data4);
	// npu_regSet(2, 4);
	// npu_regSet(5, 4);
	// npu_regSet(8, 4);
	// npu_regSet(11, 4);
	// npu_regSet(3, 0);
	// npu_regSet(6, 0);
	// npu_regSet(9, 0);
	// npu_regSet(12, 0);
        // printf("load start\n");
    // npu_load();	
        // printf("load complete\n");
    // npu_regSet(1, (long unsigned int)data5);
    // npu_regSet(4, (long unsigned int)data6);
    // npu_regSet(7, (long unsigned int)data7);
    // npu_regSet(10, (long unsigned int)data8);
    printf("data5[0] is %lx\n", data5[0]);
    printf("data5[1] is %lx\n", data5[1]);
        // printf("Address of data[5] before is %lx\n", &data5[0]);
        // printf("store start\n");
    // npu_store();
        // printf("store complete\n");

        // printf("Address of data[5] after is %lx\n", data5[0]);

    // if (data1[0] != data5[0]){
    //     printf("ERR : data1[0]: %lx , data[5]: %lx \n", (void*)data1[0], (void*)data5[0]);
	//     return 1;
    // }else{
    //     printf("data matched!!");
	//     return 0;
    // }

    
    // while(result != input) {
    //     cnt++;
    //     printf("ERR!![%d]\n", cnt);
    // }
	return 0;
}