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


unsigned long data1[2] = {0x1234L, 0x5678L};
unsigned long data2[2] = {0x0000L, 0x0000L};

int main(void)
{
	unsigned long input = 317;
	unsigned long result = 0;

    int cnt = 0;

    printf("reg_write start. input = %lx \n", input); 
	npu_regSet(1, input);
    result = npu_regGet(1);
    printf("reg_write finished. result = %lx \n", result); 

	npu_regSet(1, (long unsigned int)data1);
	npu_regSet(2, 2);
	npu_regSet(3, 0);
    npu_load();	
    npu_regSet(1, (long unsigned int)data2);
    npu_store();
    if (data1[0] != data2[0]){
        printf("ERR!!");
	    return 1;
    }else{
        printf("data matched!!");
	    return 1;
    }

    
    // while(result != input) {
    //     cnt++;
    //     printf("ERR!![%d]\n", cnt);
    // }
	return 0;
}