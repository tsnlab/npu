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


// unsigned long data = 0x3421L;

int main(void)
{
	unsigned long input = 317;
	unsigned long result = 0;
    int cnt = 0;

    printf("reg_write start. input = %lx \n", input); 
	npu_regSet(1, input);
    result = npu_regGet(1);
    printf("reg_write finished. result = %lx \n", result); 

    
    while(result != input) {
        cnt++;
        printf("ERR!![%d]\n", cnt);
    }
	return 0;
}