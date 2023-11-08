#include "rocc.h"
#include <stdio.h>
#include <stdlib.h>

static inline void reg_write(int idx, unsigned long data)
{
	ROCC_INSTRUCTION_SS(3, data, idx, 0);
}

static inline unsigned long reg_read(int idx)
{
	unsigned long value;
	ROCC_INSTRUCTION_DSS(3, value, 0, idx, 1);
	return value;
}

// unsigned long data = 0x3421L;

int main(void)
{
	unsigned long input = 317;
	unsigned long result = 0;
    int cnt = 0;

    printf("reg_write start. input = %lx \n", input); 
	reg_write(1, input);
    result = reg_read(1);
    printf("reg_write finished. result = %lx \n", result); 

    
    while(result != input) {
        cnt++;
        printf("ERR!![%d]\n", cnt);
    }
	return 0;
}