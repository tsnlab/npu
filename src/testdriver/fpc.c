#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include "fpc.h"

// operating frequence in MHz
#define    FREQ        50

// the number of element to be tested
#define    MAX_ELEMENT_SIZE    2048 // 2048
#define    ELEMENT_SIZE    512

#define    VARIABLE_ALIGN 128

#define    START_CORE_ID  0
#define    END_CORE_ID    2

#define    USED_DATA  1 // 0: random, 1: index

//#define    PRINT_SUCCESS_RESULT_DATA

//#define    LOAD_STORE_TEST

#ifdef LOAD_STORE_TEST
#define SRAM_POSITION 0 // 0: input_A, 1: input_B, 2: result_C
#endif

// core id
#define    CORE0       0
#define    CORE1       1
#define    CORE2       2
#define    CORE3       3
#define    MAX_CORE    2


// registers
#define    FPC_ID        0x00000000
#define    PROC_STATUS   0x00000008
#define    KERNEL_OFFSET 0x00000010
#define    KERNEL_SIZE   0x00000018
#define    KERNEL_LOAD   0x00000020
#define    RUN_FPU       0x00000028
#define    RUN_CYCLE     0x00000030

// FPU Code
#define    FPU_ADD        0
#define    FPU_SUB        1
#define    FPU_MUL        2
#define    FPU_DIV        3
#define    MAX_FPU        4

#define SYS_CLK 50000000

#define SIZE_M  1

#define _USE_VARIABLES_  1

// variables
#if _USE_VARIABLES_
//__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float a[ELEMENT_SIZE];
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float y[ELEMENT_SIZE] __attribute__((section(".nocache")));
#ifdef LOAD_STORE_TEST
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float y1[ELEMENT_SIZE] __attribute__((section(".nocache")));
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float y2[ELEMENT_SIZE] __attribute__((section(".nocache")));
#endif
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float a[ELEMENT_SIZE];
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float b[ELEMENT_SIZE];
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile static float r[ELEMENT_SIZE];
#else
float *d;
float *a;
float *b;
float *y;
#ifdef LOAD_STORE_TEST
float *y1;
float *y2;
#endif
float *r;
#endif
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile uint32_t kbuf[1024];
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile int load_cycle[MAX_FPU][MAX_CORE];
__attribute__ ((aligned (VARIABLE_ALIGN))) volatile int fpu_cycle[MAX_FPU][MAX_CORE];
volatile uint32_t klen    = 0;

volatile uint32_t not_ok  = 0;
volatile uint32_t not_cok  = 0;
// functions
void reg_write(uint32_t adr, uint64_t wd)
{
    FpcWrite(adr, wd);
}

uint64_t reg_read(uint64_t adr)
{
    uint64_t rd;
    FpcRead(adr, rd);
    return rd;
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

//---- generate data
#define    set_data(a, r, o) (((a) << 16) | ((r) << 8) | (o))
void generate_kernel(int fpu)
{
    int    size = MAX_ELEMENT_SIZE;
    int    count = ELEMENT_SIZE;
    int    bsiz, aa, ba, ya, am, bm, ym;

    bsiz    = size * 4;

    // main memory address
    am    = (int)a;
    bm    = (int)b;
    ym    = (int)y;

    // local sram address
    aa    = 0x100;
    ba    = aa + bsiz;
    ya    = ba + bsiz;

    // create kernel message
    int pos = 0;

    // load
    kbuf[pos++] = set_data((am>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(am&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(aa&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, count&0xFFFF, 0x03);    // load data

    // load
    kbuf[pos++] = set_data((bm>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(bm&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((ba>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(ba&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, count&0xFFFF, 0x03);    // load data

    // fpu
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(aa&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((ba>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(ba&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data((ya>>16)&0xFFFF, 0x03, 0x01); // set high value in reg3
    kbuf[pos++] = set_data(ya&0xFFFF, 0x03, 0x02);       // set low value in reg3
    kbuf[pos++] = set_data(0x00, count&0xFFFF, fpu+5);   // fpu code

    // store
    kbuf[pos++] = set_data((ym>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(ym&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((ya>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(ya&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, count&0xFFFF, 0x04);    // store data

    // return
    kbuf[pos++] = set_data(0x00, 0x00, 0x09);        // return

    klen    = 4 * pos;
}

#ifdef LOAD_STORE_TEST
void generate_load_store_kernel(int fpu)
{
    int    size = MAX_ELEMENT_SIZE;
    int    count = ELEMENT_SIZE;
    int    bsiz, aa, am, y0m, y1m,y2m;

    // buffer size in 256 elements
    bsiz    = size;

    // main memory address
    am    = (int)a;
    y0m    = (int)y;
    y1m    = (int)y1;
    y2m    = (int)y2;

    printf("am: %08x, y0m: %08x, y1m: %08x, y2m: %08x\n", am, y0m, y1m, y2m);
    
    aa    = 0x100 + bsiz * SRAM_POSITION;

    // create kernel message
    int pos = 0;

    // load
    kbuf[pos++] = set_data((am>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(am&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(aa&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, (count / SIZE_M)&0xFFFF, 0x03);    // load data

    // store
    kbuf[pos++] = set_data((y0m>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(y0m&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(aa&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, (count / SIZE_M)&0xFFFF, 0x04);    // store data

    // store
    kbuf[pos++] = set_data((y1m>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(y1m&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(aa&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, (count / SIZE_M)&0xFFFF, 0x04);    // store data

    // store
    kbuf[pos++] = set_data((y2m>>16)&0xFFFF, 0x01, 0x01); // set high value in reg1
    kbuf[pos++] = set_data(y2m&0xFFFF, 0x01, 0x02);       // set low value in reg1
    kbuf[pos++] = set_data((aa>>16)&0xFFFF, 0x02, 0x01); // set high value in reg2
    kbuf[pos++] = set_data(aa&0xFFFF, 0x02, 0x02);       // set low value in reg2
    kbuf[pos++] = set_data(0x00, (count / SIZE_M)&0xFFFF, 0x04);    // store data

    // return
    kbuf[pos++] = set_data(0x00, 0x00, 0x09);        // return

    klen    = 4 * pos;
}
#endif

void wait_done(int core)
{
    uint64_t rd;
    while((rd = reg_read(PROC_STATUS)) & (1 << core)) ;
}

static void floatToString(float floatValue, char* strValue, int maxLength) {

    int intPart = (int)floatValue;
    int decimalPart = (int)((floatValue - intPart) * 10000000); // Assuming 3 decimal places

    if(floatValue < 0) {
        decimalPart = 0 - decimalPart;
    }
    if (maxLength < 40) {
        // Buffer is too small to store anything meaningful
        return;
    }

    memset(strValue, 0, maxLength);
    snprintf(strValue, maxLength, "%d.%07d", intPart, decimalPart);
}

void print_value(int ok, int i, float fa, float fb)
{
    uint32_t *pa = (uint32_t *)&fa;
    uint32_t *pb = (uint32_t *)&fb;
    char aFloatStrValue[50];
    char bFloatStrValue[50];

    if(!ok) {
        floatToString(fa, aFloatStrValue, sizeof(aFloatStrValue));
        floatToString(fb, bFloatStrValue, sizeof(bFloatStrValue));
        printf("[F] : FPU[%4d]: %08X(%s) => %08X(%s)\n", i, *pa, aFloatStrValue, *pb, bFloatStrValue);
    } else {
#ifdef PRINT_SUCCESS_RESULT_DATA
        printf("[S] : FPU[%4d]: %08X => %08X\n", i, *pa, *pb);
#endif
    }
}

#ifdef LOAD_STORE_TEST
int loaf_store_test(void) {

    printf("---- FPU Controller Load-Store Test (Element size = %d) \n", ELEMENT_SIZE);

    printf("a: %p, y: %p, y1: %p, y2: %p\n", a, y, y1, y2);

    memset(a, 0, sizeof(a));
    memset(y, 0, sizeof(y));
    memset(y1, 0, sizeof(y1));
    memset(y2, 0, sizeof(y2));
    //---- main test
    for(int fpu = 0;fpu < 1;fpu++)
    {
        // generate kernel data
        generate_load_store_kernel(fpu);

        // init data
        for(int i = 0;i < ELEMENT_SIZE;i++)
        {
#if USED_DATA
            a[i] = (float)(i + 1);                  // a sources
#else
            a[i] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
#endif
        }

        // run fpu
        for(int core = START_CORE_ID;core < END_CORE_ID;core++)
        {
            printf("\tCORE: %d\n", core);

            // load kerneal
            reg_write(KERNEL_OFFSET, (uint64_t)kbuf);
            reg_write(KERNEL_SIZE, (uint64_t)(klen));
            reg_write(KERNEL_LOAD, (uint64_t)(1 << core));
            wait_done(core);

            // get load cycle
            load_cycle[fpu][core] = reg_read(8 * (6 + core));
            
            // run fpu
            reg_write(RUN_FPU, (uint64_t)core);
            wait_done(core);

            // get fpu cycles
            fpu_cycle[fpu][core] = reg_read(8 * (6 + core));

            printf("\nY0:\n");
            for(int i = 0;i < ELEMENT_SIZE;i++)
            {
                int ok = (y[i] >= a[i] ? y[i] - a[i] : a[i] - y[i]) < 0.0000001;
                print_value(ok, i, a[i], y[i]);
                if(!ok) {
                    not_ok++;
                }
            }
            printf("\nY1:\n");
            for(int i = 0;i < ELEMENT_SIZE;i++)
            {
                int ok = (y1[i] >= a[i] ? y1[i] - a[i] : a[i] - y1[i]) < 0.0000001;
                print_value(ok, i, a[i], y1[i]);
                if(!ok) {
                    not_ok++;
                }
            }
            printf("\nY2:\n");
            for(int i = 0;i < ELEMENT_SIZE;i++)
            {
                int ok = (y2[i] >= a[i] ? y2[i] - a[i] : a[i] - y2[i]) < 0.0000001;
                print_value(ok, i, a[i], y2[i]);
                if(!ok) {
                    not_ok++;
                }
            }
        }
    }

    // print result
    printf("[ Result ]\n");

    printf("---- FPU Controller Test Finished(not_ok: %d)\n", not_ok);

    return 0;
}
#endif

#define BUFFER_ALIGNMENT 0

#if !_USE_VARIABLES_
void release_variables() {
    if(d != NULL) {
        free(d);
    }
    if(a != NULL) {
        free(a);
    }
    if(b != NULL) {
        free(b);
    }
    if(y != NULL) {
        free(y);
    }
#ifdef LOAD_STORE_TEST
    if(y1 != NULL) {
        free(y1);
    }
    if(y2 != NULL) {
        free(y2);
    }
#endif
    if(r != NULL) {
        free(r);
    }
}
#endif

#if 0
void non_cacheable()
{
   asm volatile("li t0, 0x80000000");     // Base address of the memory region
   asm volatile("li t1, 0x80001000");     // End address of the memory region
   asm volatile("li t2, 0x8");            // Set MMU flags to mark the region as non-cacheable
   asm volatile("csrw pmpaddr0, t0");
   asm volatile("csrw pmpaddr1, t1");
   asm volatile("csrw pmpcfg0, t2");
}
#endif

int main(void) {

#if !_USE_VARIABLES_
//    d = malloc(sizeof(float) * ELEMENT_SIZE);
//    if(d == NULL) {
//        printf("OOM(d) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
//        return -1;
//    }
    a = malloc(sizeof(float) * ELEMENT_SIZE);
    if(a == NULL) {
        printf("OOM(a) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
    b = malloc(sizeof(float) * ELEMENT_SIZE);
    if(b == NULL) {
        printf("OOM(b) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
    y = malloc(sizeof(float) * ELEMENT_SIZE);
    if(y == NULL) {
        printf("OOM(y) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
#ifdef LOAD_STORE_TEST
    y1 = malloc(sizeof(float) * ELEMENT_SIZE);
    if(y1 == NULL) {
        printf("OOM(y1) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
    y2 = malloc(sizeof(float) * ELEMENT_SIZE);
    if(y2 == NULL) {
        printf("OOM(y2) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
#endif
    r = malloc(sizeof(float) * ELEMENT_SIZE);
    if(r == NULL) {
        printf("OOM(r) %u.\n", sizeof(float) * ELEMENT_SIZE + BUFFER_ALIGNMENT);
        release_variables();
        return -1;
    }
#endif

    memset(a, 0, sizeof(float) * ELEMENT_SIZE);
    memset(b, 0, sizeof(float) * ELEMENT_SIZE);
    memset(y, 0, sizeof(float) * ELEMENT_SIZE);
#ifdef LOAD_STORE_TEST
    memset(y1, 0, sizeof(float) * ELEMENT_SIZE);
    memset(y2, 0, sizeof(float) * ELEMENT_SIZE);
#endif
    memset(r, 0, sizeof(float) * ELEMENT_SIZE);

#ifdef LOAD_STORE_TEST
    int rst;

    rst = loaf_store_test();
    release_variables();
    return rst;
#endif

    printf("---- FPU Controller Test (Element size = %d) \n", ELEMENT_SIZE);
    printf("a: %p, b: %p, y: %p, r: %p\n", a, b, y, r);

//    delay_in_usec(1000000);

    //---- main test
    int count = 0;
//    for(count = 0;count < 2;count++)
    {
    for(int fpu = 0;fpu < MAX_FPU;fpu++)
    //for(int fpu = 3;fpu >= 0;fpu--)
    {
        printf("FPU: %s\n", fpu == 0 ? "ADD" : fpu == 1 ? "SUB" : fpu == 2 ? "MUL" : "DIV");

        // generate kernel data
        generate_kernel(fpu);

        // init data
        for(int i = 0;i < ELEMENT_SIZE;i++)
        {
#if USED_DATA
            a[i] = (float)(i + 1 + count);                  // a sources
            b[i] = (float)(i + 2 + count);            // b sources
#else
            a[i] = (float)rand() / RAND_MAX * 2000.0 - 1000.0;
            b[i] = (float)rand() / RAND_MAX * 3000.0 - 2500.0;
#endif
            r[i] = fpu == 0 ? a[i] + b[i] :   // reference
                   fpu == 1 ? a[i] - b[i] :
                   fpu == 2 ? a[i] * b[i] :
                      a[i] / b[i] ;
        }

        // run fpu
        for(int core = START_CORE_ID;core < END_CORE_ID;core++)
        {
            printf("\tCORE: %d\n", core);

            // load kerneal
            reg_write(KERNEL_OFFSET, (uint64_t)kbuf);
            reg_write(KERNEL_SIZE, (uint64_t)klen);
            reg_write(KERNEL_LOAD, (uint64_t)(1 << core));
            wait_done(core);

            // get load cycle
            load_cycle[fpu][core] = reg_read(8 * (6 + core));
            
            // run fpu
            reg_write(RUN_FPU, (uint64_t)core);
            wait_done(core);

            // get fpu cycles
            fpu_cycle[fpu][core] = reg_read(8 * (6 + core));

            // check result
            for(int i = 0;i < ELEMENT_SIZE;i++)
            {
                int ok = (y[i] >= r[i] ? y[i] - r[i] : r[i] - y[i]) < 0.0000001;
                volatile float *y_ptr;
                volatile float y_tmp;
                uint32_t *ptmp = (uint32_t *)&y_tmp;

                if(!ok) {
                    not_cok++;
                    y_ptr = &y[i];
                    y_tmp = *y_ptr;

                    printf("y_tmp: %08X\n", *ptmp);
                    ok = (y_tmp >= r[i] ? y_tmp - r[i] : r[i] - y_tmp) < 0.0000001;
                    // print_value(ok, i, r[i], y_tmp);
                } 

                print_value(ok, i, r[i], y[i]);
                if(!ok) {
                    not_ok++;
                }
            }
            delay_in_usec(1000);
        }
    }
    }

    // print result
    printf("[ Result ]\n");
    for(int fpu = 0;fpu < MAX_FPU;fpu++)
    {
        for(int core = 0;core < MAX_CORE;core++)
        {
            char *s = fpu == 0 ? "ADD" : fpu == 1 ? "SUB" : fpu == 2 ? "MUL" : "DIV";
            int lcyc = load_cycle[fpu][core];
            int fcyc = fpu_cycle[fpu][core];
            int tcyc = lcyc + fcyc;
            printf("%s#%d:: load: %d cycles, fpu: %d cycles, total: %d cycles, time: %d us\n", s, core, lcyc, fcyc, tcyc, tcyc / FREQ);
        }
    }

#if !_USE_VARIABLES_
    release_variables();
#endif
    printf("---- FPU Controller Test Finished(not_ok: %d, not_cok: %d)\n", not_ok, not_cok);

    return 0;
}

