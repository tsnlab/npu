# CONNX NPU - with RISC-V processor built on chipyard

## Getting Started
1. repository를 클론한다.
```bash
git clone https://github.com/tsnlab/npu.git
```
2. chipyard 서브 모듈을 불러온다.
```bash
cd chipyard
git submodule update --init
```
3. chipyard를 설치한다. 설치 방법과 순서는 아래 링크를 따른다. 이 때, 이미 npu repository는 chipyard를 submodule로 포함하고 있으므로 1.4.2 Setting up the Chipyard Repo 항목의 chipyard repository 클론 순서는 생략하고, 이후의 빌드 작업부터 다시 수행한다.
    https://chipyard.readthedocs.io/en/stable/Chipyard-Basics/Initial-Repo-Setup.html#initial-repository-setup
4. step 9에서 빌드가 실패 시, 아래 커맨드로 다시 빌드 시도한다.
```bash
./build-setup.sh riscv-tools -s 9
```

## Generating Project
1. npu 소스코드를 chipyard directory에 복사한다. 
```bash
make
```
2. env.sh를 실행한다.
```bash
cd chipyard
source env.sh
```
3. bitstream file을 생성한다.
```bash
cd fpga
make SUB_PROJECT=nexysvideo bitstream
```

## RISC-V instruction extensions
 * store 
 * load 
 * exec 
 * set reg Idx, value
 * get reg Idx

## NPU(RoCC) Registers
```C
Core {
    0: unsigned long    // reserved
    1: unsigned long    // host memory address for core#0 
    2: unsigned long    // data size (16byte width) for core#0
    3: unsigned long    // local memory address for core#0
    4: unsigned long    // host memory address for core#1
    5: unsigned long    // data size (16byte width) for core#1
    6: unsigned long    // local memory address for core#1
    7: unsigned long    // host memory address for core#2
    8: unsigned long    // data size (16byte width) for core#2
    9: unsigned long    // local memory address for core#2
    10: unsigned long   // host memory address for core#3
    11: unsigned long   // data size (16byte width) for core#3
    12: unsigned long   // local memory address for core#3
}
```

## NPU Core Registers
```C
Core {
    zero: uint32 // reg 0 zero
    a: uint32    // reg 1 a
    b: uint32    // reg 2 b
    c: uint32    // reg 3 c
    d: uint32    // reg 4 d
    e: uint32    // reg 5 e
    f: uint32    // reg 6 f
    g: uint32    // reg 7 g
    ip: uint32   // reg 14 instruction pointer (ip * 4 is memory location)
    csr: uint32  // reg 15 control status register
}
```

### CSR register bits
 0. running - 0: not running, 1: running
 1. loading - 1: loading or storing, 0: data is not moving b/w CPU and NPU
 2\~30. reserved
 31. error - 0: no error, 1: error

## NPU Core instructions
모든 instruction은 32bit 단위로 zero padding 되어있음.
모든 instruction은 little endian으로 표기됨.
아래는 NPU Core에서 지원하는 instruction 목록이자 opcode임.
모든 opcode는 uint8 크기임.

reg - unsigned 4 bits
uint# - unsigned # bits
int# - signed # bits

 00. nop
 01. set
 02. seti
 03. seti\_low
 04. seti\_high
 05. get
 06. mov
 07. load
 08. store
 09. vadd.bf16
 0a. vsub.bf16
 0b. vmul.bf16
 0c. vdiv.bf16
 0d. add.i32
 0e. sub.i32
 0f. ifz
 10. ifeq
 11. ifneq
 12. jmp
 ff. return

### for every opcode
syntax
    any opcode

pseudo code
    opcode = mem[ip * 4]
    execute opcode
    ip = ip + 1

### nop - no operator
syntax
    nop

parameters

### set - local memory 값을 register로 복사함
syntax
    set %dest %src

parameters
    %reg: reg    // register 번호
    %mem: uint20 // align된 local memory 주소

pseudo code
    reg[%reg] = mem[%mem * 4]

### seti - register의 값을 설정함
syntax
    seti %reg %value

parameters
    %dest: reg     // register 번호
    %value: uint20 // register에 입력될 값

pseudo code
    reg[%dest] = (0x000 << 24) | (%value & 0xfffff)

### seti\_low - register의 값을 설정함
syntax
    seti_low %reg %value padding

parameters
    %reg: reg      // register 번호
    padding: u4
    %value: uint16 // register에 입력될 값

pseudo code
    reg[%reg] = (reg[%reg] & 0xffff0000) | (%value & 0xffff)

### seti\_high - register의 값을 설정함
syntax
    seti_high %reg %value padding

parameters
    %reg: reg      // register 번호
    padding: u4
    %value: uint16 // register에 입력될 값

pseudo code
    reg[%reg] = ((%value & 0xffff) << 16) | (reg[%reg] & 0xffff)

### get - register의 값을 local memory로 복사함
syntax
    get %src %dest

parameters
    %src: reg     // register 번호
    %dest: uint20 // align된 local memory 주소

pseudo code
    mem[%dest * 4] = reg[%src]

### mov - register의 값을 register로 복사함
syntax
    mov %dest %src

parameters
    %dest: reg  // register 번호
    %src: reg   // register 번호

pseudo code
    reg[%dest] = reg[%src]

### load - Host memory로부터 local memory로 데이터 복사
syntax
    load %dest %src %count

parameters
    %dest: reg  
    %src: reg  
    %count: reg  

pseudo code
    dest = reg[%dest] * 4  // Local memory는 4 bytes 단위로 align 되어있다고 가정
    src = reg[%src] * 128  // Host memory는 128 bytes 단위로 align 되어있다고 가정
    size = reg[%count] * 4  // 4 bytes 단위로 roundup 되어있다고 가정

    memcpy(dest, src, size)

### store - local memory로부터 host memory로 데이터 복사
syntax
    store %dest %src %count

parameters
    %dest: reg  
    %src: reg  
    %count: reg  

pseudo code
    dest = reg[%dest] * 128 // Host memory는 128 bytes 단위로 align 되어있다고 가정
    src = reg[%src] * 4     // Local memory는 4 bytes 단위로 align 되어있다고 가정
    size = reg[%count] * 4  // 4 bytes 단위로 roundup 되어있다고 가정

    memcpy(dest, src, size)

### vadd.bf16
syntax
    vadd.bf16 %c %a %b %count padding

parameters
    %c: reg  
    %a: reg  
    %b: reg  
    %count: reg  
    padding: uint8

pseudo code
    bf16* a = reg[%a] * 4
    bf16* b = reg[%b] * 4
    bf16* c = reg[%c] * 4
    int32* count = reg[%count]

    for i in 0..*count
        c[i] = a[i] + b[i]

### vsub.bf16
syntax
    vsub.bf16 %c %a %b %count padding

parameters
    %c: reg  
    %a: reg  
    %b: reg  
    %count: reg  
    padding: uint8

pseudo code
    bf16* a = reg[%a] * 4
    bf16* b = reg[%b] * 4
    bf16* c = reg[%c] * 4
    int32* count = reg[%count]

    for i in 0..*count
        c[i] = a[i] - b[i]

### vmul.bf16
syntax
    vmul.bf16 %c %a %b %count padding

parameters
    %c: reg  
    %a: reg  
    %b: reg  
    %count: reg  
    padding: uint8

pseudo code
    bf16* a = reg[%a] * 4
    bf16* b = reg[%b] * 4
    bf16* c = reg[%c] * 4
    int32* count = reg[%count]

    for i in 0..*count
        c[i] = a[i] * b[i]

### vdiv.bf16
syntax
    vdiv.bf16 %c %a %b %count padding

parameters
    %c: reg  
    %a: reg  
    %b: reg  
    %count: reg  
    padding: uint8

pseudo code
    bf16* a = reg[%a] * 4
    bf16* b = reg[%b] * 4
    bf16* c = reg[%c] * 4
    int32* count = reg[%count]

    for i in 0..*count
        c[i] = a[i] / b[i]

### add.int32
syntax
    add.int32 %a %b %i

parameters
    %a: reg  
    %b: reg  
    %i: int16

pseudo code
    int32* a = reg[%a]
    int32* b = reg[%a]
    int32 i = (int32)%i

    *a = *a + *b + i

### sub.int32
syntax
    sub.int32 %a %b %i

parameters
    %a: reg  
    %b: reg  
    %i: int16

pseudo code
    int32* a = reg[%a]
    int32* b = reg[%a]
    int32 i = (int32)%i

    *a = *a - *b - i

### ifz
syntax
    ifz %reg %jump padding

parameters
    %reg: reg  
    padding: reg  
    %jump: int16

pseudo code
    uint32 condition = reg[%reg]
    int32 jump = (int32)%jump

    if condition == 0:
        ip += jump

### ifeq
syntax
    ifeq %a %b %jump

parameters
    %a: reg  
    %b: reg  
    %jump: int16

pseudo code
    uint32 a = reg[%a]
    uint32 b = reg[%b]
    int32 jump = (int32)%jump

    if a == b:
        ip += jump

### ifneq
syntax
    ifneq %a %b %jump

parameters
    %a: reg  
    %b: reg  
    %jump: int16

pseudo code
    uint32 a = reg[%a]
    uint32 b = reg[%b]
    int32 jump = (int32)%jump

    if a != b:
        ip += jump

### jmp
syntax
    jmp padding %jump

parameters
    padding: uint8
    %jump: int16

pseudo code
    ip += jump

### return
syntax
    return

pseudo code
    csr.running = 0
    CPU에 Interrupt를 보냄

## CONNX NPU Python implementation
[README.md](asm/README.md)

