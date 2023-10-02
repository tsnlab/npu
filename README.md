# CONNX NPU

## Generating Project
1. Xiliinx Vivado 실행
2. 메뉴바의 `Tools > Run Tcl Script...` 선택
3. npu.tcl 선택 후 `open`
4. `npu.tcl` 파일이 열리고 기다리면 `npu.xsa` 출력됨

## Caution
Tcl run을 하면 Vivado를 실행시킨 현재 디렉토리의 상위 디렉토리에서 프로젝트가 생성됩니다.
Tcl 파일이 있는 위치에서 Vivado 실행을 해야 `File or Directory does not exist Error`가 발생하지 않습니다.


## RISC-V instruction extensions
 * store NPU ID, NPU address, Host address, size
 * load NPU ID, Host address, NPU address, size
 * exec NPU ID
 * set NPU ID, reg ID, value
 * get NPU ID, reg ID

## NPU Registers
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
 1\~31. reserved
 32. error - 0: no error, 1: error

## NPU Core instructions
모든 instruction은 32bit 단위로 zero padding 되어있음.
모든 instruction은 little endian으로 표기됨.
아래는 NPU Core에서 지원하는 instruction 목록이자 opcode임.
모든 opcode는 uint8 크기임.

 00. nop
 01. set
 02. seti
 03. seti\_low
 04. seti\_high
 05. get
 06. load
 07. store
 08. vadd.bf16
 09. vsub.bf16
 0a. vmul.bf16
 0b. vdiv.bf16
 0c. add.int32
 0d. sub.int32
 0e. ifz
 0f. ifeq
 10. ifneq
 11. jmp
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
    %reg: uint4  // register 번호
    %mem: uint20 // align된 local memory 주소

pseudo code
    reg[%reg] = mem[%mem * 4]

### seti - register의 값을 설정함
syntax
    seti %reg %value

parameters
    %reg: uint4  // register 번호
    %value: uint20 // register에 입력될 값

pseudo code
    reg[%reg] = (0x000 << 24) | (%value & 0xfffff)

### seti\_low - register의 값을 설정함
syntax
    seti_low %reg %value padding

parameters
    %reg: uint4    // register 번호
    %value: uint16 // register에 입력될 값
    padding: uint8

pseudo code
    reg[%reg] = (reg[%reg] & 0xffff0000) | (%value & 0xffff)

### seti\_high - register의 값을 설정함
syntax
    seti_high %reg %value padding

parameters
    %reg: uint4    // register 번호
    %value: uint16 // register에 입력될 값
    padding: uint8

pseudo code
    reg[%reg] = (reg[%reg] & 0xffff) | ((%value & 0xffff) << 16)

### get - register의 값을 local memory로 복사함
syntax
    get %reg %mem

parameters
    %reg: uint4  // register 번호
    %mem: uint20 // align된 local memory 주소

pseudo code
    mem[%dest * 4] = reg[%reg]

### load - Host memory로부터 local memory로 데이터 복사
syntax
    load %dest %src %count

parameters
    %dest: uint4
    %src: uint4
    %count: uint4

pseudo code
    dest = reg[%dest] * 4  // Local memory는 4 bytes 단위로 align 되어있다고 가정
    src = reg[%src] * 128  // Host memory는 128 bytes 단위로 align 되어있다고 가정
    size = reg[%count] * 4  // 4 bytes 단위로 roundup 되어있다고 가정

    memcpy(dest, src, size)

### store - local memory로부터 host memory로 데이터 복사
syntax
    store %dest %src %count

parameters
    %dest: uint4
    %src: uint4
    %count: uint4

pseudo code
    dest = reg[%dest] * 128 // Host memory는 128 bytes 단위로 align 되어있다고 가정
    src = reg[%src] * 4     // Local memory는 4 bytes 단위로 align 되어있다고 가정
    size = reg[%count] * 4  // 4 bytes 단위로 roundup 되어있다고 가정

    memcpy(dest, src, size)

### vadd.bf16
syntax
    vadd.bf16 %c %a %b %count padding

parameters
    %c: uint4
    %a: uint4
    %b: uint4
    %count: uint4
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
    %c: uint4
    %a: uint4
    %b: uint4
    %count: uint4
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
    %c: uint4
    %a: uint4
    %b: uint4
    %count: uint4
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
    %c: uint4
    %a: uint4
    %b: uint4
    %count: uint4
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
    %a: uint4
    %b: uint4
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
    %a: uint4
    %b: uint4
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
    %reg: uint4
    padding: uint4
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
    %a: uint4
    %b: uint4
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
    %a: uint4
    %b: uint4
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
