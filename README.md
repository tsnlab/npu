# CONNX NPU
## Data Structure

```C
load message {
    offset: uint64 // Main memory에서 kernel의 위치
    size: uint32  // kernel의 크기 (4 bytes로 정렬된 bytes 수)
    core_id: uint16 // kernel을 실행할 Core ID
    interrupt_id: uint16 // 실행 완료 시 보낼 인터럽트 번호
}

start message {
    core_id: uint16 // kernel을 실행할 Core ID
    interrupt_id: uint16 // 실행 완료 시 보낼 인터럽트 번호
}

kernel {
    opcodes[]: uint32 // opcode 목록
}

Interpreter {
    m: uint32  // reg1 main memory
    s: uint32  // reg2 sram memory
    a: uint32  // reg3 a
    b: uint32  // reg4 b
    c: uint32  // reg5 c
    d: uint32  // reg6 d
}
```

## Opcode
모든 opcode는 32bit이며 padding은 32bit를 맞추기 위한 0 값입니다.

### nop - no operator
syntax
    nop

### set\_low - register의 상위 16 bits를 설정함
syntax
    set_low %reg %value

parameters
    set_low: uint8 = 0x02
    %reg: uint8 // register 번호
    %value: uint16 // register의 low에 입력할 값

pseudo code
    reg[%reg] = (reg[%reg] & 0xff) | (%value << 16)

### set\_high - register의 상위 16 bits를 설정함
syntax
    set_high %reg %value

parameters
    set_high: uint8 = 0x02
    %reg: uint8 // register 번호
    %value: uint16 // register의 high에 입력할 값

pseudo code
    reg[%reg] = (reg[%reg] & 0xff) | (%value << 16)

### load - DRAM에서 SRAM으로 데이터를 저장함
syntax
    load %count padding

parameters
    load: uint8 = 0x03
    %count: uint16

pseudo code
    DRAM_address = Interpreter.m * 4
    SRAM_address = Interpreter.s * 4
    size = %count * 4

    memcpy(DRAM_address, SRAM_address, size)

### store - SRAM에서 DRAM으로 데이터를 저장함
syntax
    store %count padding

parameters
    store: uint8 = 0x04
    %count: uint16

pseudo code
    DRAM_address = Interpreter.m * 4
    SRAM_address = Interpreter.s * 4
    size = %count * 4

    memcpy(SRAM_address, DRAM_address, size)

### add.f32
syntax
    add.f32 %count padding

parameters
    add.f32: uint8 = 0x05
    %count: uint16

pseudo code
    float32* A = Interpreter.s * 4 + Interpreter.a * 4
    float32* B = Interpreter.s * 4 + Interpreter.b * 4
    float32* C = Interpreter.s * 4 + Interpreter.c * 4

    for i in 0..%count
        C[i] = A[i] + B[i]

### sub.f32
syntax
    sub.f32 %count padding

parameters
    sub.f32: uint8 = 0x06
    %count: uint16

pseudo code
    float32* A = Interpreter.s * 4 + Interpreter.a * 4
    float32* B = Interpreter.s * 4 + Interpreter.b * 4
    float32* C = Interpreter.s * 4 + Interpreter.c * 4

    for i in 0..%count
        C[i] = A[i] - B[i]

### mul.f32
syntax
    mul.f32 %count padding

parameters
    mul.f32: uint8 = 0x07
    %count: uint16

pseudo code
    float32* A = Interpreter.s * 4 + Interpreter.a * 4
    float32* B = Interpreter.s * 4 + Interpreter.b * 4
    float32* C = Interpreter.s * 4 + Interpreter.c * 4

    for i in 0..%count
        C[i] = A[i] * B[i]

### div.f32
syntax
    div.f32 %count padding

parameters
    div.f32: uint8 = 0x08
    %count: uint16

pseudo code
    float32* A = Interpreter.s * 4 + Interpreter.a * 4
    float32* B = Interpreter.s * 4 + Interpreter.b * 4
    float32* C = Interpreter.s * 4 + Interpreter.c * 4

    for i in 0..%count
        C[i] = A[i] / B[i]

### return
syntax
    return

parameters
    return: uint8 = 0xff

pseudo code
    CPU에 Interrupt를 보냄

