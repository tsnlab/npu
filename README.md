# CONNX NPU

## Generating Project
1. Xiliinx Vivado 실행
2. 메뉴바의 `Tools > Run Tcl Script...` 선택
3. npu.tcl 선택 후 `open`
4. `npu.tcl` 파일이 열리고 기다리면 `npu.xsa` 출력됨

## Caution
Tcl run을 하면 Vivado를 실행시킨 현재 디렉토리의 상위 디렉토리에서 프로젝트가 생성됩니다.
Tcl 파일이 있는 위치에서 Vivado 실행을 해야 `File or Directory does not exist Error`가 발생하지 않습니다.


## Data Structure

```C
load_and_start message {
    offset: uint32 // Main memory에서 kernel의 위치
    size: uint32  // kernel의 크기 (4 bytes로 정렬된 bytes 수)
    core_id: uint16 // kernel을 실행할 Core ID
    status: 4 bits // core의 상태 0: idle, 1: busy
}

kernel {
    opcodes[]: uint32 // opcode 목록
}

Interpreter {
    a: uint32  // reg1 a
    b: uint32  // reg2 b
    c: uint32  // reg3 c
    d: uint32  // reg4 d
}
```

## Opcode
모든 opcode는 32bit 단위로 zero padding 되어있음
모든 opcode는 little endian으로 표기됨

### nop - no operator
syntax
    nop

parameters
    nop: uint8 = 0x00

### set\_high - register의 상위 16 bits를 설정함
syntax
    set_high %reg %value

parameters
    set_high: uint8 = 0x01
    %reg: uint8 // register 번호
    %value: uint16 // register의 high에 입력할 값

pseudo code
    reg[%reg] = (%value << 16) | (reg[%reg] & 0xffff)

### set\_low - register의 상위 16 bits를 설정함
syntax
    set_low %reg %value

parameters
    set_low: uint8 = 0x02
    %reg: uint8 // register 번호
    %value: uint16 // register의 low에 입력할 값

pseudo code
    reg[%reg] = (reg[%reg] & 0xffff0000) | (%value & 0xffff)

### load - Host memory로부터 local memory로 데이터 복사
syntax
    load %count

parameters
    load: uint8 = 0x03
    %count: uint16

pseudo code
    size = %count * 4

    memcpy(B, A, size)

### store - local memory로부터 host memory로 데이터 복사
syntax
    store %count

parameters
    store: uint8 = 0x04
    %count: uint16

pseudo code
    size = %count * 4

    memcpy(A, B, size)

### add.f32
syntax
    add.f32 %count

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
    sub.f32 %count

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
    mul.f32 %count

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
    div.f32 %count

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
    return: uint8 = 0x09

pseudo code
    CPU에 Interrupt를 보냄

## CONNX NPU Python implementation
[README.md](asm/README.md)
