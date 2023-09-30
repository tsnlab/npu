# CONNX NPU Python implementation

## Install
 - poetry 1.6 is required

```bash
asm$ poetry install
```

## Assembler
```bash
asm$ poetry run python asm.py ../examples/add.asm -o add
```

 - add.bin - kernel
 - add.200000.data - data for main memory at 0x200000
 - add.200800.data - data for main memory at 0x200800
 - add.201000.data - data for main memory at 0x201000

## Interpreter
```bash
asm$ poetry run python interpreter.py add
```
