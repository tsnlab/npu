# CONNX NPU Python implementation

## Install
 - poetry 1.6 is required

```bash
asm$ poetry install
```

## Assembler
```bash
asm$ poetry run python asm.py ../examples/vadd.asm
```

Outputs
 - vadd.bin : kernel
 - vadd.script : script

## Interpreter
```bash
asm$ poetry run python sim.py vadd
```
