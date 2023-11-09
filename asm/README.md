# CONNX NPU Python implementation

## Install
 - poetry 1.6 is required

```bash
asm$ curl -sSL https://install.python-poetry.org | python3 -
asm$ pip install jax
asm$ poetry add jax
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
