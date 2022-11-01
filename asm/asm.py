import sys
import struct


class Opcode:
    opcodes = []

    def get_opcode(tokens):
        for opcode in Opcode.opcodes:
            if opcode.get_name() == tokens[0]:
                return opcode

        raise Exception(f'There is no such opcode for the code: {tokens}')

    def __init__(self, code, name, params, compiler=None):
        self.code = code
        self.name = name
        self.params = params
        self.compiler = compiler

    def get_code(self):
        return self.code

    def get_name(self):
        return self.name

    def get_reg(self, token):
        if token == '%a' or token == '%A' or token == '%1':
            return 1
        elif token == '%b' or token == '%B' or token == '%2':
            return 2
        elif token == '%c' or token == '%C' or token == '%3':
            return 3
        elif token == '%d' or token == '%D' or token == '%4':
            return 4
        else:
            raise Exception(f'Illegal register name: {token}')

    def get_value(self, bits, token):
        maximum = 2 ** bits
        value = int(token, 0)
        if value < 0 or value > maximum:
            raise Exception(f'Out of integer bounds: {token}, expected: 0 ~ 2^{bits}')
        else:
            return value

    def compile(self, target, target_file, tokens):
        if self.compiler is not None:
            return self.compiler(target, target_file, tokens)

        packed = 0

        # Check parameter counts
        if len(self.params) != len(tokens):
            raise Exception(f'Illegal tokens: {tokens}, expected: {self.params}')

        # Write opcode
        target_file.write(struct.pack('B', self.code))
        packed += 1

        # Write params
        for kind, token in zip(self.params, tokens):
            if kind == 'r':
                reg = self.get_reg(token)
                target_file.write(struct.pack('B', reg))
                packed += 1
            elif kind == 'b':
                value = elf.get_value(8, token)
                target_file.write(struct.pack('B', value))
                packed += 1
            elif kind == 's':
                value = self.get_value(16, token)
                target_file.write(struct.pack('H', value))
                packed += 2
            else:
                raise Exception(f'Illegal parameter type: {kind}')

        while packed % 4 != 0:
            target_file.write(struct.pack('B', 0))
            packed += 1


def compile_data_f32(target, target_file, tokens):
    addr = int(tokens[0], 0)

    data = eval(' '.join(tokens[1:]))
    if not isinstance(data, list) and not isinstance(data, tuple):
        raise Exception(f'Illegal data type: {type(data)}')

    with open(target + '.' + hex(addr)[2:] + '.data', 'wb') as f:
        for i in range(len(data)):
            f.write(struct.pack('f', data[i]))

    
Opcode.opcodes = [
    Opcode(0, 'nop', ''),
    Opcode(1, 'set_high', 'rs'),
    Opcode(2, 'set_low', 'rs'),
    Opcode(3, 'load', 's'),
    Opcode(4, 'store', 's'),
    Opcode(5, 'add.f32', 's'),
    Opcode(6, 'sub.f32', 's'),
    Opcode(7, 'mul.f32', 's'),
    Opcode(8, 'div.f32', 's'),
    Opcode(9, 'return', ''),
    Opcode(255, 'data.f32', 'd', compiler=compile_data_f32)
]



def compile(source, target):
    with open(source, 'r') as source_file:
        with open(target + '.bin', 'wb') as target_file:
            while True:
                line = source_file.readline();

                if not line:
                    break

                pos = line.find('#')
                if pos >= 0:
                    line = line[:pos]

                line = line.strip()

                if len(line) == 0:
                    continue

                tokens = line.split(' ')

                if len(tokens) < 1:
                    raise Exception(f'Illegal opcode: {line}')

                opcode = Opcode.get_opcode(tokens)

                opcode.compile(target, target_file, tokens[1:])


if len(sys.argv) < 4 or sys.argv[2] != '-o':
    print(f'Usage: python {sys.argv[0]} [input file] -o [target]')
    sys.exit(0)

compile(sys.argv[1], sys.argv[3])
