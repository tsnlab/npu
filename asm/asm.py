import sys


class Opcode:
    def __init__(self, code, name, params):
        self.code = code
        self.name = name
        self.params = params

    def get_code(self):
        return self.code

    def get_name(self):
        return self.name

    def get_param_count(self):
        return len(self.params)

    def get_reg(self, token):
        if len(token) != 2:
            raise Exception('Illegal register name: {token}')

        if token[0] != '%':
            raise Exception('Illegal register name: {token}')

        if token[1] == 'm' or token[1] == '1':
            return 1
        elif token[1] == 's' or token[1] == '2':
            return 2
        elif token[1] == 'a' or token[1] == '3':
            return 3
        elif token[1] == 'b' or token[1] == '4':
            return 4
        elif token[1] == 'c' or token[1] == '5':
            return 5
        elif token[1] == 'd' or token[1] == '6':
            return 6
        else:
            raise Exception('Illegal register name: {token}')

    def get_value(self, bits, token):
        maximum = 2 ** bits
        value = int(token, 0)
        if value < 0 or value > maximum:
            raise Exception(f'Out of integer bounds: {token}, expected: 0 ~ 2^{bits}')
        else:
            return value

    def compile(self, tokens):
        if len(self.params) != len(tokens):
            raise Exception(f'Illegal parameters: {tokens}, exptected: {self.params}')

        print(self.code, self.name, end=' ')
        for kind, token in zip(self.params, tokens):
            if kind == 'r':
                reg = self.get_reg(token)
                print(f'%{reg}', end=' ')
            elif kind == 'b':
                value = elf.get_value(8, token)
                print(f'{value}', end=' ')
            elif kind == 's':
                value = self.get_value(16, token)
                print(f'{value}', end=' ')
            else:
                raise Exception(f'Illegal parameter type: {kind}')

        print()

opcodes = [
    Opcode(0, 'nop', ''),
    Opcode(1, 'set_low', 'rs'),
    Opcode(2, 'set_high', 'rs'),
    Opcode(3, 'load', 's'),
    Opcode(4, 'store', 's'),
    Opcode(5, 'add.f32', 's'),
    Opcode(6, 'sub.f32', 's'),
    Opcode(7, 'mul.f32', 's'),
    Opcode(8, 'div.f32', 's'),
    Opcode(9, 'return', '')
]


def get_opcode(tokens):
    for opcode in opcodes:
        if opcode.get_name() == tokens[0] and len(tokens) - 1 == opcode.get_param_count():
            return opcode

    raise Exception(f'There is no such opcode for the code: {tokens}')


def compile(path):
    with open(path, 'r') as f:
        while True:
            line = f.readline();

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

            opcode = get_opcode(tokens)

            opcode.compile(tokens[1:])


if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} [input file]')
    sys.exit(0)

compile(sys.argv[1])
