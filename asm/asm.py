import sys
import struct


is_debug = True

class Instruction:
    instructions = []

    def get_instruction(tokens):
        for instruction in Instruction.instructions:
            if instruction.get_name() == tokens[0]:
                return instruction

        raise Exception(f'There is no such instruction for the code: {tokens}')

    def __init__(self, code, name, params):
        self.code = code
        self.name = name
        self.params = params.split(' ') if len(params) != 0 else []

    def get_code(self):
        return self.code

    def get_name(self):
        return self.name

    def parse_reg(self, token):
        if token == '%zero' or token == '%0':
            return 0
        elif token == '%a' or token == '%1':
            return 1
        elif token == '%b' or token == '%2':
            return 2
        elif token == '%c' or token == '%3':
            return 3
        elif token == '%d' or token == '%4':
            return 4
        elif token == '%e' or token == '%5':
            return 5
        elif token == '%f' or token == '%6':
            return 6
        elif token == '%g' or token == '%7':
            return 7
        elif token == '%ip' or token == '%14':
            return 14
        elif token == '%csr' or token == '%15':
            return 15
        else:
            raise Exception(f'Illegal register name: {token}')

    def parse_uint(self, bits, token):
        maximum = 2 ** bits
        value = int(token, 0)
        if value < 0 or value > maximum:
            raise Exception(f'Out of integer bounds: {token}, expected: 0 ~ 2^{bits}')
        else:
            return value

    def parse_int(self, bits, token):
        maximum = 2 ** bits / 2 - 1
        minimum = -2 ** bits / 2
        value = int(token, 0)
        if value < minimum or value > maximum:
            raise Exception(f'Out of integer bounds: {token}, expected: -2^{bits} / 2 ~ 2^{bits} / 2 - 1')
        else:
            return value

    def compile(self, target, target_file, tokens):
        # Add padding to tokens
        for i, kind in enumerate(self.params):
            if kind[0] == 'p':
                tokens.insert(i, 0)

        opcode = 0
        offset = 32

        if is_debug:
            print(f'{self.name}', end=' ')

        def write(token, value, bits):
            nonlocal opcode
            nonlocal offset

            opcode |= (value & (2 ** bits - 1)) << (offset - bits)
            offset -= bits

            if is_debug:
                print(f'{token}', end=' ')

            if offset < 0:
                raise Exception('opcode overflow: {offset}')

        # Check parameter counts
        if len(self.params) != len(tokens):
            raise Exception(f'Illegal tokens: {tokens}, expected: {self.params}')

        # Write opcode
        write(f'{self.code:02x}', self.code, 8)

        # Write params
        for kind, token in zip(self.params, tokens):
            if kind == 'r':
                value = self.parse_reg(token)
                write(token, value, 4)
            elif kind == 'u8':
                value = self.parse_uint(8, token)
                write(token, value, 8)
            elif kind == 'u16':
                value = self.parse_uint(16, token)
                write(token, value, 16)
            elif kind == 'u20':
                value = self.parse_uint(20, token)
                write(token, value, 20)
            elif kind == 'i8':
                value = self.parse_int(8, token)
                write(token, value, 8)
            elif kind == 'i16':
                value = self.parse_int(16, token)
                write(token, value, 16)
            elif kind == 'i20':
                value = self.parse_int(20, token)
                write(token, value, 20)
            elif kind == 'p8':
                write(token, 0, 8)
            elif kind == 'p16':
                write(token, 0, 16)
            elif kind == 'p20':
                write(token, 0, 20)
            else:
                raise Exception(f'Illegal parameter type: {kind}')

        if is_debug:
            print(f'  # {opcode:08x}')

        target_file.write(struct.pack('I', opcode))


Instruction.instructions = [
    Instruction(0x00, 'nop', ''),
    Instruction(0x01, 'set', 'r u20'),
    Instruction(0x02, 'seti', 'r u20'),
    Instruction(0x03, 'seti_low', 'r p4 u16'),
    Instruction(0x04, 'seti_high', 'r p4 u16'),
    Instruction(0x05, 'get', 'r u20'),
    Instruction(0x06, 'mov', 'r r'),
    Instruction(0x07, 'load', 'r r r'),
    Instruction(0x08, 'store', 'r r r'),
    Instruction(0x09, 'vadd.bf16', 'r r r r'),
    Instruction(0x0a, 'vsub.bf16', 'r r r r'),
    Instruction(0x0b, 'vmul.bf16', 'r r r r'),
    Instruction(0x0c, 'vdiv.bf16', 'r r r r'),
    Instruction(0x0d, 'add.int32', 'r r i16'),
    Instruction(0x0e, 'sub.int32', 'r r i16'),
    Instruction(0x0f, 'ifz', 'r r i16'),
    Instruction(0x10, 'ifeq', 'r r i16'),
    Instruction(0x11, 'ifneq', 'r r i16'),
    Instruction(0x12, 'jmp', 'p8 i16'),
    Instruction(0xff, 'return', ''),
]


def compile(source, target):
    with open(source, 'r') as source_file:
        with open(target + '.bin', 'wb') as target_file:
            while True:
                line = source_file.readline();

                if not line:
                    break

                # Save script file
                if line.startswith('### script'):
                    with open(target + '.script', 'w') as script_file:
                        while True:
                            line = source_file.readline();

                            if not line:
                                break

                            if line.startswith('###'):
                                break

                            script_file.write(line)

                # Skip comment
                pos = line.find('#')
                if pos >= 0:
                    line = line[:pos]

                line = line.strip()

                # Skip empty line
                if len(line) == 0:
                    continue

                # tokenize
                tokens = line.split(' ')

                if len(tokens) < 1:
                    raise Exception(f'Illegal opcode: {line}')

                # asm -> opcode
                instruction = Instruction.get_instruction(tokens)
                instruction.compile(target, target_file, tokens[1:])

if len(sys.argv) < 4 or sys.argv[2] != '-o':
    print(f'Usage: python {sys.argv[0]} [input file] -o [target]')
    sys.exit(0)

compile(sys.argv[1], sys.argv[3])
