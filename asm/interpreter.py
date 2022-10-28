import sys
import glob
import struct


class Interpreter:
    def __init__(self, mem_size=512 * 1024):
        self.data = {}
        self.updated = {}
        self.reg = [0, 0, 0, 0, 0]
        self.target = None
        self.memory = bytearray(mem_size)

    def load(self, target):
        self.target = target

        data_paths = glob.glob(f'{target}.*.data')

        for data_path in data_paths:
            addr = int(data_path[len(target) + 1:-5], 16)

            with open(data_path, 'rb') as f:
                self.data[addr] = bytearray(f.read())

    def get_data(self, offset, size):
        for addr, data in self.data.items():
            if addr >= offset and addr + len(data) <= offset + size:
                return addr, data

        raise Exception(f'There is no such data: {hex(offset)} ~ {hex(offset + size)}')

    def exec(self, op):
        if op[0] == 0x00:  # nop
            return True
        elif op[0] == 0x01:  # set_high
            reg = op[1]
            value = struct.unpack('H', op[2:])[0]

            self.reg[reg] = value << 16 | (self.reg[reg] & 0xffff)

            return True
        elif op[0] == 0x02:  # set_low
            reg = op[1]
            value = struct.unpack('H', op[2:])[0]

            self.reg[reg] = (self.reg[reg] & 0xffff0000) | (value & 0xffff)

            return True
        elif op[0] == 0x03:  # load
            count = struct.unpack('H', op[1:3])[0]
            size = count * 4

            A = self.reg[1]
            B = self.reg[2]

            addr, data = self.get_data(A, size)
            delta = A - addr

            self.memory[B:B + size] = data[delta:delta + size]

            return True
        elif op[0] == 0x04:  # store
            count = struct.unpack('H', op[1:3])[0]
            size = count * 4

            A = self.reg[1]
            B = self.reg[2]

            addr, data = self.get_data(A, size)

            delta = A - addr
            data[delta: delta + size] = self.memory[B:B + size]

            self.updated[addr] = True

            return True
        elif op[0] == 0x05:  # add.f32
            count = struct.unpack('H', op[1:3])[0]

            A = self.reg[1]
            B = self.reg[2]
            C = self.reg[3]

            for i in range(count):
                A_i = struct.unpack('f', self.memory[A + i * 4: A + i * 4 + 4])[0]
                B_i = struct.unpack('f', self.memory[B + i * 4: B + i * 4 + 4])[0]
                C_i = A_i + B_i
                self.memory[C + i * 4: C + i * 4 + 4] = struct.pack('f', C_i)

            return True
        elif op[0] == 0x06:  # sub.f32
            count = struct.unpack('H', op[1:3])[0]

            A = self.reg[1]
            B = self.reg[2]
            C = self.reg[3]

            for i in range(count):
                A_i = struct.unpack('f', self.memory[A + i * 4: A + i * 4 + 4])[0]
                B_i = struct.unpack('f', self.memory[B + i * 4: B + i * 4 + 4])[0]
                C_i = A_i - B_i
                self.memory[C + i * 4: C + i * 4 + 4] = struct.pack('f', C_i)

            return True
        elif op[0] == 0x07:  # mul.f32
            count = struct.unpack('H', op[1:3])[0]

            A = self.reg[1]
            B = self.reg[2]
            C = self.reg[3]

            for i in range(count):
                A_i = struct.unpack('f', self.memory[A + i * 4: A + i * 4 + 4])[0]
                B_i = struct.unpack('f', self.memory[B + i * 4: B + i * 4 + 4])[0]
                C_i = A_i * B_i
                self.memory[C + i * 4: C + i * 4 + 4] = struct.pack('f', C_i)

            return True
        elif op[0] == 0x08:  # div.f32
            count = struct.unpack('H', op[1:3])[0]

            A = self.reg[1]
            B = self.reg[2]
            C = self.reg[3]

            for i in range(count):
                A_i = struct.unpack('f', self.memory[A + i * 4: A + i * 4 + 4])[0]
                B_i = struct.unpack('f', self.memory[B + i * 4: B + i * 4 + 4])[0]
                C_i = A_i / B_i
                self.memory[C + i * 4: C + i * 4 + 4] = struct.pack('f', C_i)

            return True
        elif op[0] == 0x09:
            return False
        else:
            raise Exception(f'Not supported opcode: 0x{op[0]}')
            

    def run(self):
        with open(self.target + '.bin', 'rb') as f:
            while True:
                op = f.read(4)

                if len(op) == 0:
                    raise Exception('There is not return')

                if not self.exec(op):
                    break


        print('# Dump register')

        for i in range(len(self.reg)):
            print(f'%{i} = {hex(self.reg[i])}')


        print('# Dump data')

        for addr, _ in self.updated.items():
            data = self.data[addr]

            for i in range(len(data) // 4):
                value = struct.unpack('f', data[i * 4:i * 4 + 4])[0]
                print(f'[{i}] = {value}')


if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} [target]')
    sys.exit(0)


inter = Interpreter()
inter.load(sys.argv[1])
inter.run()
