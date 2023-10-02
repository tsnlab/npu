import sys
import glob
import struct
import threading
import numpy as np

is_debug = True

REG_ZERO = 0
REG_A = 1
REG_B = 2
REG_C = 3
REG_D = 4
REG_E = 5
REG_F = 6
REG_G = 7
REG_IP = 14
REG_CSR = 15

def IS_READABLE_REG(reg):
    return reg >= REG_ZERO and reg <= REG_G or reg >= REG_IP and reg <= REG_CSR

def IS_WRITABLE_REG(reg):
    return reg >= REG_A and reg <= REG_G

STATUS_RUNNING = 1 << 0
STATUS_ERROR = 1 << 31

def float32_to_bf16(f32_value):
    """
    f32_bytes = struct.pack('f', value)
    f32_int = struct.unpack('I', f32_bytes)[0]

    sign_bit = (f32_int >> 31) & 0x01
    exponent_bits = (f32_int >> 23) & 0xff
    mantissa_bits = f32_int & 0x7fffff

    bf16_exponent = exponent_bits - 112
    bf16_mantissa = mantissa_bits >> 13

    bf16_value = (sign_bit << 15) | (bf16_exponent << 10) | bf16_mantissa

    print(f'{bf16_value:04x}', sign_bit, bf16_exponent, bf16_mantissa)
    return struct.pack('H', bf16_value)
    """
    """
    bf16_value = np.float32(f32_value).view(np.uint32)
    bf16_value >>= 16  # Right-shift to get the BF16 representation
    bf16_value = np.uint16(bf16_value)
    return struct.pack('H', u16_value)
    """
    f32_bytes = struct.pack('f', f32_value)
    return f32_bytes[2:]

def bf16_to_float32(bf16_buf):
    """
    bf16_value = struct.unpack('H', bf16_buf)[0]

    sign_bit = (bf16_value >> 15) & 0x01
    exponent_bits = (bf16_value >> 10) & 0x1f
    matissa_bits = bf16_value & 0x03ff

    float32_exponent_bits = exponent_bits + 112
    float32_mantissa_bits = mantissa_bits << 13

    float32_bits = (sign_bit << 31) | (float32_exponent_bits << 23) | float32_mantissa_bits
    float32_bytes = struct.pack('I', float32_bits)

    return struct.unpack('f', float32_bytes)[0]
    """
    """
    bf16_value = np.float32(f32_value).view(np.uint32)
    bf16_value >>= 16  # Right-shift to get the BF16 representation
    bf16_value = np.uint16(bf16_value)

    bf16_value = struct.unpack('H', bf16_buf)[0]
    bfloat16_as_uint16 = np.uint16(bf16_value)
    f32_as_uint32 = (bfloat16_as_uint16 << 16) | (bfloat16_as_uint16 & 0x7fff)
    return np.float32(f32_as_uint32)
    """
    f32_bytes = bytes([0, 0, bf16_buf[0], bf16_buf[1]])
    return struct.unpack('f', f32_bytes)[0]

def _dump_bf16(data):
    count = 1
    i = 0
    while i < len(data):
        value = bf16_to_float32(data[i:i + 2])

        print(value, end=' ')
        if count % 8 == 0:
            print()

        i += 2
        count += 1

class Host:
    def __init__(self, mem_size=512 * 1024, npus=4, out='stdout'):
        self.kernel = None
        self.data = {}
        self.updated = {}
        self.npus = []
        self.script = None

        for i in range(npus):
            self.npus.append(NPU(i, mem_size, out=out))

    def init(self, target):
        # Read kernel
        with open(f'{target}.bin', 'rb') as f:
            self.kernel = bytearray(f.read())

        # Read data
        data_paths = glob.glob(f'{target}.*.data')

        for data_path in data_paths:
            addr = int(data_path[len(target) + 1:-5], 16)

            with open(data_path, 'rb') as f:
                self.data[addr] = bytearray(f.read())

        # Read init script
        with open(f'{target}.script', 'r') as f:
            self.script = f.read()

        # Run init script
        exec(self.script + '\ninit(host)', None, { 'host': self })

    def get_data(self, offset, size):
        for addr, data in self.data.items():
            if addr >= offset and addr + len(data) <= offset + size:
                return addr, data

        raise Exception(f'There is no such data: {offset:08x} ~ {offset + size:08x}')

    def run(self):
        # Run run script
        exec(self.script + '\nrun(host)', None, { 'host': self })

        for npu in self.npus:
            if npu.is_alive():
                npu.join()

        # Run finalizescript
        exec(self.script + '\nfinalize(host)', None, { 'host': self })

    def store(self, npu_id, npu_address, host_address, size=None):
        if npu_id < 0 or npu_id >= len(self.npus):
            raise Exception(f'Illegal NPU id: {npu_id}')

        if host_address == 0x00:
            data = self.kernel
            size = len(self.kernel)
        else:
            addr, data = self.get_data(host_address, size)
            data = data[host_address - addr: host_address - addr + size]

        npu = self.npus[npu_id]
        npu.memory[npu_address:npu_address + size] = data

    def load(self, npu_id, host_address, npu_address):
        pass

    def exec(self, npu_id):
        self.npus[npu_id].start()

    def set(self, npu_id, reg_id, value):
        pass

    def get(self, npu_id, reg_id):
        pass

    def bf16(self, values):
        if not isinstance(values, list) and not isinstance(values, tuple):
            raise Exception(f'Illegal values type: {type(values)}')

        buf = bytearray()

        for value in values:
            buf.extend(float32_to_bf16(value))

        return buf

    def zeros(self, size):
        buf = bytearray()

        while size > 0:
            buf.extend(struct.pack('I', 0))
            size -= 4

        return buf

    def dump_bf16(self, addr):
        data = self.data[addr]
        _dump_bf16(data)

    def dump(self, addr):
        data = self.data[addr]

        count = 1
        for i in range(len(data)):
            b = data[i]

            print(f'{b:02x}', end=' ')

            if count % 16 == 0:
                print()
            elif count % 8 == 0:
                print(end=' ')

            count += 1


class NPU(threading.Thread):
    def __init__(self, id, mem_size=512 * 1024, out='stdout'):
        super().__init__()

        self.id = id
        self.reg = [0] * 16
        self.memory = bytearray(mem_size)
        self.out = out

    def parse(self, opcode, params):
        result = []

        offset = 32 - 8

        def read(bits):
            nonlocal opcode
            nonlocal offset

            offset -= bits

            return (opcode >> offset) & (2 ** bits - 1)

        for kind in params:
            if kind == 'r':
                result.append(read(4))
            elif kind == 'u8':
                result.append(read(8))
            elif kind == 'u16':
                result.append(read(16))
            elif kind == 'u20':
                result.append(read(20))
            elif kind == 'i8':
                result.append(read(8))
            elif kind == 'i16':
                result.append(read(16))
            elif kind == 'i20':
                result.append(read(20))
            elif kind == 'p8':
                pass
            elif kind == 'p16':
                pass
            elif kind == 'p20':
                pass
            else:
                raise Exception(f'Illegal parameter type: {kind}')

        return result

    def is_inbound_memory(self, addr, size):
        return addr >= 0 and addr + size <= len(self.memory)

    def exec(self, opcode):
        if is_debug:
            print(f'NPU {self.id} [{self.reg[REG_IP]:>3}] {opcode:08x}')

        instruction = opcode >> 24

        if instruction == 0x00:  # nop
            pass
        elif instruction == 0x01:  # set
            dest, src = self.parse(opcode, ['r', 'u20'])

            if not IS_WRITABLE_REG(dest) or not self.is_inbound_memory(src):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.reg[dest] = struct.unpack('I', self.memory[src * 4:src * 4 + 4])[0]
        elif instruction == 0x02:  # seti
            dest, value = self.parse(opcode, ['r', 'u20'])

            if not IS_WRITABLE_REG(dest):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.reg[dest] = value
        elif instruction == 0x03:  # seti_low
            dest, value = self.parse(opcode, ['r', 'p4', 'u16'])

            if not IS_WRITABLE_REG(dest):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.reg[dest] = (self.reg[dest] & 0xffff0000) | (value & 0xffff)
        elif instruction == 0x04:  # seti_high
            dest, value = self.parse(opcode, ['r', 'p4', 'u16'])

            if not IS_WRITABLE_REG(dest):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.reg[dest] = ((value & 0xffff) << 16) | (self.reg[dest] & 0xffff)
        elif instruction == 0x05:  # get
            src, dest = self.parse(opcode, ['r', 'u20'])

            if not IS_READABLE_REG(src):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.memory[dest:dest + 4] = struct.pack('I', self.reg[src])
        elif instruction == 0x06:  # mov
            dest, src = self.parse(opcode, ['r', 'r'])

            if not IS_WRITABLE_REG(dest) or not IS_READABLE_REG(src):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            self.reg[dest] = self.reg[src]
        elif instruction == 0x07:  # load
            dest, src, count = self.parse(opcode, ['r', 'r', 'r'])

            if not IS_READABLE_REG(dest) or not IS_READABLE_REG(src) or not IS_READABLE_REG(count):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            dest_addr = self.reg[dest] * 4
            src_addr = self.reg[src] * 128
            size = self.reg[count] * 4

            if not self.is_inbound_memory(dest_addr, size):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                if is_debug:
                    print(f'NPU{self.id}: Out of memory: 0x{dest_addr:08x} ~ +{size} bytes')
                return

            addr, data = host.get_data(src_addr, size)
            self.memory[dest:dest + size] = data[src_addr - addr: src_addr - addr + size]
        elif instruction == 0x08:  # store
            dest, src, count = self.parse(opcode, ['r', 'r', 'r'])

            if not IS_READABLE_REG(dest) or not IS_READABLE_REG(src) or not IS_READABLE_REG(count):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return

            dest_addr = self.reg[dest] * 128
            src_addr = self.reg[src] * 4
            size = self.reg[count] * 4

            if not self.is_inbound_memory(src_addr, size):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                if is_debug:
                    print(f'NPU{self.id}: Out of memory: 0x{src_addr:08x} ~ +{size} bytes')
                return

            addr, data = host.get_data(dest_addr, size)
            data[dest_addr - addr:dest_addr - addr + size] = self.memory[src_addr:src_addr + size]
        elif instruction == 0x09:  # vadd.bf16
            c, a, b, count = self.parse(opcode, ['r', 'r', 'r', 'r'])

            if not IS_READABLE_REG(c) or not IS_READABLE_REG(a) or \
               not IS_READABLE_REG(b) or not IS_READABLE_REG(count):
                self.reg[REG_CSR] |= STATUS_ERROR
                self.reg[REG_CSR] &= ~STATUS_RUNNING
                return
            
            c_addr = self.reg[c] * 4
            a_addr = self.reg[a] * 4
            b_addr = self.reg[b] * 4
            count = self.reg[count]

            for i in range(count):
                a = bf16_to_float32(self.memory[a_addr:a_addr + 2])
                b = bf16_to_float32(self.memory[b_addr:b_addr + 2])
                c_bytes = float32_to_bf16(a + b)
                self.memory[c_addr:c_addr + 2] = c_bytes
                #print('[', i, ']', a, '(', a_addr, ')', '+', b, '(', b_addr, ')', '=', a+b)

                a_addr += 2
                b_addr += 2
                c_addr += 2

        elif instruction == 0x0a:  # vsub.bf16
            pass
        elif instruction == 0x0b:  # vmul.bf16
            pass
        elif instruction == 0x0c:  # vdiv.bf16
            pass
        elif instruction == 0x0d:  # add.int32
            pass
        elif instruction == 0x0e:  # sub.int32
            pass
        elif instruction == 0x0f:  # ifz
            pass
        elif instruction == 0x10:  # ifeq
            pass
        elif instruction == 0x11:  # ifneq
            pass
        elif instruction == 0x12:  # jmp
            pass
        elif instruction == 0xff:  # return
            self.reg[REG_CSR] &= ~STATUS_RUNNING

        return

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
        elif op[0] == 0xff:
            self.reg[REG_CSR] ^= STATUS_RUNNING
        else:
            raise Exception(f'Not supported opcode: 0x{op[0]}')

    def run(self):
        for i in range(len(self.reg)):
            self.reg[i] = 0

        self.reg[REG_CSR] |= STATUS_RUNNING

        while (self.reg[REG_CSR] & STATUS_RUNNING) != 0:
            ip = self.reg[REG_IP] * 4
            op = self.memory[ip:ip + 4]
            op = struct.unpack('I', op)[0]

            self.exec(op)

            self.reg[REG_IP] += 1

    def dump_regs(self):
        print(f'zero: {self.reg[REG_ZERO]:08x} {self.reg[REG_ZERO]}')
        print(f'a   : {self.reg[REG_A]:08x} {self.reg[REG_A]}')
        print(f'b   : {self.reg[REG_B]:08x} {self.reg[REG_B]}')
        print(f'c   : {self.reg[REG_C]:08x} {self.reg[REG_C]}')
        print(f'd   : {self.reg[REG_D]:08x} {self.reg[REG_D]}')
        print(f'e   : {self.reg[REG_E]:08x} {self.reg[REG_E]}')
        print(f'f   : {self.reg[REG_F]:08x} {self.reg[REG_F]}')
        print(f'g   : {self.reg[REG_G]:08x} {self.reg[REG_G]}')
        print(f'ip  : {self.reg[REG_IP]:08x} {self.reg[REG_IP]}')
        print(f'csr : {self.reg[REG_CSR]:08x} {self.reg[REG_CSR]}')

    def dump_bf16(self, addr=0, size=None):
        if size is None:
            size = len(self.memory) - addr

        data = self.memory[addr:addr + size]
        _dump_bf16(data)

if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} [target]')
    sys.exit(0)

kwargs = {}

if len(sys.argv) >= 4 and sys.argv[2] == '-o':
    kwargs['out'] = sys.argv[3]


host = Host(**kwargs)
host.init(sys.argv[1])
host.run()
