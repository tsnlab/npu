import glob
import struct
import sys
import threading

import jax.numpy as jnp


is_debug = False

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

RED = '\033[0;31m'
END = '\033[0m'

STATUS_RUNNING = 1 << 0
STATUS_LOADING = 1 << 1
STATUS_ERROR = 1 << 31


def IS_READABLE_REG(reg):
    return reg >= REG_ZERO and reg <= REG_G or reg >= REG_IP and reg <= REG_CSR


def IS_WRITABLE_REG(reg):
    return reg >= REG_A and reg <= REG_G


def float32_to_bf16(value):
    if isinstance(value, tuple) or isinstance(value, list):
        f32_values = value

        buf = bytearray()

        for f32_value in f32_values:
            buf.extend(float32_to_bf16(f32_value))

        return buf
    else:
        f32_value = value

        f32_bytes = struct.pack('f', f32_value)
        return f32_bytes[2:]


def bf16_to_float32(value):
    if len(value) > 2:
        bf16_bufs = value
        f32_values = []

        i = 0
        while i < len(bf16_bufs):
            f32_values.append(bf16_to_float32(bf16_bufs[i: i + 2]))
            i += 2

        return f32_values
    else:
        bf16_buf = value

        f32_bytes = bytes([0, 0, bf16_buf[0], bf16_buf[1]])
        return struct.unpack('f', f32_bytes)[0]


def zeros(size):
    buf = bytearray()

    while size > 0:
        buf.extend(struct.pack('I', 0))
        size -= 4

    return buf


def compare(a, b, epsilon=1e-4):
    diff = a - b if a > b else b - a

    return diff < epsilon


def dump_compare(A, B, epsilon=1e-4):
    A_count = len(A)
    B_count = len(B)
    count = max(A_count, B_count)

    incorrect = 0

    for i in range(count):
        if i < A_count and i < B_count:
            a = A[i]
            b = B[i]

            if compare(a, b, epsilon):
                print(f'{a}', end=' ')
            else:
                print(f'{RED}{a} != {b}{END}', end=' ')
                incorrect += 1
        elif i < A_count:
            print(f'{RED}{a} != None{END}', end=' ')
            incorrect += 1
        else:
            print(f'{RED}None != {b}', end=' ')
            incorrect += 1

        if (i + 1) % 16 == 0:
            print()
    print(f'Incorrect: {incorrect} / {count}')


def dump_bf16(data):
    count = 1
    i = 0
    while i < len(data):
        value = jnp.frombuffer(data[i:i + 2], dtype=jnp.bfloat16)[0]

        print(value, end=' ')
        if count % 16 == 0:
            print()

        i += 2
        count += 1


def dump_hex(data):
    count = 1
    for i in range(len(data)):
        b = data[i]

        print(f'{b:02x}', end=' ')

        if count % 16 == 0:
            print()
        elif count % 8 == 0:
            print(end=' ')

        count += 1


class Host:
    def __init__(self, mem_size=512 * 1024, npus=4, out='stdout'):
        self.kernel = None
        self.data = {}
        self.updated = {}
        self.npus = []
        self.script = None
        self._context = {
            'host': self,
            'float32_to_bf16': float32_to_bf16,
            'bf16_to_float32': bf16_to_float32,
            'zeros': zeros,
            'compare': compare,
            'dump_compare': dump_compare,
            'dump_bf16': dump_bf16,
            'dump_hex': dump_hex,
            'jnp': jnp
        }

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
        exec(self.script + '\ninit(host)', None, self._context)

    def get_data(self, offset, size):
        if offset == 0x00:
            return 0, self.kernel

        for addr, data in self.data.items():
            if addr >= offset and addr + len(data) <= offset + size:
                return addr, data

        raise Exception(f'There is no such data: {offset:08x} ~ {offset + size:08x}')

    def run(self):
        # Run run script
        exec(self.script + '\nrun(host)', None, self._context)

        for npu in self.npus:
            if npu.is_alive():
                npu.join()

        # Run finalizescript
        exec(self.script + '\nfinalize(host)', None, self._context)

    def store(self, npu_id, npu_address, host_address, size):
        if npu_id < 0 or npu_id >= len(self.npus):
            raise Exception(f'Illegal NPU id: {npu_id}')

        npu = self.npus[npu_id]

        # set npu_address
        self.set(npu_id, REG_A, npu_address // 4)

        # set host_address
        self.set(npu_id, REG_B, host_address // 128)

        # set size
        self.set(npu_id, REG_C, (size + 3) // 4)

        # store (from host side, load from NPU side)
        npu.op_load(REG_A, REG_B, REG_C)

    def load(self, npu_id, host_address, npu_address, size):
        if npu_id < 0 or npu_id >= len(self.npus):
            raise Exception(f'Illegal NPU id: {npu_id}')

        npu = self.npus[npu_id]

        # set host_address
        self.set(npu_id, REG_A, host_address // 128)

        # set npu_address
        self.set(npu_id, REG_B, npu_address // 4)

        # set size
        self.set(npu_id, REG_C, (size + 3) // 4)

        # load (from host side, store from NPU side)
        npu.op_store(REG_A, REG_B, REG_C)

    def exec(self, npu_id):
        self.npus[npu_id].start()

    def set(self, npu_id, reg_id, value):
        if npu_id < 0 or npu_id >= len(self.npus):
            raise Exception(f'Illegal NPU id: {npu_id}')

        npu = self.npus[npu_id]

        if value > 2 ** 20:
            npu.op_seti_low(reg_id, value & 0xffff)
            npu.op_seti_high(reg_id, (value >> 16) & 0xffff)
        else:
            npu.op_seti(reg_id, value)

    def get(self, npu_id, reg_id):
        if npu_id < 0 or npu_id >= len(self.npus):
            raise Exception(f'Illegal NPU id: {npu_id}')

        npu = self.npus[npu_id]

        return npu.reg[reg_id]


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

    def op_nop(self):
        pass

    def op_set(self, dest, src):
        if not IS_WRITABLE_REG(dest) or not IS_READABLE_REG(src) or not self.is_inbound_memory(self.reg[src] * 4, 4):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x}, 0x{src:02x} '
                                'or memory out of bound: 0x{src:08x} ~ + 4 bytes')
            return

        self.reg[dest] = struct.unpack('I', self.memory[src * 4:src * 4 + 4])[0]

    def op_seti(self, dest, value):
        if not IS_WRITABLE_REG(dest):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x}')
            return

        self.reg[dest] = value

    def op_seti_low(self, dest, value):
        if not IS_WRITABLE_REG(dest):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x}')
            return

        self.reg[dest] = (self.reg[dest] & 0xffff0000) | (value & 0xffff)

    def op_seti_high(self, dest, value):
        if not IS_WRITABLE_REG(dest):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x}')
            return

        self.reg[dest] = ((value & 0xffff) << 16) | (self.reg[dest] & 0xffff)

    def op_get(self, src, dest):
        if not IS_READABLE_REG(src):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{src:02x}')
            return

        self.memory[dest:dest + 4] = struct.pack('I', self.reg[src])

    def op_mov(self, dest, src):
        if not IS_WRITABLE_REG(dest) or not IS_READABLE_REG(src):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x} or 0x{src:02x}')
            return

        self.reg[dest] = self.reg[src]

    def op_load(self, dest, src, count):
        if not IS_READABLE_REG(dest) or not IS_READABLE_REG(src) or not IS_READABLE_REG(count):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{dest:02x}, 0x{src:02x} or 0x{count:02x}')
            return

        dest_addr = self.reg[dest] * 4
        src_addr = self.reg[src] * 128
        size = self.reg[count] * 4

        if not self.is_inbound_memory(dest_addr, size):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'NPU{self.id}: Out of memory: 0x{dest_addr:08x} ~ +{size} bytes')
            return

        if self.reg[REG_CSR] & STATUS_LOADING != 0:
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'NPU{self.id}: Duplicated load/store operator executed')
            return

        self.reg[REG_CSR] |= STATUS_LOADING
        addr, data = host.get_data(src_addr, size)
        self.memory[dest_addr:dest_addr + size] = data[src_addr - addr: src_addr - addr + size]
        self.reg[REG_CSR] ^= STATUS_LOADING

    def op_store(self, dest, src, count):
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
                raise Exception(f'NPU{self.id}: Out of memory: 0x{src_addr:08x} ~ +{size} bytes')
            return

        if self.reg[REG_CSR] & STATUS_LOADING != 0:
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'NPU{self.id}: Duplicated load/store operator executed')
            return

        self.reg[REG_CSR] |= STATUS_LOADING
        addr, data = host.get_data(dest_addr, size)
        data[dest_addr - addr:dest_addr - addr + size] = self.memory[src_addr:src_addr + size]
        self.reg[REG_CSR] ^= STATUS_LOADING

    def op_vadd_bf16(self, c, a, b, count):
        if not IS_READABLE_REG(c) or not IS_READABLE_REG(a) or \
           not IS_READABLE_REG(b) or not IS_READABLE_REG(count):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{c:02x}, 0x{a:02x}, 0x{b:02x} or 0x{count:02x}')
            return

        c_addr = self.reg[c] * 4
        a_addr = self.reg[a] * 4
        b_addr = self.reg[b] * 4
        count = self.reg[count]

        for i in range(count):
            a = jnp.frombuffer(self.memory[a_addr:a_addr + 2], dtype=jnp.bfloat16)[0]
            b = jnp.frombuffer(self.memory[b_addr:b_addr + 2], dtype=jnp.bfloat16)[0]
            c = a + b
            self.memory[c_addr:c_addr + 2] = c.tobytes()

            a_addr += 2
            b_addr += 2
            c_addr += 2

    def op_vsub_bf16(self, c, a, b, count):
        if not IS_READABLE_REG(c) or not IS_READABLE_REG(a) or \
           not IS_READABLE_REG(b) or not IS_READABLE_REG(count):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{c:02x}, 0x{a:02x}, 0x{b:02x} or 0x{count:02x}')
            return

        c_addr = self.reg[c] * 4
        a_addr = self.reg[a] * 4
        b_addr = self.reg[b] * 4
        count = self.reg[count]

        for i in range(count):
            a = jnp.frombuffer(self.memory[a_addr:a_addr + 2], dtype=jnp.bfloat16)[0]
            b = jnp.frombuffer(self.memory[b_addr:b_addr + 2], dtype=jnp.bfloat16)[0]
            c = a - b
            self.memory[c_addr:c_addr + 2] = c.tobytes()

            a_addr += 2
            b_addr += 2
            c_addr += 2

    def op_vmul_bf16(self, c, a, b, count):
        if not IS_READABLE_REG(c) or not IS_READABLE_REG(a) or \
           not IS_READABLE_REG(b) or not IS_READABLE_REG(count):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{c:02x}, 0x{a:02x}, 0x{b:02x} or 0x{count:02x}')
            return

        c_addr = self.reg[c] * 4
        a_addr = self.reg[a] * 4
        b_addr = self.reg[b] * 4
        count = self.reg[count]

        for i in range(count):
            a = jnp.frombuffer(self.memory[a_addr:a_addr + 2], dtype=jnp.bfloat16)[0]
            b = jnp.frombuffer(self.memory[b_addr:b_addr + 2], dtype=jnp.bfloat16)[0]
            c = a * b
            self.memory[c_addr:c_addr + 2] = c.tobytes()

            a_addr += 2
            b_addr += 2
            c_addr += 2

    def op_vdiv_bf16(self, c, a, b, count):
        if not IS_READABLE_REG(c) or not IS_READABLE_REG(a) or \
           not IS_READABLE_REG(b) or not IS_READABLE_REG(count):
            self.reg[REG_CSR] |= STATUS_ERROR
            self.reg[REG_CSR] &= ~STATUS_RUNNING
            if is_debug:
                raise Exception(f'Illegal register: 0x{c:02x}, 0x{a:02x}, 0x{b:02x} or 0x{count:02x}')
            return

        c_addr = self.reg[c] * 4
        a_addr = self.reg[a] * 4
        b_addr = self.reg[b] * 4
        count = self.reg[count]

        for i in range(count):
            a = jnp.frombuffer(self.memory[a_addr:a_addr + 2], dtype=jnp.bfloat16)[0]
            b = jnp.frombuffer(self.memory[b_addr:b_addr + 2], dtype=jnp.bfloat16)[0]
            c = a / b
            self.memory[c_addr:c_addr + 2] = c.tobytes()

            a_addr += 2
            b_addr += 2
            c_addr += 2

    def op_return(self):
        self.reg[REG_CSR] &= ~STATUS_RUNNING

    def exec(self, opcode):
        if is_debug:
            print(f'NPU {self.id} [{self.reg[REG_IP]:>3}] {opcode:08x}')

        instruction = opcode >> 24

        if instruction == 0x00:  # nop
            self.op_nop()
        elif instruction == 0x01:  # set
            dest, src = self.parse(opcode, ['r', 'u20'])
            self.op_set(dest, src)
        elif instruction == 0x02:  # seti
            dest, value = self.parse(opcode, ['r', 'u20'])
            self.op_seti(dest, value)
        elif instruction == 0x03:  # seti_low
            dest, value = self.parse(opcode, ['r', 'p4', 'u16'])
            self.op_seti_low(dest, value)
        elif instruction == 0x04:  # seti_high
            dest, value = self.parse(opcode, ['r', 'p4', 'u16'])
            self.op_seti_high(dest, value)
        elif instruction == 0x05:  # get
            src, dest = self.parse(opcode, ['r', 'u20'])
            self.op_get(src, dest)
        elif instruction == 0x06:  # mov
            dest, src = self.parse(opcode, ['r', 'r'])
            self.op_mov(dest, src)
        elif instruction == 0x07:  # load
            dest, src, count = self.parse(opcode, ['r', 'r', 'r'])
            self.op_load(dest, src, count)
        elif instruction == 0x08:  # store
            dest, src, count = self.parse(opcode, ['r', 'r', 'r'])
            self.op_store(dest, src, count)
        elif instruction == 0x09:  # vadd.bf16
            c, a, b, count = self.parse(opcode, ['r', 'r', 'r', 'r'])
            self.op_vadd_bf16(c, a, b, count)
        elif instruction == 0x0a:  # vsub.bf16
            c, a, b, count = self.parse(opcode, ['r', 'r', 'r', 'r'])
            self.op_vsub_bf16(c, a, b, count)
        elif instruction == 0x0b:  # vmul.bf16
            c, a, b, count = self.parse(opcode, ['r', 'r', 'r', 'r'])
            self.op_vmul_bf16(c, a, b, count)
        elif instruction == 0x0c:  # vdiv.bf16
            c, a, b, count = self.parse(opcode, ['r', 'r', 'r', 'r'])
            self.op_vdiv_bf16(c, a, b, count)
        elif instruction == 0x0d:  # add.i32
            pass
        elif instruction == 0x0e:  # sub.i32
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
            self.op_return()
        else:
            if is_debug:
                raise Exception(f'Not supported opcode: 0x{instruction:02x}')

            self.reg[REG_CSR] |= STATUS_ERROR

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


if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} [target]')
    sys.exit(0)

kwargs = {}

if len(sys.argv) >= 4 and sys.argv[2] == '-o':
    kwargs['out'] = sys.argv[3]


host = Host(**kwargs)
host.init(sys.argv[1])
host.run()
