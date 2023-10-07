### script
def init(host):
    host.data[0x200000] = float32_to_bf16([v * 1.1 for v in range(512)])
    host.data[0x200400] = float32_to_bf16([v * 0.1 for v in range(512)])
    host.data[0x200800] = zeros(512 * 2)

    host.store(0, 0x00, 0x00, len(host.kernel))  # 0x00 means kernel
    host.store(0, 0x80, 0x200000, 512 * 2)
    host.store(0, 0x480, 0x200400, 512 * 2)

def run(host):
    host.exec(0)

def finalize(host):
    #host.npus[0].dump_regs()
    host.dump_bf16(0x200800)
    f32_values = bf16_to_float32(host.data[0x200800])
    dump_compare([v * 1.1 + v * 0.1 for v in range(512)], f32_values)
    #print('A')
    #host.npus[0].dump_bf16(0x80, 512)
    #print('B')
    #host.npus[0].dump_bf16(0x480, 512)
    #print('C')
    #host.npus[0].dump_bf16(0x880, 512)
###

# kernel is stored at 0
# A is stored at 128
seti %a 0x20  # 128 / 4
# B is stored at 128 + 1024
seti %b 0x120  # (128 + 1024) / 4
# C is stored at 128 + 1024 + 1024
seti %c 0x220  # (128 + 1024) / 4
# count is 512
seti %d 512

vadd.bf16 %c %a %b %d

# Store C to 0x200800
seti %a 0x4010  # 0x200800 / 128
seti %b 0x220  # (128 + 1024 + 1024) / 4
seti %c 256  # (512 * 2) / 4
store %a %b %c

# Interrupt to CPU
return
