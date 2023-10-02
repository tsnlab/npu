### script
def init(host):
    host.data[0x200000] = host.bf16([v * 1.1 for v in range(512)])
    host.data[0x200400] = host.bf16([v * 0.1 for v in range(512)])
    host.data[0x200800] = host.zeros(512 * 2)

    host.store(0, 0x00, 0x00)  # 0x00 means kernel
    host.store(0, 0x100, 0x200000, 512 * 2)
    host.store(0, 0x500, 0x200400, 512 * 2)

def run(host):
    host.exec(0)

def finalize(host):
    #host.dump_bf16(0x200800)
    #host.dump(0x200800)
    #host.npus[0].dump_bf16()
    host.npus[0].dump_regs()
###

# A is stored at 0x200
seti %a 0x80  # 0x200 / 4
# B is stored at 0x600
seti %b 0x180  # 0x600 / 4
# C is stored at 0xa00
seti %c 0x280  # 0xa00 / 4
# count is 512
seti %d 512

vadd.bf16 %c %a %b %d

# Store C to 0x200800
seti %a 0x410  # 0x200800 / 128
seti %b 0x280  # 0xa00 / 4
seti %c 256  # (512 * 2) / 4
store %a %b %c

# Interrupt to CPU
return
