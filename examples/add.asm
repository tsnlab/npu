data.bf16 0x200000 [v * 1.1 for v in range(512)]
data.bf16 0x200400 [v * 1.2 for v in range(512)]
data.bf16 0x200800 [0 for v in range(512)]

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
