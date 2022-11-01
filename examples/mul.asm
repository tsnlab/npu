data.f32 0x200000 [v * 1.1 for v in range(512)]
data.f32 0x200800 [v * 1.2 for v in range(512)]
data.f32 0x201000 [0 for v in range(512)]

# Load from 0x0020 0000 to 0x0000 0000
set %a 0x00200000
set %b 0x00000100
load 512

# Load from 0x0020 0800 to 0x0000 0800
set %a 0x00200800
set %b 0x00000900
load 512

# A = 0x0000 0000 B = 0x0000 0800 C = 0x0000 1000
set %a 0x00000100
set %b 0x00000900
set %c 0x00001100
mul.f32 512

# Store to 0x0020 1000 from 0x0000 1000
set %a 0x00201000
set %b 0x00001100
store 512

# Interrupt to CPU
return
