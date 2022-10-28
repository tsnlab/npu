data.f32 0x200000 [v * 1.1 for v in range(512)]
data.f32 0x200800 [v * 1.2 for v in range(512)]
data.f32 0x201000 [(v * 1.1) * (v * 1.2) for v in range(512)]

# Load from 0x0020 0000 to 0x0000 0000
set_high %a 0x0020
set_low %a 0x0000
set_high %b 0x0000
set_low %b 0x0100
load 512

# Load from 0x0020 0800 to 0x0000 0800
set_high %a 0x0020
set_low %a 0x0800
set_high %b 0x0000
set_low %b 0x0900
load 512

# A = 0x0000 0000 B = 0x0000 0800 C = 0x0000 1000
set_high %a 0x0000
set_low %a 0x0100
set_high %b 0x0000
set_low %b 0x0900
set_high %c 0x0000
set_low %c 0x1100
mul.f32 512

# Store to 0x0020 1000 from 0x0000 1000
set_high %a 0x0020
set_low %a 0x1000
set_high %b 0x0000
set_low %b 0x1100
store 512

# Interrupt to CPU
return
