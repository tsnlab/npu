data.f32 0x200000 [v * 1.1 for v in range(512)] # A
data.f32 0x200800 [v * 1.2 for v in range(512)] # B
data.f32 0x201000 [v * 1.3 for v in range(512)] # C
data.f32 0x201800 [0.0 for v in range(512)]

# Load from 0x0020 0000 to 0x0000 0000
set %a 0x00200000
set %b 0x00000100
load 512

# Load from 0x0020 0800 to 0x0000 0800
set %a 0x00200800
set %b 0x00000900
load 512

# Load from 0x0020 0800 to 0x0000 0800
set %a 0x00201000
set %b 0x00001100
load 512

# D = A * B
set %a 0x00000100
set %b 0x00000900
set %c 0x00001900
mul.f32 512

# E = B + C
set %a 0x00000900
set %b 0x00001100
set %c 0x00002100
add.f32 512

# E = D - E
set %a 0x00001900
set %b 0x00002100
set %c 0x00002100
sub.f32 512

set %a 0x00201800
set %b 0x00002100
store 512

# Interrupt to CPU
return
