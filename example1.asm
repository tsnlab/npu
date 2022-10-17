set_low %m 0x0000
set_high %m 0x0020
set_low %s 0x0000
set_high %s 0x0000

load 512
add.f32 512
store 512
return
