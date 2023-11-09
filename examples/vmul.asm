### script
def init(host):
    host.data[0x200000] = jnp.array([v * 0.11 for v in range(512)], dtype=jnp.bfloat16).tobytes()
    host.data[0x200400] = jnp.array([v * 0.01 for v in range(512)], dtype=jnp.bfloat16).tobytes()
    host.data[0x200800] = zeros(512 * 2)

    host.store(0, 0x00, 0x00, len(host.kernel))  # 0x00 means kernel
    host.store(0, 0x80, 0x200000, 512 * 2)
    host.store(0, 0x480, 0x200400, 512 * 2)

def run(host):
    host.exec(0)

def finalize(host):
    print('# Save data to files')
    save(host.data[0x200000], 'vadd.200000.data')
    save(host.data[0x200400], 'vadd.200400.data')
    save(host.data[0x200800], 'vadd.200800.data')

    print('# Compare result')
    result = jnp.frombuffer(host.data[0x200800], dtype=jnp.bfloat16)

    A = jnp.array([v * 0.11 for v in range(512)], dtype=jnp.bfloat16)
    B = jnp.array([v * 0.01 for v in range(512)], dtype=jnp.bfloat16)

    dump_compare(result, A * B)
###

# kernel is stored at 0
seti %a 0x20  # 128 / 4  # A is stored at 128
seti %b 0x120  # (128 + 1024) / 4  # B is stored at 128 + 1024
seti %c 0x220  # (128 + 1024) / 4  # C is stored at 128 + 1024 + 1024
seti %d 512  # count is 512

vmul.bf16 %c %a %b %d

# Store C to 0x200800
seti %a 0x4010  # 0x200800 / 128
seti %b 0x220  # (128 + 1024 + 1024) / 4
seti %c 256  # (512 * 2) / 4
store %a %b %c

# Interrupt to CPU
return
