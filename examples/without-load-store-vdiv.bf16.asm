### script
def init(host):
    host.data[0x200000] = jnp.array([(v + 1) * 1.1 for v in range(2048)], dtype=jnp.bfloat16).tobytes()
    host.data[0x201000] = jnp.array([(v + 1) * 0.1 for v in range(2048)], dtype=jnp.bfloat16).tobytes()
    host.data[0x202000] = zeros(2048 * 2)

    host.store(0, 0x00, 0x00, len(host.kernel))  # 0x00 means kernel
    host.store(0, 0x80, 0x200000, 2048 * 2)
    host.store(0, 0x1080, 0x201000, 2048 * 2)

def run(host):
    host.exec(0)

def finalize(host):
    print('# Load data into host memory')
    host.load(0, 0x202000, 0x2080, 2048 * 2)

    print('# Save data to files')
    save(host.data[0x200000], 'vdiv.200000.data')
    save(host.data[0x201000], 'vdiv.201000.data')
    save(host.data[0x202000], 'vdiv.202000.data')

    print('# Compare result')
    result = jnp.frombuffer(host.data[0x202000], dtype=jnp.bfloat16)

    A = jnp.array([(v + 1) * 1.1 for v in range(2048)], dtype=jnp.bfloat16)
    B = jnp.array([(v + 1) * 0.1 for v in range(2048)], dtype=jnp.bfloat16)

    dump_compare(result, A / B)
###

# kernel is stored at 0
# Assume the kernel size is 128 or less.
# The total data size is set to 2048 of the maximum bfloat16 type(2 Bytes) variable size.

# vdiv.bf16 C = A / B
seti %a 0x20  # 128 / 4  # A is stored at 128
seti %b 0x420  # (128 + 4096) / 4  # B is stored at 128 + 4096
seti %c 0x820  # (128 + 4096 *2) / 4  # C is stored at 128 + 4096 + 4096
seti %d 2048  # count is 2048

vdiv.bf16 %c %a %b %d

# Interrupt to CPU
return
