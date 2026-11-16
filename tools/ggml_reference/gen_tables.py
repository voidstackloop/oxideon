import re, sys
LLAMA = sys.argv[1] if len(sys.argv) > 1 else 'llama.cpp'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'crates/oxideon-core/src/serialize/gguf_iq_tables.rs'
src = open(LLAMA + '/ggml/src/ggml-common.h').read()

def table(name, ctype, size):
    m = re.search(r'GGML_TABLE_BEGIN\(%s, %s, (\w+)\)(.*?)GGML_TABLE_END\(\)' % (ctype, name), src, re.S)
    assert m, name
    vals = [int(v, 0) for v in re.findall(r'0x[0-9a-fA-F]+|-?\d+', re.sub(r'//.*', '', m.group(2)))]
    assert len(vals) == size, (name, len(vals), size)
    return vals

tables = [
    ('KSIGNS_IQ2XS', 'ksigns_iq2xs', 'uint8_t', 128, 'u8'),
    ('IQ2XXS_GRID', 'iq2xxs_grid', 'uint64_t', 256, 'u64'),
    ('IQ2XS_GRID', 'iq2xs_grid', 'uint64_t', 512, 'u64'),
    ('IQ2S_GRID', 'iq2s_grid', 'uint64_t', 1024, 'u64'),
    ('IQ3XXS_GRID', 'iq3xxs_grid', 'uint32_t', 256, 'u32'),
    ('IQ3S_GRID', 'iq3s_grid', 'uint32_t', 512, 'u32'),
    ('IQ1S_GRID', 'iq1s_grid', 'uint64_t', 2048, 'u64'),
]
out = ['''//! Codebook lookup tables for the `IQ1_*`, `IQ2_*` and `IQ3_*` GGUF quantizations.
//!
//! Generated verbatim from llama.cpp / GGML `ggml/src/ggml-common.h` (MIT License,
//! Copyright (c) 2023-2024 The ggml authors) by extracting the `GGML_TABLE_BEGIN` arrays; do not edit.
//! Each grid entry packs 8 (`u64`) or 4 (`u32`) unsigned/signed byte weights, least-significant byte first.
''']
for const, name, ctype, size, rust in tables:
    vals = table(name, ctype, size)
    width = {'u8': 2, 'u32': 8, 'u64': 16}[rust]
    out.append('pub const %s: [%s; %d] = [' % (const, rust, size))
    per = 8 if rust == 'u8' else 4
    for i in range(0, size, per):
        out.append('    ' + ', '.join(('0x%0' + str(width) + 'x') % v for v in vals[i:i + per]) + ',')
    out.append('];\n')
open(OUT, 'w').write('\n'.join(out))
print('tables written')
