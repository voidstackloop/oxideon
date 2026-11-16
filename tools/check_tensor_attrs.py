import oxideon as ox

t = ox.randn([2, 3])
for attr in ["real", "imag", "conj", "conjugate", "angle", "is_floating_point", "is_complex", "is_signed", "type_as"]:
    print(attr, hasattr(t, attr))
