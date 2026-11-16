import pytest
import numpy as np
import oxideon as ox


def test_dtypes_existence_and_properties():
    # Integer dtypes
    assert ox.int8.itemsize == 1
    assert ox.int8.bits == 8
    assert ox.int8.is_signed is True
    assert ox.int8.is_floating_point is False
    assert ox.int8.is_complex is False
    assert ox.char == ox.int8

    assert ox.int16.itemsize == 2
    assert ox.int16.bits == 16
    assert ox.int16.is_signed is True
    assert ox.short == ox.int16

    assert ox.int32.itemsize == 4
    assert ox.int_ == ox.int32

    assert ox.int64.itemsize == 8
    assert ox.long == ox.int64

    # Unsigned dtypes
    assert ox.uint8.itemsize == 1
    assert ox.uint8.is_signed is False
    assert ox.byte == ox.uint8

    assert ox.uint16.itemsize == 2
    assert ox.uint16.is_signed is False

    assert ox.uint32.itemsize == 4
    assert ox.uint32.is_signed is False

    assert ox.uint64.itemsize == 8
    assert ox.uint64.is_signed is False

    # Floating point dtypes
    assert ox.float16.itemsize == 2
    assert ox.half == ox.float16
    assert ox.float16.is_floating_point is True

    assert ox.bfloat16.itemsize == 2
    assert ox.bfloat16.is_floating_point is True

    assert ox.float32.itemsize == 4
    assert ox.float_ == ox.float32
    assert ox.float32.is_floating_point is True

    assert ox.float64.itemsize == 8
    assert ox.double == ox.float64
    assert ox.float64.is_floating_point is True

    # FP8 dtypes
    assert ox.float8_e4m3fn.itemsize == 1
    assert ox.float8_e4m3fn.bits == 8
    assert ox.float8_e4m3fn.is_floating_point is True

    assert ox.float8_e5m2.itemsize == 1
    assert ox.float8_e5m2.bits == 8
    assert ox.float8_e5m2.is_floating_point is True

    # Complex dtypes
    assert ox.complex64.itemsize == 8
    assert ox.cfloat == ox.complex64
    assert ox.complex64.is_complex is True
    assert ox.complex64.is_floating_point is True

    assert ox.complex128.itemsize == 16
    assert ox.cdouble == ox.complex128
    assert ox.complex128.is_complex is True

    # String equality and normalization
    assert ox.int8 == "int8"
    assert ox.int8 == "i8"
    assert ox.int16 == "short"
    assert ox.int64 == "long"
    assert ox.float32 == "float"
    assert ox.float64 == "double"
    assert ox.complex64 == "cfloat"
    assert ox.complex128 == "cdouble"
    assert ox.float8_e4m3fn == "e4m3"
    assert ox.float8_e5m2 == "e5m2"
    assert ox.bool_ == "bool"
    assert ox.bool_ == bool


def test_finfo_iinfo():
    f32_info = ox.finfo(ox.float32)
    assert f32_info.bits == 32
    assert abs(f32_info.eps - 1.1920929e-07) < 1e-12
    assert f32_info.max > 3e38
    assert f32_info.min < -3e38

    f64_info = ox.finfo(ox.float64)
    assert f64_info.bits == 64
    assert abs(f64_info.eps - 2.220446049250313e-16) < 1e-20

    f16_info = ox.finfo(ox.float16)
    assert f16_info.bits == 16
    assert f16_info.max == 65504.0
    assert f16_info.min == -65504.0

    bf16_info = ox.finfo(ox.bfloat16)
    assert bf16_info.bits == 16
    assert bf16_info.eps == 0.0078125

    c64_info = ox.finfo(ox.complex64)
    assert c64_info.bits == 64

    fp8_info = ox.finfo(ox.float8_e4m3fn)
    assert fp8_info.bits == 8
    assert fp8_info.max == 448.0

    fp8_e5_info = ox.finfo(ox.float8_e5m2)
    assert fp8_e5_info.bits == 8
    assert fp8_e5_info.max == 57344.0

    # iinfo
    i8_info = ox.iinfo(ox.int8)
    assert i8_info.bits == 8
    assert i8_info.min == -128
    assert i8_info.max == 127

    u8_info = ox.iinfo(ox.uint8)
    assert u8_info.bits == 8
    assert u8_info.min == 0
    assert u8_info.max == 255

    i16_info = ox.iinfo(ox.int16)
    assert i16_info.bits == 16
    assert i16_info.min == -32768
    assert i16_info.max == 32767

    u16_info = ox.iinfo(ox.uint16)
    assert u16_info.bits == 16
    assert u16_info.min == 0
    assert u16_info.max == 65535

    i32_info = ox.iinfo(ox.int32)
    assert i32_info.bits == 32
    assert i32_info.min == -2147483648
    assert i32_info.max == 2147483647

    u32_info = ox.iinfo(ox.uint32)
    assert u32_info.bits == 32
    assert u32_info.min == 0
    assert u32_info.max == 4294967295

    i64_info = ox.iinfo(ox.int64)
    assert i64_info.bits == 64
    assert i64_info.min == -9223372036854775808
    assert i64_info.max == 9223372036854775807

    u64_info = ox.iinfo(ox.uint64)
    assert u64_info.bits == 64
    assert u64_info.min == 0
    assert u64_info.max == 18446744073709551615

    # Should raise on wrong category
    with pytest.raises(TypeError):
        ox.finfo(ox.int32)
    with pytest.raises(TypeError):
        ox.iinfo(ox.float32)


def test_promote_types():
    assert ox.promote_types(ox.int8, ox.int16) == ox.int16
    assert ox.promote_types(ox.int16, ox.int32) == ox.int32
    assert ox.promote_types(ox.int32, ox.int64) == ox.int64
    assert ox.promote_types(ox.uint8, ox.int16) == ox.int16
    assert ox.promote_types(ox.int8, ox.uint8) == ox.int16

    assert ox.promote_types(ox.float16, ox.float32) == ox.float32
    assert ox.promote_types(ox.float32, ox.float64) == ox.float64
    assert ox.promote_types(ox.bfloat16, ox.float32) == ox.float32

    assert ox.promote_types(ox.int32, ox.float32) == ox.float32
    assert ox.promote_types(ox.int64, ox.float32) == ox.float64
    assert ox.promote_types(ox.int64, ox.float64) == ox.float64

    assert ox.promote_types(ox.bool_, ox.int32) == ox.int32
    assert ox.promote_types(ox.bool_, ox.float32) == ox.float32

    assert ox.promote_types(ox.float32, ox.complex64) == ox.complex64
    assert ox.promote_types(ox.float64, ox.complex64) == ox.complex128
    assert ox.promote_types(ox.complex64, ox.complex128) == ox.complex128


def test_can_cast():
    assert ox.can_cast(ox.int8, ox.int16) is True
    assert ox.can_cast(ox.int16, ox.int8) is False
    assert ox.can_cast(ox.int32, ox.float64) is True
    assert ox.can_cast(ox.float32, ox.int32) is False

    assert ox.can_cast(ox.int32, ox.int8, casting="same_kind") is True
    assert ox.can_cast(ox.float64, ox.int8, casting="same_kind") is False
    assert ox.can_cast(ox.float64, ox.int8, casting="unsafe") is True


def test_type_predicates():
    f = ox.tensor([1.0, 2.0], dtype=ox.float32)
    i = ox.tensor([1, 2], dtype=ox.int64)
    u = ox.tensor([1, 2], dtype=ox.uint8)

    assert ox.is_floating_point(f) is True
    assert ox.is_floating_point(i) is False
    assert ox.is_floating_point(ox.float32) is True
    assert ox.is_floating_point(ox.int32) is False

    assert ox.is_signed(i) is True
    assert ox.is_signed(u) is False
    assert ox.is_signed(ox.int8) is True
    assert ox.is_signed(ox.uint8) is False

    assert f.is_floating_point() is True
    assert i.is_floating_point() is False
    assert i.is_signed() is True
    assert u.is_signed() is False


def test_tensor_casting_methods():
    x = ox.tensor([1.5, 2.5, 3.5], dtype=ox.float32)

    # .short() -> int16 / i32
    s = x.short()
    assert s.to_numpy().tolist() == [1, 2, 3]

    # .char() -> int8 / i32
    c = x.char()
    assert c.to_numpy().tolist() == [1, 2, 3]

    # .type_as()
    target = ox.tensor([0.0], dtype=ox.float64)
    x64 = x.type_as(target)
    assert x64.dtype == "f64" or x64.dtype == ox.float64

    # .type()
    assert isinstance(x.type(), str)
    x_cast = x.type(ox.float64)
    assert x_cast.dtype == "f64" or x_cast.dtype == ox.float64


def test_complex_tensors():
    r = ox.tensor([1.0, 2.0, 3.0], dtype=ox.float32, requires_grad=True)
    im = ox.tensor([4.0, 5.0, 6.0], dtype=ox.float32, requires_grad=True)

    c = ox.complex(r, im)
    assert c.is_complex() is True
    assert ox.is_complex(c) is True
    assert c.real.to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert c.imag.to_numpy().tolist() == [4.0, 5.0, 6.0]

    # conj
    conj_c = c.conj()
    assert conj_c.imag.to_numpy().tolist() == [-4.0, -5.0, -6.0]

    # abs
    c_abs = c.abs()
    expected_abs = np.sqrt(np.array([1, 2, 3])**2 + np.array([4, 5, 6])**2)
    assert np.allclose(c_abs.to_numpy(), expected_abs)

    # angle
    c_angle = c.angle()
    expected_angle = np.arctan2(np.array([4, 5, 6]), np.array([1, 2, 3]))
    assert np.allclose(c_angle.to_numpy(), expected_angle)

    # arithmetic
    c2 = ox.complex(ox.tensor([2.0, 1.0, 0.0]), ox.tensor([1.0, 2.0, 3.0]))
    sum_c = c + c2
    assert np.allclose(sum_c.real.to_numpy(), [3.0, 3.0, 3.0])
    assert np.allclose(sum_c.imag.to_numpy(), [5.0, 7.0, 9.0])

    diff_c = c - c2
    assert np.allclose(diff_c.real.to_numpy(), [-1.0, 1.0, 3.0])
    assert np.allclose(diff_c.imag.to_numpy(), [3.0, 3.0, 3.0])

    # multiplication: (1+4i)(2+1i) = (2-4) + (1+8)i = -2 + 9i
    prod_c = c * c2
    np_c1 = np.array([1+4j, 2+5j, 3+6j])
    np_c2 = np.array([2+1j, 1+2j, 0+3j])
    assert np.allclose(prod_c.numpy(), np_c1 * np_c2)

    # division
    div_c = c / c2
    assert np.allclose(div_c.numpy(), np_c1 / np_c2)

    # polar
    mag = ox.tensor([2.0, 3.0])
    phase = ox.tensor([0.0, np.pi / 2])
    pol = ox.polar(mag, phase)
    assert np.allclose(pol.real.to_numpy(), [2.0, 0.0], atol=1e-6)
    assert np.allclose(pol.imag.to_numpy(), [0.0, 3.0], atol=1e-6)

    # view_as_complex and view_as_real
    real_mat = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    comp_mat = ox.view_as_complex(real_mat)
    assert comp_mat.shape == (2,)
    assert comp_mat.real.to_numpy().tolist() == [1.0, 3.0]
    assert comp_mat.imag.to_numpy().tolist() == [2.0, 4.0]

    back_real = ox.view_as_real(comp_mat)
    assert np.allclose(back_real.to_numpy(), real_mat.to_numpy())

    # autograd check through complex
    loss = (c.real * 2.0 + c.imag * 3.0).sum()
    loss.backward()
    assert r.grad.to_numpy().tolist() == [2.0, 2.0, 2.0]
    assert im.grad.to_numpy().tolist() == [3.0, 3.0, 3.0]


def test_real_tensor_complex_methods():
    x = ox.tensor([1.0, -2.0, 3.0])
    assert x.real.to_numpy().tolist() == [1.0, -2.0, 3.0]
    assert x.imag.to_numpy().tolist() == [0.0, 0.0, 0.0]
    assert x.conj().to_numpy().tolist() == [1.0, -2.0, 3.0]
    assert x.conjugate().to_numpy().tolist() == [1.0, -2.0, 3.0]
    assert x.is_complex() is False
