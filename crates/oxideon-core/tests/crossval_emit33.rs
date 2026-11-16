//! Cross-validation batch 33: FFT family (power-of-two and arbitrary lengths, 2-D, STFT, complex helpers), values and
//! gradients vs torch.fft (tools/crossval/check_batch33.py).
use oxideon_core::tensor::Tensor;

fn genv(n: usize, salt: f64) -> Vec<f64> {
    (0..n).map(|i| (0.53 * i as f64 + salt).sin() * 1.1 + 0.2 * (0.17 * i as f64 * salt).cos()).collect()
}

fn t(shape: &[usize], salt: f64) -> Tensor {
    Tensor::from_slice(&genv(shape.iter().product(), salt), shape).unwrap()
}

fn emit(name: &str, x: &Tensor) {
    let v = x.to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|a| format!("{a:.13e}")).collect::<Vec<_>>().join(" "));
}

fn weight(shape: &[usize]) -> Tensor {
    t(shape, 77.0)
}

fn case(name: &str, x: &Tensor, out: &Tensor) {
    emit(&format!("{name}_out"), out);
    out.mul(&weight(out.shape())).unwrap().sum().unwrap().backward().unwrap();
    emit(&format!("{name}_dx"), &x.grad().unwrap());
}

#[test]
#[ignore]
fn emit_fft() {
    for n in [8usize, 16, 12, 7, 10, 1, 2, 9] {
        let x = t(&[3, n, 2], 1.0 + n as f64).requires_grad_(true);
        case(&format!("fft{n}"), &x, &x.fft().unwrap());
        let x = t(&[3, n, 2], 2.0 + n as f64).requires_grad_(true);
        case(&format!("ifft{n}"), &x, &x.ifft().unwrap());
        let x = t(&[2, n], 3.0 + n as f64).requires_grad_(true);
        case(&format!("rfft{n}"), &x, &x.rfft().unwrap());
        let x = t(&[2, n, 2], 4.0 + n as f64).requires_grad_(true);
        case(&format!("irfft{n}"), &x, &x.irfft().unwrap());
    }
    // 2-D (power-of-two and mixed sizes)
    for (h, w) in [(4usize, 8usize), (6, 5), (3, 4)] {
        let x = t(&[2, h, w, 2], 5.0 + h as f64).requires_grad_(true);
        case(&format!("fft2_{h}x{w}"), &x, &x.fft2d().unwrap());
        let x = t(&[2, h, w, 2], 6.0 + h as f64).requires_grad_(true);
        case(&format!("ifft2_{h}x{w}"), &x, &x.ifft2d().unwrap());
    }
    // complex helpers
    let (a, b) = (t(&[4, 5, 2], 7.0).requires_grad_(true), t(&[4, 5, 2], 8.0));
    case("cmul", &a, &a.complex_mul(&b).unwrap());
    let a = t(&[4, 5, 2], 9.0).requires_grad_(true);
    case("cabs", &a, &a.complex_abs().unwrap());
    // STFT (hann periodic window, no centering) and without window
    for (len, n_fft, hop, win) in [(40usize, 8usize, 4usize, true), (37, 12, 5, true), (32, 8, 8, false), (20, 16, 1, true)] {
        let x = t(&[2, len], 10.0 + len as f64).requires_grad_(true);
        case(&format!("stft_{len}_{n_fft}_{hop}_{}", u8::from(win)), &x, &x.stft(n_fft, hop, win).unwrap());
    }
}
