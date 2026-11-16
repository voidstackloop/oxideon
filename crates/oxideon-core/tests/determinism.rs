//! Results must not depend on how many threads run: reductions, matmul, conv and softmax are bit-identical on 1, 2, 3 and 8
//! threads (a parallel reduction that adds in scheduling order would break this).

use oxideon_core::Tensor;

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.37).sin() * 3.0 + 1e-3 * (i % 7) as f32).collect()
}

fn run() -> Vec<Vec<u32>> {
    let a = Tensor::from_slice(&data(300 * 257, 0.0), &[300, 257]).unwrap();
    let b = Tensor::from_slice(&data(257 * 190, 1.0), &[257, 190]).unwrap();
    let img = Tensor::from_slice(&data(2 * 3 * 40 * 40, 2.0), &[2, 3, 40, 40]).unwrap();
    let w = Tensor::from_slice(&data(8 * 3 * 3 * 3, 3.0), &[8, 3, 3, 3]).unwrap();
    let outs = [
        a.sum().unwrap(),
        a.mean_axis(0, false).unwrap(),
        a.sum_axis(1, false).unwrap(),
        a.matmul(&b).unwrap(),
        a.softmax(-1).unwrap(),
        img.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap(),
        a.var_axis(1, true, false).unwrap(),
    ];
    outs.iter().map(|t| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()).collect()
}

#[test]
fn results_are_bit_identical_across_thread_counts() {
    let reference = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
    for n in [2, 3, 8] {
        let got = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(run);
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(g, r, "op #{i} differs between 1 and {n} threads");
        }
    }
    // and repeated runs on the default pool agree too
    assert_eq!(run(), run());
}
