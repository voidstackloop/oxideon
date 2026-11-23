use oxideon_core::backend::{cuda_ops::download_count, cuda_rt};
use oxideon_core::nn::*;
use oxideon_core::optim::{Adam, Optimizer};
use oxideon_core::{Device, Tensor};
fn rnd(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}

#[test]
fn conv_bn_attention_and_adam_training_steps_never_touch_the_host() {
    if !cuda_rt::available() { return; }
    let d = Device::Cuda(0);
    // CNN
    let conv = Conv2d::new(3, 8, (3, 3), (1, 1), (1, 1)).unwrap();
    let bn = BatchNorm2d::new(8).unwrap();
    let fc = Linear::new(8 * 4 * 4, 5).unwrap();
    // transformer-ish
    let emb = Embedding::new(50, 16).unwrap();
    let mha = MultiHeadAttention::new(16, 4).unwrap();
    let ln = LayerNorm::new(16, 1e-5).unwrap();
    let head = Linear::new(16, 50).unwrap();
    let mut all = vec![]; conv.to_device(&d).unwrap(); bn.to_device(&d).unwrap(); fc.to_device(&d).unwrap(); emb.to_device(&d).unwrap(); mha.to_device(&d).unwrap(); ln.to_device(&d).unwrap(); head.to_device(&d).unwrap();
    for m in [conv.parameters(), bn.parameters(), fc.parameters(), emb.parameters(), mha.parameters(), ln.parameters(), head.parameters()] { all.extend(m); }
    let mut opt = Adam::new(all, 1e-3);
    let img = rnd(&[4, 3, 8, 8]).to_device(&d).unwrap();
    let ids = Tensor::from_slice(&(0..24).map(|i| (i * 7 % 50) as i64).collect::<Vec<_>>(), &[4, 6]).unwrap().to_device(&d).unwrap();
    for step in 0..3 {
        opt.zero_grad();
        let b0 = download_count();
        let h = bn.forward(&conv.forward(&img).unwrap()).unwrap().relu().unwrap().max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        let l1 = fc.forward(&h.flatten_range(1, -1).unwrap()).unwrap().log_softmax(-1).unwrap().mean().unwrap().neg().unwrap();
        l1.backward().unwrap();
        let b1 = download_count();
        let e = emb.forward(&ids).unwrap();
        let a = mha.forward_qkv(&e, &e, &e, true).unwrap();
        let o = head.forward(&ln.forward(&a.add(&e).unwrap()).unwrap()).unwrap().log_softmax(-1).unwrap().mean().unwrap().neg().unwrap();
        o.backward().unwrap();
        let b2 = download_count();
        opt.step().unwrap();
        let b3 = download_count();
        assert_eq!((b1 - b0, b2 - b1, b3 - b2), (0, 0, 0), "step {step} touched the host");
    }
}
