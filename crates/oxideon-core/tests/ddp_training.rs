//! Data-parallel SGD over a process group must reproduce single-process full-batch SGD (same weights after several steps),
//! over the in-process transport and over TCP sockets.
use oxideon_core::comm::{DistributedDataParallel, ProcessGroup};
use oxideon_core::tensor::{Tensor, no_grad};
use std::net::TcpListener;

const STEPS: usize = 8;
const LR: f64 = 0.1;

fn dataset() -> (Tensor, Tensor) {
    let x: Vec<f64> = (0..48 * 3).map(|i| ((i * 37 % 101) as f64) / 50.0 - 1.0).collect();
    let y: Vec<f64> = (0..48).map(|r| 0.5 * x[r * 3] - 0.3 * x[r * 3 + 1] + 0.2 * x[r * 3 + 2] + 0.1).collect();
    (Tensor::from_slice(&x, &[48, 3]).unwrap(), Tensor::from_slice(&y, &[48, 1]).unwrap())
}

fn init() -> (Tensor, Tensor) {
    (Tensor::from_slice(&[0.2f64, -0.1, 0.05], &[3, 1]).unwrap().requires_grad_(true), Tensor::from_slice(&[0.0f64], &[1]).unwrap().requires_grad_(true))
}

fn step(w: &Tensor, b: &Tensor, x: &Tensor, y: &Tensor, sync: impl Fn()) {
    w.zero_grad();
    b.zero_grad();
    let loss = x.matmul(w).unwrap().add(b).unwrap().sub(y).unwrap().square().unwrap().mean().unwrap();
    loss.backward().unwrap();
    sync();
    no_grad(|| {
        for p in [w, b] {
            p.assign(&p.sub(&p.grad().unwrap().mul_scalar(LR).unwrap()).unwrap()).unwrap();
        }
    });
}

fn serial() -> Vec<f64> {
    let (x, y) = dataset();
    let (w, b) = init();
    for _ in 0..STEPS {
        step(&w, &b, &x, &y, || {});
    }
    [w.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap()].concat()
}

fn rank_work(pg: ProcessGroup) -> Vec<f64> {
    let (rank, world) = (pg.rank(), pg.world_size());
    let (x, y) = dataset();
    let per = 48 / world;
    let (xs, ys) = (x.narrow(0, rank * per, per).unwrap().contiguous().unwrap(), y.narrow(0, rank * per, per).unwrap().contiguous().unwrap());
    // every rank starts from different (wrong) weights: DDP must broadcast rank 0's
    let (w, b) = init();
    if rank != 0 {
        no_grad(|| w.assign(&Tensor::from_slice(&[9.0f64, 9.0, 9.0], &[3, 1]).unwrap()).unwrap());
    }
    let ddp = DistributedDataParallel::new(vec![w.clone(), b.clone()], pg, 16).unwrap();
    for _ in 0..STEPS {
        step(&w, &b, &xs, &ys, || ddp.sync_gradients().unwrap());
    }
    [w.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap()].concat()
}

#[test]
fn local_data_parallel_matches_serial_training() {
    let want = serial();
    for world in [2usize, 3, 4] {
        for got in ProcessGroup::run_local(world, rank_work) {
            assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-9), "world {world}: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn tcp_data_parallel_matches_serial_training() {
    let want = serial();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let addr = format!("127.0.0.1:{port}");
    let handles: Vec<_> = (0..4).map(|r| {
        let addr = addr.clone();
        std::thread::spawn(move || rank_work(ProcessGroup::tcp(r, 4, &addr).unwrap()))
    }).collect();
    for h in handles {
        let got = h.join().unwrap();
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-9), "{got:?} vs {want:?}");
    }
}

/// Reference AdamW (decoupled decay) on the full batch.
fn serial_adamw(steps: usize, lr: f64, wd: f64) -> Vec<f64> {
    let (x, y) = dataset();
    let (w, b) = init();
    let mut state: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![0.0; 3], vec![0.0; 3]), (vec![0.0], vec![0.0])];
    for t in 1..=steps {
        w.zero_grad();
        b.zero_grad();
        x.matmul(&w).unwrap().add(&b).unwrap().sub(&y).unwrap().square().unwrap().mean().unwrap().backward().unwrap();
        for (p, (m, v)) in [&w, &b].into_iter().zip(state.iter_mut()) {
            let g = p.grad().unwrap().to_vec::<f64>().unwrap();
            let mut x = p.to_vec::<f64>().unwrap();
            for i in 0..x.len() {
                m[i] = 0.9 * m[i] + 0.1 * g[i];
                v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
                x[i] -= lr * ((m[i] / (1.0 - 0.9f64.powi(t as i32))) / ((v[i] / (1.0 - 0.999f64.powi(t as i32))).sqrt() + 1e-8) + wd * x[i]);
            }
            no_grad(|| p.assign(&Tensor::from_slice(&x, p.shape()).unwrap()).unwrap());
        }
    }
    [w.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap()].concat()
}

#[test]
fn sharded_adamw_matches_full_adamw_and_shards_the_state() {
    use oxideon_core::comm::ShardedAdamW;
    let want = serial_adamw(6, 0.05, 0.01);
    for world in [2usize, 3] {
        let res = ProcessGroup::run_local(world, move |pg| {
            let (rank, world) = (pg.rank(), pg.world_size());
            let (x, y) = dataset();
            let per = 48 / world;
            let (xs, ys) = (x.narrow(0, rank * per, per).unwrap().contiguous().unwrap(), y.narrow(0, rank * per, per).unwrap().contiguous().unwrap());
            let (w, b) = init();
            let mut opt = ShardedAdamW::new(vec![w.clone(), b.clone()], pg, 0.05, 0.01).unwrap();
            for _ in 0..6 {
                opt.zero_grad();
                xs.matmul(&w).unwrap().add(&b).unwrap().sub(&ys).unwrap().square().unwrap().mean().unwrap().backward().unwrap();
                opt.step().unwrap();
            }
            ([w.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap()].concat(), opt.state_elements())
        });
        for (got, state) in res {
            assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-9), "world {world}: {got:?} vs {want:?}");
            assert_eq!(state, 2 * 4usize.div_ceil(world)); // 2 moment buffers over a 4-element parameter set
        }
    }
}
