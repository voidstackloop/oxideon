//! Cross-validation batch 53: event files written by SummaryWriter, read back by TensorBoard's own event accumulator
//! (tools/crossval/check_batch53.py).
use oxideon_core::summary_writer::SummaryWriter;

#[test]
#[ignore]
fn emit_event_file() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tb_run");
    let _ = std::fs::remove_dir_all(dir);
    let mut w = SummaryWriter::new(dir).unwrap();
    for step in 0..20 {
        w.add_scalar("loss/train", 1.0 / (1.0 + step as f64), step).unwrap();
        w.add_scalars("metrics", &[("acc", step as f64 / 20.0), ("lr", 0.1)], step).unwrap();
    }
    let vals: Vec<f64> = (0..1000).map(|i| ((i as f64) * 0.37).sin() * 2.0 + 0.5).collect();
    w.add_histogram("weights", &vals, 30, 7).unwrap();
    w.flush().unwrap();
    println!("written {}", w.path().display());
}
