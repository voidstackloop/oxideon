//! Operator profiler and anomaly detection.
//!
//! [`profile`] (or [`start`] / [`stop`]) records every backend operation (unary / binary / comparison / reduction / matmul /
//! dtype and device conversions) with its wall time, thread and input shape, plus user ranges opened with [`record_function`].
//! A [`Profile`] prints an aggregated table (like `torch.profiler`'s `key_averages().table()`) and exports a Chrome trace
//! (`chrome://tracing`, Perfetto). When profiling is off the hooks cost one atomic load.
//!
//! [`set_detect_anomaly`] makes the library remember the first operation whose result contained a NaN or infinity
//! ([`first_non_finite`]), the tool for finding where a training run first diverged.

use crate::error::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ANOMALY: AtomicBool = AtomicBool::new(false);

struct State {
    origin: Instant,
    events: Vec<Event>,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(State { origin: Instant::now(), events: Vec::new() }))
}

type Offender = Option<(String, Vec<usize>)>;

fn first_bad() -> &'static Mutex<Offender> {
    static S: OnceLock<Mutex<Offender>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// One recorded operation or user range.
#[derive(Debug, Clone)]
pub struct Event {
    pub name: String,
    pub category: &'static str,
    pub thread: u64,
    /// Start, in microseconds since the profile began.
    pub start_us: f64,
    pub dur_us: f64,
    pub shape: Vec<usize>,
}

fn thread_id() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    h.finish() % 100_000
}

/// RAII timer returned by the hooks; records its event when dropped.
pub struct Scope {
    name: String,
    category: &'static str,
    shape: Vec<usize>,
    start: Instant,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !ACTIVE.load(Ordering::Relaxed) {
            return; // the profile was stopped while this operation was running
        }
        let end = Instant::now();
        let mut s = state().lock().unwrap();
        let start_us = self.start.saturating_duration_since(s.origin).as_secs_f64() * 1e6;
        let dur_us = end.duration_since(self.start).as_secs_f64() * 1e6;
        s.events.push(Event { name: std::mem::take(&mut self.name), category: self.category, thread: thread_id(), start_us, dur_us, shape: std::mem::take(&mut self.shape) });
    }
}

/// The hook the backend calls at the start of an operation (`None`, and free, when no profile is running).
#[inline]
pub fn scope(name: &'static str, shape: &[usize]) -> Option<Scope> {
    if !ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    Some(Scope { name: name.to_string(), category: "op", shape: shape.to_vec(), start: Instant::now() })
}

/// A named range of user code (a layer, a training phase) shown next to the operations it contains.
pub fn record_function(name: &str) -> Option<Scope> {
    if !ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    Some(Scope { name: name.to_string(), category: "user", shape: Vec::new(), start: Instant::now() })
}

/// Starts recording (discarding any earlier events).
pub fn start() {
    let mut s = state().lock().unwrap();
    s.origin = Instant::now();
    s.events.clear();
    ACTIVE.store(true, Ordering::SeqCst);
}

/// Stops recording and returns what was captured.
pub fn stop() -> Profile {
    ACTIVE.store(false, Ordering::SeqCst);
    let mut s = state().lock().unwrap();
    let mut events = std::mem::take(&mut s.events);
    events.sort_by(|a, b| a.start_us.partial_cmp(&b.start_us).unwrap_or(std::cmp::Ordering::Equal));
    Profile { events }
}

/// Runs `f` under the profiler.
pub fn profile<R>(f: impl FnOnce() -> R) -> (R, Profile) {
    start();
    let r = f();
    (r, stop())
}

pub fn is_profiling() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Aggregated statistics of one operation name.
#[derive(Debug, Clone)]
pub struct OpStats {
    pub name: String,
    pub calls: usize,
    pub total_us: f64,
    pub mean_us: f64,
    pub max_us: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub events: Vec<Event>,
}

impl Profile {
    /// Per-operation totals, slowest first. User ranges are listed too (their time includes the operations inside them).
    pub fn key_averages(&self) -> Vec<OpStats> {
        let mut m: HashMap<&str, (usize, f64, f64)> = HashMap::new();
        for e in &self.events {
            let x = m.entry(&e.name).or_insert((0, 0.0, 0.0));
            x.0 += 1;
            x.1 += e.dur_us;
            x.2 = x.2.max(e.dur_us);
        }
        let mut v: Vec<OpStats> = m.into_iter().map(|(n, (c, t, mx))| OpStats { name: n.to_string(), calls: c, total_us: t, mean_us: t / c as f64, max_us: mx }).collect();
        v.sort_by(|a, b| b.total_us.partial_cmp(&a.total_us).unwrap_or(std::cmp::Ordering::Equal).then(a.name.cmp(&b.name)));
        v
    }

    /// Sum of the time spent in backend operations (user ranges excluded; nested operations are counted once each).
    pub fn total_op_time_us(&self) -> f64 {
        self.events.iter().filter(|e| e.category == "op").map(|e| e.dur_us).sum()
    }

    /// A fixed-width table of [`key_averages`](Self::key_averages).
    pub fn table(&self) -> String {
        let total = self.total_op_time_us().max(1e-9);
        let mut out = format!("{:<24} {:>8} {:>12} {:>12} {:>12} {:>7}\n", "name", "calls", "total (us)", "mean (us)", "max (us)", "% ops");
        for s in self.key_averages() {
            out.push_str(&format!("{:<24} {:>8} {:>12.1} {:>12.2} {:>12.1} {:>6.1}%\n", s.name, s.calls, s.total_us, s.mean_us, s.max_us, 100.0 * s.total_us / total));
        }
        out
    }

    /// Chrome trace event format (open in `chrome://tracing` or Perfetto).
    pub fn chrome_trace(&self) -> String {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let items: Vec<String> = self
            .events
            .iter()
            .map(|e| {
                format!(
                    "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"X\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":{},\"args\":{{\"shape\":\"{:?}\"}}}}",
                    esc(&e.name),
                    e.category,
                    e.start_us,
                    e.dur_us,
                    e.thread,
                    e.shape
                )
            })
            .collect();
        format!("{{\"traceEvents\":[{}],\"displayTimeUnit\":\"ms\"}}", items.join(","))
    }

    pub fn save_chrome_trace(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path.as_ref(), self.chrome_trace()).map_err(|e| crate::error::Error::Msg(format!("profiler: cannot write {}: {e}", path.as_ref().display())))
    }
}

// ------------------------------------------------------------------------------------------------ anomaly detection

/// Turns the NaN / infinity watch on or off. While on, every operation result created through autograd is checked and the
/// first offender is remembered.
pub fn set_detect_anomaly(on: bool) {
    ANOMALY.store(on, Ordering::SeqCst);
    if on {
        *first_bad().lock().unwrap() = None;
    }
}

#[inline]
pub(crate) fn anomaly_enabled() -> bool {
    ANOMALY.load(Ordering::Relaxed)
}

pub(crate) fn note_non_finite(op: &str, shape: &[usize]) {
    let mut g = first_bad().lock().unwrap();
    if g.is_none() {
        *g = Some((op.to_string(), shape.to_vec()));
    }
}

/// The operation (and its output shape) that first produced a NaN or infinity since [`set_detect_anomaly`]`(true)`.
pub fn first_non_finite() -> Option<(String, Vec<usize>)> {
    first_bad().lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::tensor::Tensor;

    // the profiler and anomaly flags are global: run these tests one at a time
    static ONE: Mutex<()> = Mutex::new(());

    #[test]
    fn records_operations_user_ranges_and_exports_a_trace() {
        let _g = ONE.lock().unwrap_or_else(|p| p.into_inner());
        let a = Tensor::randn(&[64, 64]).unwrap();
        let (_, prof) = profile(|| {
            let _r = record_function("my_block");
            let y = a.matmul(&a).unwrap().relu().unwrap().add(&a).unwrap();
            y.sum().unwrap()
        });
        let names: Vec<String> = prof.key_averages().into_iter().map(|s| s.name).collect();
        for want in ["matmul", "unary_op", "binary_op", "reduce_op", "my_block"] {
            assert!(names.iter().any(|n| n == want), "{want} missing from {names:?}");
        }
        assert!(prof.total_op_time_us() > 0.0);
        assert!(prof.table().contains("matmul"));
        let trace = prof.chrome_trace();
        assert!(trace.starts_with("{\"traceEvents\":[") && trace.contains("\"ph\":\"X\"") && trace.contains("my_block"));
        // nothing is recorded once stopped
        a.matmul(&a).unwrap();
        assert!(!is_profiling());
    }

    #[test]
    fn anomaly_detection_names_the_first_non_finite_operation() {
        let _g = ONE.lock().unwrap_or_else(|p| p.into_inner());
        set_detect_anomaly(true);
        let x = Tensor::from_slice(&[1.0f32, 0.0, 2.0], &[3]).unwrap().requires_grad_(true);
        let _ = x.exp().unwrap().sum().unwrap(); // finite
        assert!(first_non_finite().is_none());
        let bad = x.recip().unwrap(); // 1 / 0 = inf
        let _ = bad.mul_scalar(2.0).unwrap();
        let (op, shape) = first_non_finite().expect("the infinity should have been noticed");
        assert!(op.contains("recip") || op.contains("div"), "{op}");
        assert_eq!(shape, vec![3]);
        set_detect_anomaly(false);
        let _ = DType::F32;
    }
}
