//! A tiny, seedable pseudo-random number generator (SplitMix64).
//!
//! Oxideon keeps a thread-local global generator so results are reproducible
//! after calling [`manual_seed`].

use crate::device::Device;
use crate::error::{Error, Result};
use std::cell::RefCell;

fn entropy_seed() -> u64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(123456789);
    let mut z = (nanos as u64) ^ 0x517C_C1B7_2722_0A95;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64 generator. Fast, small state, good statistical quality for ML
/// initialisation and shuffling (not cryptographically secure).
#[derive(Clone, Debug)]
pub struct Rng {
    pub(crate) state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform sample in `[0, 1)` with 64-bit precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.next_f32(); // (0, 1]
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> Result<usize> {
        if n == 0 {
            return Err(Error::Msg("Rng::below requires n > 0".into()));
        }
        Ok(self.below_unchecked(n))
    }

    /// Infallible `below` for call sites with a statically positive bound.
    fn below_unchecked(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "Rng::below_unchecked requires n > 0");
        (self.next_u64() % n as u64) as usize
    }

    /// In-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below_unchecked(i + 1);
            items.swap(i, j);
        }
    }
}

/// Standalone pseudo-random number generator (`torch.Generator`).
/// Manages independent PRNG streams per thread, task, device, or request.
#[derive(Clone, Debug)]
pub struct Generator {
    rng: Rng,
    initial_seed: u64,
    device: Device,
}

impl Default for Generator {
    fn default() -> Self {
        Self::new()
    }
}

impl Generator {
    /// Creates a new generator initialized with system entropy.
    pub fn new() -> Self {
        let s = entropy_seed();
        Self {
            rng: Rng::new(s),
            initial_seed: s,
            device: Device::Cpu,
        }
    }

    /// Creates a generator initialized with an explicit seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            initial_seed: seed,
            device: Device::Cpu,
        }
    }

    /// Creates a generator associated with a specific device.
    pub fn with_device(seed: u64, device: Device) -> Self {
        Self {
            rng: Rng::new(seed),
            initial_seed: seed,
            device,
        }
    }

    /// Re-seeds this generator (`generator.manual_seed(seed)`).
    pub fn manual_seed(&mut self, seed: u64) -> &mut Self {
        self.rng = Rng::new(seed);
        self.initial_seed = seed;
        self
    }

    /// Re-seeds this generator with entropy and returns the newly chosen seed (`generator.seed()`).
    pub fn seed(&mut self) -> u64 {
        let s = entropy_seed();
        self.manual_seed(s);
        s
    }

    /// Returns the initial seed that initialized or last reset this generator.
    pub fn initial_seed(&self) -> u64 {
        self.initial_seed
    }

    /// Returns the device assigned to this generator.
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    pub fn next_f32(&mut self) -> f32 {
        self.rng.next_f32()
    }

    pub fn next_f64(&mut self) -> f64 {
        self.rng.next_f64()
    }

    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.rng.uniform(low, high)
    }

    pub fn normal(&mut self) -> f32 {
        self.rng.normal()
    }

    pub fn below(&mut self, n: usize) -> Result<usize> {
        self.rng.below(n)
    }

    /// Exports serialized PRNG state bytes (`generator.get_state()`).
    pub fn get_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&self.rng.state.to_le_bytes());
        out.extend_from_slice(&self.initial_seed.to_le_bytes());
        out
    }

    /// Restores serialized PRNG state bytes (`generator.set_state(bytes)`).
    pub fn set_state(&mut self, state: &[u8]) -> Result<()> {
        if state.len() != 16 {
            return Err(Error::Msg(format!(
                "invalid RNG state length: expected exactly 16 bytes, got {}",
                state.len()
            )));
        }
        let rng_state = u64::from_le_bytes(state[0..8].try_into().unwrap());
        let init_s = u64::from_le_bytes(state[8..16].try_into().unwrap());
        self.rng.state = rng_state;
        self.initial_seed = init_s;
        Ok(())
    }
}

struct GlobalRngState {
    rng: Rng,
    initial_seed: u64,
}

thread_local! {
    static GLOBAL_RNG: RefCell<GlobalRngState> = RefCell::new(GlobalRngState {
        rng: Rng::new(0x0D1E_0000_5EED),
        initial_seed: 0x0D1E_0000_5EED,
    });
}

/// Re-seeds the thread-local global generator (`ox.manual_seed(seed)` / `torch.manual_seed(seed)`).
pub fn manual_seed(seed: u64) {
    GLOBAL_RNG.with(|r| {
        let mut cell = r.borrow_mut();
        cell.rng = Rng::new(seed);
        cell.initial_seed = seed;
    });
}

/// Returns the initial seed set on the global generator (`torch.initial_seed()`).
pub fn initial_seed() -> u64 {
    GLOBAL_RNG.with(|r| r.borrow().initial_seed)
}

/// Re-seeds the global generator with fresh system entropy and returns the chosen seed (`torch.seed()`).
pub fn seed() -> u64 {
    let s = entropy_seed();
    manual_seed(s);
    s
}

/// Returns the serialized state bytes of the global thread-local generator (`torch.get_rng_state()`).
pub fn get_rng_state() -> Vec<u8> {
    GLOBAL_RNG.with(|r| {
        let cell = r.borrow();
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&cell.rng.state.to_le_bytes());
        out.extend_from_slice(&cell.initial_seed.to_le_bytes());
        out
    })
}

/// Sets the serialized state bytes of the global thread-local generator (`torch.set_rng_state(state)`).
pub fn set_rng_state(state: &[u8]) -> Result<()> {
    if state.len() != 16 {
        return Err(Error::Msg(format!(
            "invalid RNG state length: expected exactly 16 bytes, got {}",
            state.len()
        )));
    }
    let rng_state = u64::from_le_bytes(state[0..8].try_into().unwrap());
    let init_s = u64::from_le_bytes(state[8..16].try_into().unwrap());
    GLOBAL_RNG.with(|r| {
        let mut cell = r.borrow_mut();
        cell.rng.state = rng_state;
        cell.initial_seed = init_s;
    });
    Ok(())
}

/// Restores the saved global RNG state on drop, so [`fork_rng`] restores it even
/// when the closure panics (like `torch.random.fork_rng`).
struct ForkRestoreGuard {
    state: Vec<u8>,
}

impl Drop for ForkRestoreGuard {
    fn drop(&mut self) {
        let _ = set_rng_state(&self.state);
    }
}

/// Executes a closure with an isolated RNG state, restoring the pre-execution RNG state upon exit (`torch.random.fork_rng`).
pub fn fork_rng<R>(f: impl FnOnce() -> R) -> R {
    let _guard = ForkRestoreGuard {
        state: get_rng_state(),
    };
    f()
}

/// Runs `f` with mutable access to the thread-local global generator.
pub fn with_rng<R>(f: impl FnOnce(&mut Rng) -> R) -> R {
    GLOBAL_RNG.with(|r| f(&mut r.borrow_mut().rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn rng_manual_seed_is_deterministic() {
        manual_seed(42);
        let a = Tensor::randn(&[2, 8]).unwrap().to_vec::<f32>().unwrap();
        manual_seed(42);
        let b = Tensor::randn(&[2, 8]).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(a, b);
        manual_seed(43);
        let c = Tensor::randn(&[2, 8]).unwrap().to_vec::<f32>().unwrap();
        assert_ne!(a, c);
        assert_eq!(initial_seed(), 43);
    }

    #[test]
    fn rng_state_roundtrip_and_fork() {
        manual_seed(7);
        let state = get_rng_state();
        let mid = Tensor::randn(&[4]).unwrap().to_vec::<f32>().unwrap();
        set_rng_state(&state).unwrap();
        let again = Tensor::randn(&[4]).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(mid, again);
        // forked draws do not advance the outer stream
        manual_seed(9);
        let outer: Vec<f32> = fork_rng(|| Tensor::randn(&[4]).unwrap().to_vec::<f32>().unwrap());
        let replay = Tensor::randn(&[4]).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(outer, replay);
        assert!(set_rng_state(&[0u8; 15]).is_err());
        assert!(set_rng_state(&[0u8; 17]).is_err());
    }

    #[test]
    fn rng_generator_streams() {
        let mut g1 = Generator::with_seed(123);
        let mut g2 = Generator::with_seed(123);
        assert_eq!(g1.next_u64(), g2.next_u64());
        assert_eq!(g1.initial_seed(), 123);
        g1.manual_seed(5);
        let state = g1.get_state();
        let a = g1.next_f64();
        g1.set_state(&state).unwrap();
        assert_eq!(g1.next_f64(), a);
        assert_eq!(g1.initial_seed(), 5);
        assert!(g1.set_state(&[0u8; 3]).is_err());
        // entropy seeds differ in practice and always reseed cleanly
        let s = g2.seed();
        assert_eq!(g2.initial_seed(), s);
    }

    #[test]
    fn rng_normal_finite_and_below_shuffle() {
        let mut g = Generator::with_seed(2024);
        let mut sum = 0.0f64;
        for _ in 0..10_000 {
            let v = g.normal();
            assert!(v.is_finite());
            sum += v as f64;
        }
        assert!(sum.abs() / 10_000.0 < 0.1, "mean drift: {}", sum / 10_000.0);
        assert!(g.below(0).is_err());
        for _ in 0..100 {
            assert!(g.below(7).unwrap() < 7);
        }
        // shuffle is a deterministic permutation
        let mut v: Vec<i32> = (0..50).collect();
        Rng::new(3).shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        let mut w: Vec<i32> = (0..50).collect();
        Rng::new(3).shuffle(&mut w);
        assert_eq!(v, w);
    }
}
