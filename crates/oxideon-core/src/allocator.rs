//! Size-Class Bucketed Caching Memory Allocator (`torch.cuda.memory` / `CachingAllocator`).
//!
//! Eliminates repeated heap/OS allocations during training & inference loops by rounding
//! buffer requests up to power-of-two size classes (`>= 64` elements) and recycling freed
//! buffers from a thread-safe bucket pool, while tracking live/cached/peak byte watermarks.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Telemetry snapshot from [`CachingAllocator`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocatorStats {
    pub active_bytes: usize,
    pub cached_bytes: usize,
    pub peak_active_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

#[derive(Default)]
struct PoolState {
    buckets: HashMap<usize, Vec<Vec<f32>>>,
    stats: AllocatorStats,
}

/// Thread-safe size-class bucketed caching allocator for tensor buffers.
pub struct CachingAllocator {
    state: Mutex<PoolState>,
}

static GLOBAL_ALLOCATOR: LazyLock<CachingAllocator> = LazyLock::new(CachingAllocator::new);

impl Default for CachingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CachingAllocator {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(PoolState::default()),
        }
    }

    /// Global singleton allocator instance.
    pub fn global() -> &'static Self {
        &GLOBAL_ALLOCATOR
    }

    #[inline]
    pub fn bucket_capacity(numel: usize) -> usize {
        // Saturates instead of panicking on overflow; `alloc_f32` validates first.
        numel
            .max(64)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
    }

    /// Allocates a zero-initialized `Vec<f32>` of length `numel` using the size-class cache.
    pub fn alloc_f32(&self, numel: usize) -> crate::error::Result<Vec<f32>> {
        crate::limits::check_at_most("allocator request", numel, crate::limits::MAX_ELEMENTS)?;
        let cap = Self::bucket_capacity(numel);
        let bytes = cap.saturating_mul(std::mem::size_of::<f32>());
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let maybe_buf = guard.buckets.get_mut(&cap).and_then(Vec::pop);
        let mut buf = if let Some(mut recycled) = maybe_buf {
            guard.stats.cached_bytes = guard.stats.cached_bytes.saturating_sub(bytes);
            guard.stats.cache_hits = guard.stats.cache_hits.saturating_add(1);
            recycled.clear();
            // A pooled buffer may be smaller than its bucket (failed upsize
            // during recycle); grow fallibly so OOM is an error, not an abort.
            recycled.try_reserve(numel).map_err(|_| {
                crate::error::Error::Msg(format!("allocator: cannot reserve {numel} f32 elements"))
            })?;
            recycled.resize(numel, 0.0);
            recycled
        } else {
            guard.stats.cache_misses = guard.stats.cache_misses.saturating_add(1);
            let mut v = Vec::new();
            v.try_reserve_exact(cap).map_err(|_| {
                crate::error::Error::Msg(format!("allocator: cannot reserve {cap} f32 elements"))
            })?;
            v.resize(numel, 0.0);
            v
        };
        let _ = &mut buf;
        guard.stats.active_bytes = guard.stats.active_bytes.saturating_add(bytes);
        if guard.stats.active_bytes > guard.stats.peak_active_bytes {
            guard.stats.peak_active_bytes = guard.stats.active_bytes;
        }
        Ok(buf)
    }

    /// Returns a `Vec<f32>` buffer to its power-of-two bucket for reuse without freeing memory.
    pub fn recycle_f32(&self, mut buf: Vec<f32>) {
        let cap = Self::bucket_capacity(buf.capacity().max(buf.len()));
        if buf.capacity() < cap {
            // Best-effort upsizing; a failed reserve still pools a usable buffer.
            let _ = buf.try_reserve_exact(cap - buf.capacity());
        }
        let bytes = cap.saturating_mul(std::mem::size_of::<f32>());
        buf.clear();
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.stats.active_bytes = guard.stats.active_bytes.saturating_sub(bytes);
        guard.stats.cached_bytes = guard.stats.cached_bytes.saturating_add(bytes);
        guard.buckets.entry(cap).or_default().push(buf);
    }

    /// Releases all cached buffers back to the OS (`torch.cuda.empty_cache()`).
    pub fn empty_cache(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.buckets.clear();
        guard.stats.cached_bytes = 0;
    }

    /// Resets peak memory statistics (`torch.cuda.reset_peak_memory_stats()`).
    pub fn reset_peak_stats(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.stats.peak_active_bytes = guard.stats.active_bytes;
    }

    /// Returns current allocator statistics (`torch.cuda.memory_stats()`).
    pub fn stats(&self) -> AllocatorStats {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_lock_does_not_brick_allocator() {
        let a = CachingAllocator::new();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = a.state.lock().unwrap();
            panic!("intentional poison");
        }));
        // pre-fix: every accessor below panicked on the poisoned lock
        let buf = a.alloc_f32(100).unwrap();
        assert_eq!(buf.len(), 100);
        a.recycle_f32(buf);
        a.empty_cache();
        a.reset_peak_stats();
        let _ = a.stats();
    }
}
