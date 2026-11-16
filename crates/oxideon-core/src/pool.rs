//! Recycling of large `f32` buffers. A fresh multi-megabyte allocation is handed out by the operating system as untouched
//! pages, and faulting them in costs several times more than the arithmetic that fills them (measured: 595 us for a new
//! 4 MB result against 132 us into a warm buffer). Storage dropped by a tensor is kept here and handed to the next kernel
//! that produces a result of the same length.
//!
//! The pool is bounded ([`set_limit_bytes`], default 256 MiB, at most [`PER_LEN`] buffers per length); when it is full the
//! lengths that have gone unused the longest are evicted, so a program that moves on to other tensor sizes keeps
//! recycling. [`clear`] empties it. Recycled buffers still hold their old values, so every kernel that takes one
//! overwrites all of it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Buffers shorter than this (elements) are not worth pooling: small allocations come from warm heap pages anyway.
pub const MIN_LEN: usize = 32 * 1024;
/// Buffers kept per distinct length.
pub const PER_LEN: usize = 16;

static LIMIT: AtomicUsize = AtomicUsize::new(256 << 20);
static HELD: AtomicUsize = AtomicUsize::new(0);
static HITS: AtomicUsize = AtomicUsize::new(0);
static MISSES: AtomicUsize = AtomicUsize::new(0);

/// `(hits, misses)`: requests for a large buffer that were served from the pool / had to allocate.
pub fn hit_stats() -> (usize, usize) {
    (HITS.load(Relaxed), MISSES.load(Relaxed))
}

#[derive(Default)]
struct Shelves {
    by_len: HashMap<usize, (u64, Vec<Vec<f32>>)>, // length -> (last use, buffers)
    clock: u64,
}

fn shelves() -> &'static Mutex<Shelves> {
    static S: std::sync::OnceLock<Mutex<Shelves>> = std::sync::OnceLock::new();
    S.get_or_init(|| Mutex::new(Shelves::default()))
}

/// Bytes currently parked in the pool.
pub fn held_bytes() -> usize {
    HELD.load(Relaxed)
}

/// Caps the pool (0 turns recycling off and frees what is held).
pub fn set_limit_bytes(bytes: usize) {
    LIMIT.store(bytes, Relaxed);
    if held_bytes() > bytes {
        clear();
    }
}

/// Frees every pooled buffer.
pub fn clear() {
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    g.by_len.clear();
    HELD.store(0, Relaxed);
}

/// Offers a buffer that is no longer needed.
pub fn give(v: Vec<f32>) {
    let len = v.len();
    let limit = LIMIT.load(Relaxed);
    if len < MIN_LEN || len * 4 > limit {
        return;
    }
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    g.clock += 1;
    let now = g.clock;
    if g.by_len.get(&len).is_some_and(|(_, shelf)| shelf.len() >= PER_LEN) {
        return;
    }
    // make room by dropping the lengths that have not been used for the longest time
    while HELD.load(Relaxed) + len * 4 > limit {
        let Some(victim) = g.by_len.iter().filter(|(l, (_, shelf))| **l != len && !shelf.is_empty()).min_by_key(|(_, (used, _))| *used).map(|(l, _)| *l) else {
            return;
        };
        if let Some((_, shelf)) = g.by_len.remove(&victim) {
            HELD.fetch_sub(shelf.len() * victim * 4, Relaxed);
        }
    }
    let entry = g.by_len.entry(len).or_default();
    entry.0 = now;
    entry.1.push(v);
    HELD.fetch_add(len * 4, Relaxed);
}

/// A warm buffer of exactly `len` elements holding arbitrary old values, if one is pooled.
pub fn take_f32(len: usize) -> Option<Vec<f32>> {
    if len < MIN_LEN {
        return None;
    }
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    g.clock += 1;
    let now = g.clock;
    let got = g.by_len.get_mut(&len).and_then(|entry| {
        entry.0 = now;
        entry.1.pop()
    });
    match &got {
        Some(_) => {
            HELD.fetch_sub(len * 4, Relaxed);
            HITS.fetch_add(1, Relaxed);
        }
        None => {
            MISSES.fetch_add(1, Relaxed);
        }
    }
    got
}

/// [`take_f32`] for generic kernels: `Some` only when `T` is `f32`.
pub fn take<T: 'static>(len: usize) -> Option<Vec<T>> {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return None;
    }
    let boxed: Box<dyn std::any::Any> = Box::new(take_f32(len)?);
    boxed.downcast::<Vec<T>>().ok().map(|b| *b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_recycled_by_length_only_for_f32_and_within_the_limit() {
        // a length no other test uses, so concurrent tests cannot take or add these buffers
        let n = MIN_LEN + 12_345;
        assert!(take_f32(n).is_none());
        give(vec![7.0f32; n]);
        give(vec![0.0f32; 10]); // too small: ignored
        assert!(take::<f64>(n).is_none());
        let v: Vec<f32> = take(n).expect("the buffer given above");
        assert_eq!((v.len(), v[0]), (n, 7.0));
        assert!(take_f32(n).is_none());
        for _ in 0..PER_LEN + 3 {
            give(vec![1.0f32; n]);
        }
        let mut got = 0;
        while take_f32(n).is_some() {
            got += 1;
        }
        assert_eq!(got, PER_LEN);
    }
}
