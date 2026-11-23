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
    // every change of the limit and of the accounting happens under the `f32` shelf lock, so a `give` racing with this call sees
    // either the old limit before it or the new one after it, never a stale one
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    LIMIT.store(bytes, Relaxed);
    if held_bytes() > bytes {
        clear_locked(&mut g);
    }
}

/// Frees every pooled buffer.
pub fn clear() {
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    clear_locked(&mut g);
}

fn clear_locked(g: &mut Shelves) {
    g.by_len.clear();
    // the typed shelves are emptied and the count zeroed under their own lock, so a typed `take` (which only holds that lock)
    // either runs before this and has already subtracted its bytes, or after it and finds nothing
    let mut typed = typed_shelves().lock().unwrap_or_else(|p| p.into_inner());
    typed.clear();
    HELD.store(0, Relaxed);
}

/// Offers a buffer that is no longer needed.
pub fn give(v: Vec<f32>) {
    let len = v.len();
    if len < MIN_LEN {
        return;
    }
    let mut g = shelves().lock().unwrap_or_else(|p| p.into_inner());
    let limit = LIMIT.load(Relaxed);
    if len * 4 > limit {
        return;
    }
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

/// Shelf of one non-`f32` element type and length: `(last use, element size in bytes, buffers)`.
type TypedShelf = (u64, usize, Vec<Box<dyn std::any::Any + Send>>);
type TypedShelves = HashMap<(std::any::TypeId, usize), TypedShelf>;

/// Typed shelves for non-`f32` dtypes (`f64`, `i64`, `i32`, `u8`, `u16`, etc.). They share the byte budget of the `f32`
/// pool: their bytes are part of [`held_bytes`] and the least recently used lengths are evicted to stay within the limit.
fn typed_shelves() -> &'static Mutex<TypedShelves> {
    static TYPED: std::sync::OnceLock<Mutex<TypedShelves>> = std::sync::OnceLock::new();
    TYPED.get_or_init(|| Mutex::new(HashMap::new()))
}

static TYPED_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Offers a typed buffer of any `'static + Send` element type (`f32`, `f64`, `i32`, `i64`, `u8`, `u16`, ...).
pub fn give_typed<T: 'static + Send>(v: Vec<T>) {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let boxed: Box<dyn std::any::Any> = Box::new(v);
        if let Ok(vf32) = boxed.downcast::<Vec<f32>>() {
            give(*vf32);
        }
        return;
    }
    let len = v.len();
    let elem = std::mem::size_of::<T>();
    let byte_len = len.saturating_mul(elem);
    if len < MIN_LEN || elem == 0 {
        return;
    }
    // lock order: the f32 shelves first (they own the shared byte accounting), then the typed ones
    let _accounting = shelves().lock().unwrap_or_else(|p| p.into_inner());
    let limit = LIMIT.load(Relaxed);
    if byte_len > limit {
        return;
    }
    let key = (std::any::TypeId::of::<T>(), len);
    let mut g = typed_shelves().lock().unwrap_or_else(|p| p.into_inner());
    if g.get(&key).is_some_and(|(_, _, shelf)| shelf.len() >= PER_LEN) {
        return;
    }
    // make room by dropping the typed lengths that have not been used for the longest time
    while HELD.load(Relaxed) + byte_len > limit {
        let Some(victim) = g.iter().filter(|(k, (_, _, shelf))| **k != key && !shelf.is_empty()).min_by_key(|(_, (used, _, _))| *used).map(|(k, _)| *k) else {
            return; // only f32 buffers (or nothing else) are left to evict: skip this one rather than exceed the limit
        };
        if let Some((_, victim_elem, shelf)) = g.remove(&victim) {
            HELD.fetch_sub(shelf.len() * victim.1 * victim_elem, Relaxed);
        }
    }
    let entry = g.entry(key).or_insert((0, elem, Vec::new()));
    entry.0 = TYPED_CLOCK.fetch_add(1, Relaxed) + 1;
    entry.2.push(Box::new(v));
    HELD.fetch_add(byte_len, Relaxed);
}

/// Takes a pooled buffer of element type `T` and exact length `len`.
pub fn take<T: 'static>(len: usize) -> Option<Vec<T>> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let boxed: Box<dyn std::any::Any> = Box::new(take_f32(len)?);
        return boxed.downcast::<Vec<T>>().ok().map(|b| *b);
    }
    if len < MIN_LEN {
        return None;
    }
    // only the typed lock: a take just subtracts what it removes, so it cannot overshoot the limit, and `clear` zeroes the count under
    // this same lock
    let mut g = typed_shelves().lock().unwrap_or_else(|p| p.into_inner());
    let entry = g.get_mut(&(std::any::TypeId::of::<T>(), len))?;
    let boxed = entry.2.pop()?;
    entry.0 = TYPED_CLOCK.fetch_add(1, Relaxed) + 1;
    HELD.fetch_sub(len * entry.1, Relaxed);
    boxed.downcast::<Vec<T>>().ok().map(|b| *b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // the pool is global and one test below changes its limit, so the pool tests must not overlap
    static POOL_TESTS: Mutex<()> = Mutex::new(());

    /// Several threads give, take and re-limit f32 and typed buffers at once: the byte count must never wrap around (a typed take racing a
    /// `clear`) or sit above the limit once the dust settles (a give racing a lowered limit).
    #[test]
    fn concurrent_give_take_and_limit_changes_keep_the_accounting_sound() {
        let _g = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let old_limit = LIMIT.load(Relaxed);
        clear();
        let (high, low) = (12 * MIN_LEN * 8, 5 * MIN_LEN * 8);
        set_limit_bytes(high);
        std::thread::scope(|scope| {
            // one thread keeps lowering and raising the limit while the others give and take
            scope.spawn(|| {
                for i in 0..4_000usize {
                    set_limit_bytes(if i % 2 == 0 { low } else { high });
                }
                set_limit_bytes(low);
            });
            for t in 1..6usize {
                scope.spawn(move || {
                    for i in 0..4_000usize {
                        let len = MIN_LEN + (i + t) % 5;
                        match (i + t) % 4 {
                            0 | 1 => give(vec![0.0f32; len]),
                            2 => give_typed(vec![0.0f64; len]),
                            _ => drop(take::<f64>(len)),
                        }
                        assert!(held_bytes() < 1 << 40, "the byte count wrapped around: {}", held_bytes());
                    }
                });
            }
        });
        // no further `set_limit_bytes` here: a late give that overshot the final limit must still be visible
        assert!(held_bytes() <= low, "pool holds {} bytes over a limit of {low}", held_bytes());
        clear();
        assert_eq!(held_bytes(), 0);
        set_limit_bytes(old_limit);
    }

    #[test]
    fn buffers_are_recycled_by_length_only_for_f32_and_within_the_limit() {
        let _g = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
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

    #[test]
    fn typed_buffers_are_recycled_for_non_f32_dtypes() {
        let _g = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let n = MIN_LEN + 23_456;
        give_typed(vec![42i64; n]);
        give_typed(vec![3.25f64; n]);
        give_typed(vec![9u8; n]);
        assert_eq!(take::<i64>(n).unwrap()[0], 42);
        assert_eq!(take::<f64>(n).unwrap()[0], 3.25);
        assert_eq!(take::<u8>(n).unwrap()[0], 9);
        assert!(take::<i64>(n).is_none());
    }

    #[test]
    fn typed_buffers_stay_inside_the_global_byte_limit() {
        let _g = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let old_limit = LIMIT.load(Relaxed);
        clear();
        let budget = 6 * MIN_LEN * 8; // room for about six u64 buffers
        set_limit_bytes(budget);
        // many distinct lengths: without accounting they would all be kept
        for i in 0..40 {
            give_typed(vec![0u64; MIN_LEN + i]);
            assert!(held_bytes() <= budget, "pool holds {} bytes over a budget of {budget}", held_bytes());
        }
        assert!(held_bytes() > 0, "recent buffers are still recycled");
        // the most recently given length is still there, the oldest was evicted
        assert!(take::<u64>(MIN_LEN + 39).is_some());
        assert!(take::<u64>(MIN_LEN).is_none());
        // taking everything back leaves the books balanced
        for i in 0..40 {
            let _ = take::<u64>(MIN_LEN + i);
        }
        assert_eq!(held_bytes(), 0);
        set_limit_bytes(old_limit);
        clear();
    }
}
