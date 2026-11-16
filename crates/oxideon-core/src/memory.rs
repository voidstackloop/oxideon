//! Allocator statistics: bytes of tensor storage currently alive, the peak since the last reset, and allocation counts
//! (the CPU analogue of `torch.cuda.memory_allocated` / `max_memory_allocated`). Views share their base storage, so only
//! real buffers count; a buffer's bytes are returned when its last handle is dropped.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BUFFERS: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn on_alloc(bytes: usize) {
    let now = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(now, Relaxed);
    ALLOCS.fetch_add(1, Relaxed);
    LIVE_BUFFERS.fetch_add(1, Relaxed);
}

pub(crate) fn on_free(bytes: usize) {
    LIVE.fetch_sub(bytes, Relaxed);
    LIVE_BUFFERS.fetch_sub(1, Relaxed);
}

/// A snapshot of the counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub live_buffers: usize,
    pub total_allocations: usize,
}

pub fn stats() -> MemoryStats {
    MemoryStats { allocated_bytes: LIVE.load(Relaxed), peak_bytes: PEAK.load(Relaxed), live_buffers: LIVE_BUFFERS.load(Relaxed), total_allocations: ALLOCS.load(Relaxed) }
}

/// Bytes of tensor storage alive right now.
pub fn allocated_bytes() -> usize {
    LIVE.load(Relaxed)
}

/// Highest [`allocated_bytes`] since startup or the last [`reset_peak`].
pub fn peak_bytes() -> usize {
    PEAK.load(Relaxed)
}

/// Restarts peak tracking from the current usage.
pub fn reset_peak() {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

// ------------------------------------------------------------------------------------------------ pinned memory

#[cfg(windows)]
unsafe extern "system" {
    fn VirtualLock(addr: *mut std::ffi::c_void, size: usize) -> i32;
    fn VirtualUnlock(addr: *mut std::ffi::c_void, size: usize) -> i32;
}
#[cfg(unix)]
unsafe extern "C" {
    fn mlock(addr: *const std::ffi::c_void, len: usize) -> i32;
    fn munlock(addr: *const std::ffi::c_void, len: usize) -> i32;
}

const PAGE: usize = 4096;

/// A page-aligned, page-locked (`VirtualLock` / `mlock`) host buffer of `f32`: the OS will not page it out, which keeps
/// uploads to a device and latency-sensitive loops free of page faults (`pin_memory` in other frameworks).
///
/// Locking can be refused (working-set quota on Windows, `RLIMIT_MEMLOCK` on Unix); the buffer then still works and
/// [`is_locked`](Self::is_locked) reports `false`.
pub struct PinnedBuffer {
    ptr: std::ptr::NonNull<f32>,
    len: usize,
    layout: std::alloc::Layout,
    locked: bool,
}

// SAFETY: the buffer owns its allocation exclusively, like a `Vec<f32>`.
unsafe impl Send for PinnedBuffer {}
// SAFETY: shared access only hands out `&[f32]`.
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// A zero-filled buffer of `len` values.
    pub fn new(len: usize) -> crate::error::Result<Self> {
        crate::limits::check_elements("pinned buffer", &[len])?;
        let bytes = (len.max(1) * 4).div_ceil(PAGE) * PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).map_err(|e| crate::error::Error::Msg(format!("pinned buffer: {e}")))?;
        // SAFETY: `layout` has a non-zero size.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw as *mut f32).ok_or_else(|| crate::error::Error::Msg("pinned buffer: allocation failed".into()))?;
        // SAFETY: `raw` points to `bytes` bytes owned by this buffer.
        #[cfg(windows)]
        let locked = unsafe { VirtualLock(raw as *mut std::ffi::c_void, bytes) != 0 };
        // SAFETY: as above.
        #[cfg(unix)]
        let locked = unsafe { mlock(raw as *const std::ffi::c_void, bytes) == 0 };
        #[cfg(not(any(windows, unix)))]
        let locked = false;
        Ok(Self { ptr, len, layout, locked })
    }

    /// A pinned copy of `data`.
    pub fn from_slice(data: &[f32]) -> crate::error::Result<Self> {
        let mut b = Self::new(data.len())?;
        b.as_mut_slice().copy_from_slice(data);
        Ok(b)
    }

    /// A pinned copy of a tensor's values (converted to `f32`, logical order).
    pub fn from_tensor(t: &crate::tensor::Tensor) -> crate::error::Result<Self> {
        Self::from_slice(&t.to_dtype(crate::dtype::DType::F32)?.to_vec::<f32>()?)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the operating system accepted the lock.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `ptr` is valid for `len` initialised `f32` values for the life of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as above, and `&mut self` gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// A tensor with a copy of the buffer's values in `shape`.
    pub fn to_tensor(&self, shape: &[usize]) -> crate::error::Result<crate::tensor::Tensor> {
        crate::tensor::Tensor::from_slice(self.as_slice(), shape)
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let raw = self.ptr.as_ptr() as *mut u8;
        // SAFETY: `raw` / `layout` are the allocation made in `new`; it is unlocked (if it was locked) and freed once.
        unsafe {
            if self.locked {
                #[cfg(windows)]
                VirtualUnlock(raw as *mut std::ffi::c_void, self.layout.size());
                #[cfg(unix)]
                munlock(raw as *const std::ffi::c_void, self.layout.size());
            }
            std::alloc::dealloc(raw, self.layout);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pinned_buffers_are_page_aligned_zeroed_writable_and_round_trip_tensors() {
        let mut b = super::PinnedBuffer::new(5000).unwrap(); // spans several pages
        assert_eq!(b.len(), 5000);
        assert_eq!(b.as_slice().as_ptr() as usize % super::PAGE, 0);
        assert!(b.as_slice().iter().all(|&v| v == 0.0));
        // a small buffer is within every default quota, so the lock itself must succeed on a normal machine
        assert!(b.is_locked(), "the OS refused to lock 20 KiB");
        for (i, v) in b.as_mut_slice().iter_mut().enumerate() {
            *v = i as f32;
        }
        let t = b.to_tensor(&[50, 100]).unwrap();
        assert_eq!(t.to_vec::<f32>().unwrap()[4999], 4999.0);
        let back = super::PinnedBuffer::from_tensor(&t.transpose(0, 1).unwrap()).unwrap();
        assert_eq!(back.as_slice()[1], 100.0); // logical order of the transposed view
        assert!(super::PinnedBuffer::new(0).unwrap().is_empty());
    }


    use crate::tensor::Tensor;

    // global counters: other tests allocate concurrently, so compare with inequalities on a large buffer
    #[test]
    fn a_large_tensor_raises_live_bytes_and_peak_and_releases_them_on_drop() {
        let big = 8 << 20; // 8 MiB of f32 = 2M elements
        let before = super::stats();
        let t = Tensor::zeros(&[big / 4], crate::dtype::DType::F32, &crate::device::Device::Cpu).unwrap();
        let view = t.reshape(&[2, big / 8]).unwrap(); // a view adds no buffer
        let during = super::stats();
        assert!(during.allocated_bytes >= before.allocated_bytes + big - (1 << 20) || during.peak_bytes >= big);
        assert!(during.peak_bytes >= big);
        drop(view);
        drop(t);
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(super::allocated_bytes() < during.allocated_bytes || during.allocated_bytes < big);
        assert!(super::stats().total_allocations > before.total_allocations);
    }
}
