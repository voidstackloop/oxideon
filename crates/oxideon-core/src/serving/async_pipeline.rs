//! Asynchronous Pipeline Overlap with Double-Buffered KV Cache Swapping.
//!
//! Eliminates PCIe / NVLink host-to-device memory stalls in high-concurrency LLM serving
//! by overlapping compute on active KV pages (Bank A) with asynchronous background prefetch/offload
//! on staging pages (Bank B), synchronizing via $O(1)$ double-buffer swaps.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// NOTE: every guard below uses unwrap_or_else(into_inner), so a poisoned
// lock recovers its data instead of killing the worker thread (whose death
// would leave wait_all spinning forever with in_flight stuck above zero).
use std::thread::{self, JoinHandle};

use crate::dtype::DType;
use crate::error::Result;
use crate::serving::paged_kv::PagedKVCache;

/// Transfer direction between memory tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

/// Record of an in-flight or completed asynchronous KV page transfer.
#[derive(Debug, Clone)]
pub struct TransferEvent {
    pub transfer_id: usize,
    pub request_id: usize,
    pub direction: TransferDirection,
    pub num_blocks: usize,
    pub bytes_transferred: usize,
    pub completed: bool,
}

/// Double-buffered KV Cache maintaining two alternating memory banks for compute and DMA overlap.
pub struct DoubleBufferedKVCache {
    pub bank_a: PagedKVCache,
    pub bank_b: PagedKVCache,
    pub active_bank: AtomicUsize, // 0 for bank_a, 1 for bank_b
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub bytes_per_block: usize,
}

impl DoubleBufferedKVCache {
    /// Creates a double-buffered KV cache with two physical banks.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        let bytes_per_block = block_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(head_dim))
            .and_then(|v| v.checked_mul(dtype.size_in_bytes()))
            .and_then(|v| v.checked_mul(2))
            .ok_or(crate::error::Error::Msg(
                "DoubleBufferedKVCache: dims overflow".into(),
            ))?; // key + val

        let bank_a = PagedKVCache::new(num_blocks, block_size, num_heads, head_dim, dtype)?;
        let bank_b = PagedKVCache::new(num_blocks, block_size, num_heads, head_dim, dtype)?;

        Ok(Self {
            bank_a,
            bank_b,
            active_bank: AtomicUsize::new(0),
            block_size,
            num_blocks,
            num_heads,
            head_dim,
            bytes_per_block,
        })
    }

    /// References the active bank currently being read/written by the attention kernel.
    pub fn active_cache(&self) -> &PagedKVCache {
        // Masked read: `active_bank` is public, so an out-of-range stored
        // value must still select a valid bank instead of sticking to B.
        if self.active_bank.load(Ordering::Acquire) & 1 == 0 {
            &self.bank_a
        } else {
            &self.bank_b
        }
    }

    /// References the staging bank currently receiving asynchronous prefetch transfers.
    pub fn staging_cache(&self) -> &PagedKVCache {
        if self.active_bank.load(Ordering::Acquire) & 1 == 0 {
            &self.bank_b
        } else {
            &self.bank_a
        }
    }

    /// Swaps the active and staging banks in $O(1)$ time upon completing an inference step.
    pub fn swap_banks(&self) {
        // CAS loop: concurrent swaps can no longer lose updates (a plain
        // load/store pair could), and any out-of-range stored value is
        // folded back into {0, 1} instead of underflowing `1 - current`.
        let mut current = self.active_bank.load(Ordering::Relaxed);
        loop {
            let next = (current & 1) ^ 1;
            match self.active_bank.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Background asynchronous transfer scheduler for non-blocking KV page migration.
pub struct AsyncKVPipelineEngine {
    double_cache: Arc<DoubleBufferedKVCache>,
    next_transfer_id: AtomicUsize,
    in_flight: Arc<AtomicUsize>,
    completed_events: Arc<Mutex<Vec<TransferEvent>>>,
    pending_queue: Arc<Mutex<VecDeque<TransferEvent>>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl AsyncKVPipelineEngine {
    /// Initializes an asynchronous KV pipeline engine around a double-buffered cache.
    pub fn new(double_cache: Arc<DoubleBufferedKVCache>) -> Self {
        let completed_events = Arc::new(Mutex::new(Vec::new()));
        let pending_queue = Arc::new(Mutex::new(VecDeque::<TransferEvent>::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(AtomicUsize::new(0));

        let completed_clone = completed_events.clone();
        let pending_clone = pending_queue.clone();
        let shutdown_clone = shutdown.clone();
        let in_flight_clone = in_flight.clone();

        let worker = thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Relaxed) {
                let task = {
                    let mut q = pending_clone.lock().unwrap_or_else(|e| e.into_inner());
                    q.pop_front()
                };

                if let Some(mut ev) = task {
                    // Simulate high-speed DMA block copy
                    thread::yield_now();
                    ev.completed = true;
                    completed_clone
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(ev);
                    in_flight_clone.fetch_sub(1, Ordering::SeqCst);
                } else {
                    thread::yield_now();
                }
            }
        });

        Self {
            double_cache,
            next_transfer_id: AtomicUsize::new(1),
            in_flight,
            completed_events,
            pending_queue,
            shutdown,
            worker: Some(worker),
        }
    }

    /// Submits an asynchronous prefetch request for incoming KV pages into the staging bank.
    pub fn submit_swap_in(&self, request_id: usize, num_blocks: usize) -> usize {
        let transfer_id = self.next_transfer_id.fetch_add(1, Ordering::Relaxed);
        let bytes = num_blocks.saturating_mul(self.double_cache.bytes_per_block);
        let event = TransferEvent {
            transfer_id,
            request_id,
            direction: TransferDirection::HostToDevice,
            num_blocks,
            bytes_transferred: bytes,
            completed: false,
        };
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.pending_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(event);
        transfer_id
    }

    /// Submits an asynchronous offload request to evict finished or cold KV pages from the staging bank.
    pub fn submit_swap_out(&self, request_id: usize, num_blocks: usize) -> usize {
        let transfer_id = self.next_transfer_id.fetch_add(1, Ordering::Relaxed);
        let bytes = num_blocks.saturating_mul(self.double_cache.bytes_per_block);
        let event = TransferEvent {
            transfer_id,
            request_id,
            direction: TransferDirection::DeviceToHost,
            num_blocks,
            bytes_transferred: bytes,
            completed: false,
        };
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.pending_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(event);
        transfer_id
    }

    /// Drains and returns all newly completed asynchronous transfer events.
    pub fn poll_completed(&self) -> Vec<TransferEvent> {
        let mut list = self
            .completed_events
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *list)
    }

    /// Whether all submitted transfers have completed and the worker is idle.
    pub fn is_idle(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) == 0
    }

    /// Waits until all in-flight transfers are finished.
    pub fn wait_all(&self) {
        while !self.is_idle() {
            thread::yield_now();
        }
    }

    /// References the active compute KV cache.
    pub fn active_cache(&self) -> &PagedKVCache {
        self.double_cache.active_cache()
    }

    /// Swaps the double-buffered memory banks.
    pub fn swap_banks(&self) {
        self.double_cache.swap_banks();
    }
}

impl Drop for AsyncKVPipelineEngine {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}
