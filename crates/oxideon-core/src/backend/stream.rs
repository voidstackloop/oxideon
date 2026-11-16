//! Asynchronous Compute Streams, Stream Events, and Capture-Replay Compute Graphs.
//!
//! Provides PyTorch CUDA/Metal-style stream concurrency (`torch.cuda.Stream`, `StreamEvent`)
//! and capture-and-replay compute graphs (`torch.cuda.CUDAGraph`) to eliminate host-device
//! kernel launch and Python dispatch latency during inference and iterative training.

use crate::device::device_type::Device;
use crate::error::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Unique identifier for an asynchronous execution stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub usize);

/// Synchronization event recorded on a [`ComputeStream`].
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub event_id: usize,
    completed: Arc<AtomicBool>,
}

impl StreamEvent {
    /// Creates a new uncompleted event.
    pub fn new(event_id: usize) -> Self {
        Self {
            event_id,
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signals that the event has been completed.
    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::Release);
    }

    /// Queries whether the event has completed execution.
    pub fn query(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Blocks until this event has finished execution.
    ///
    /// Spins briefly, then yields the thread instead of hot-spinning; for a
    /// bounded wait use [`StreamEvent::synchronize_timeout`].
    pub fn synchronize(&self) {
        let mut spins = 0u32;
        while !self.query() {
            if spins < 1000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Blocks until this event finishes or `timeout` elapses, returning an
    /// error on timeout instead of hanging forever.
    pub fn synchronize_timeout(&self, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();
        let mut spins = 0u32;
        while !self.query() {
            if start.elapsed() >= timeout {
                return Err(Error::Msg(format!(
                    "StreamEvent {} synchronize timed out after {timeout:?}",
                    self.event_id
                )));
            }
            if spins < 1000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        Ok(())
    }
}

/// Asynchronous compute execution stream managing ordered kernel executions.
#[derive(Debug, Clone)]
pub struct ComputeStream {
    pub id: StreamId,
    pub device: Device,
    next_event_id: Arc<Mutex<usize>>,
}

impl ComputeStream {
    /// Creates a new compute stream assigned to the given device.
    pub fn new(id: usize, device: Device) -> Self {
        Self {
            id: StreamId(id),
            device,
            next_event_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Records an event on this stream.
    pub fn record_event(&self) -> StreamEvent {
        let mut id_guard = self
            .next_event_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let eid = *id_guard;
        *id_guard += 1;
        let event = StreamEvent::new(eid);
        // In synchronous/CPU baseline, mark complete immediately; in asynchronous pipeline, marked by executor.
        event.mark_completed();
        event
    }

    /// Makes all future work submitted to this stream wait until `event` has completed.
    pub fn wait_event(&self, event: &StreamEvent) {
        event.synchronize();
    }

    /// Blocks the host until all operations submitted to this stream are finished.
    pub fn synchronize(&self) {
        // Stream completed
    }
}

/// Type-erased executable operation recorded inside a [`ComputeGraph`].
pub type GraphOp = Arc<dyn Fn() -> Result<()> + Send + Sync>;

/// Capture-and-replay compute graph (`torch.cuda.CUDAGraph` style).
///
/// Captures a static sequence of kernel executions and memory transfers, then replays
/// the entire execution graph with a single call, slashing kernel launch overhead.
#[derive(Default, Clone)]
pub struct ComputeGraph {
    is_capturing: bool,
    captured_ops: Vec<GraphOp>,
    replay_count: usize,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts capturing kernel executions into this graph.
    pub fn capture_begin(&mut self) -> Result<()> {
        if self.is_capturing {
            return Err(Error::Msg("ComputeGraph is already capturing".into()));
        }
        self.is_capturing = true;
        self.captured_ops.clear();
        Ok(())
    }

    /// Records an executable operation into the active capture buffer.
    pub fn record_op<F>(&mut self, op: F) -> Result<()>
    where
        F: Fn() -> Result<()> + Send + Sync + 'static,
    {
        if !self.is_capturing {
            return Err(Error::Msg(
                "Cannot record op: ComputeGraph is not capturing".into(),
            ));
        }
        self.captured_ops.push(Arc::new(op));
        Ok(())
    }

    /// Finishes capturing and finalizes the graph for fast replay.
    pub fn capture_end(&mut self) -> Result<usize> {
        if !self.is_capturing {
            return Err(Error::Msg("ComputeGraph was not in capturing mode".into()));
        }
        self.is_capturing = false;
        Ok(self.captured_ops.len())
    }

    /// Replays the captured graph sequentially.
    pub fn replay(&mut self) -> Result<()> {
        if self.is_capturing {
            return Err(Error::Msg(
                "Cannot replay while graph is actively capturing".into(),
            ));
        }
        for op in &self.captured_ops {
            op()?;
        }
        self.replay_count += 1;
        Ok(())
    }

    /// Number of operations captured in the graph.
    pub fn num_ops(&self) -> usize {
        self.captured_ops.len()
    }

    /// Total number of replays executed.
    pub fn replay_count(&self) -> usize {
        self.replay_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_lock_does_not_brick_event_ids() {
        let s = ComputeStream::new(0, Device::Cpu);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = s.next_event_id.lock().unwrap();
            panic!("intentional poison");
        }));
        // pre-fix: record_event panicked on the poisoned lock
        let _ = s.record_event();
        let _ = s.record_event();
    }
}
