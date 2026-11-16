//! Data pipeline pieces beyond the in-memory [`DataLoader`](crate::data::DataLoader):
//! - [`PrefetchLoader`]: worker threads build batches ahead of the training loop. Batches always come out in the same order
//!   whatever the number of workers, the order of epoch `e` is a pure function of `(seed, e)`, and an epoch can be resumed
//!   from any batch (deterministic resume after a restart).
//! - [`NpyDataset`]: random access to the rows of `.npy` files without loading them (the file stays on disk).

use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::serialize::npy::npy_header;
use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The sample order of an epoch: a permutation fixed by `(seed, epoch)` (or the identity without shuffling).
pub fn epoch_order(len: usize, shuffle: bool, seed: u64, epoch: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..len).collect();
    if shuffle {
        let mut st = seed ^ epoch.wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(0x1234_5678);
        for i in (1..len).rev() {
            let j = (splitmix(&mut st) % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
    }
    idx
}

type Batch = Result<(Tensor, Tensor)>;

struct Shared {
    next_job: AtomicUsize,
    consumed: AtomicUsize,
    stop: AtomicBool,
    ready: Mutex<BTreeMap<usize, Batch>>,
    cv: Condvar,
}

/// Multi-threaded, order-preserving, bounded-prefetch loader over a shared dataset.
pub struct PrefetchLoader<D: Dataset + Send + Sync + 'static> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    workers: usize,
    prefetch: usize,
    drop_last: bool,
}

impl<D: Dataset + Send + Sync + 'static> PrefetchLoader<D> {
    pub fn new(dataset: Arc<D>, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        Self { dataset, batch_size: batch_size.max(1), shuffle, seed, workers: 2, prefetch: 4, drop_last: false }
    }
    /// Number of worker threads (at least 1).
    pub fn workers(mut self, n: usize) -> Self {
        self.workers = n.max(1);
        self
    }
    /// How many batches may be built ahead of the one the consumer is on (at least 1).
    pub fn prefetch(mut self, n: usize) -> Self {
        self.prefetch = n.max(1);
        self
    }
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }

    pub fn num_batches(&self) -> usize {
        if self.drop_last { self.dataset.len() / self.batch_size } else { self.dataset.len().div_ceil(self.batch_size) }
    }

    /// The batches of epoch `epoch`.
    pub fn epoch(&self, epoch: u64) -> PrefetchIter {
        self.epoch_from(epoch, 0)
    }

    /// The batches of epoch `epoch` starting at batch `start_batch` (resume after a checkpoint).
    pub fn epoch_from(&self, epoch: u64, start_batch: usize) -> PrefetchIter {
        let order = epoch_order(self.dataset.len(), self.shuffle, self.seed, epoch);
        let mut batches: Vec<Vec<usize>> = order.chunks(self.batch_size).map(<[usize]>::to_vec).collect();
        if self.drop_last {
            batches.retain(|b| b.len() == self.batch_size);
        }
        let batches = Arc::new(batches.split_off(start_batch.min(batches.len())));
        let total = batches.len();
        let shared = Arc::new(Shared { next_job: AtomicUsize::new(0), consumed: AtomicUsize::new(0), stop: AtomicBool::new(false), ready: Mutex::new(BTreeMap::new()), cv: Condvar::new() });
        let handles = (0..self.workers.min(total.max(1)))
            .map(|_| {
                let (ds, sh, bs, prefetch) = (self.dataset.clone(), shared.clone(), batches.clone(), self.prefetch);
                std::thread::spawn(move || loop {
                    let i = sh.next_job.fetch_add(1, Ordering::SeqCst);
                    if i >= bs.len() {
                        break;
                    }
                    // bounded read-ahead: wait until the consumer is within `prefetch` batches of this one
                    let mut g = sh.ready.lock().unwrap();
                    while i >= sh.consumed.load(Ordering::SeqCst) + prefetch && !sh.stop.load(Ordering::SeqCst) {
                        g = sh.cv.wait(g).unwrap();
                    }
                    drop(g);
                    if sh.stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let b = ds.get_batch(&bs[i]);
                    sh.ready.lock().unwrap().insert(i, b);
                    sh.cv.notify_all();
                })
            })
            .collect();
        PrefetchIter { shared, handles, pos: 0, total }
    }
}

pub struct PrefetchIter {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    pos: usize,
    total: usize,
}

impl Iterator for PrefetchIter {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        if self.pos >= self.total {
            return None;
        }
        let mut g = self.shared.ready.lock().unwrap();
        loop {
            if let Some(b) = g.remove(&self.pos) {
                self.pos += 1;
                self.shared.consumed.store(self.pos, Ordering::SeqCst);
                drop(g);
                self.shared.cv.notify_all();
                return Some(b);
            }
            if self.handles.iter().all(|h| h.is_finished()) && !g.contains_key(&self.pos) {
                // the workers ended without producing this batch (a panic in the dataset)
                self.pos = self.total;
                return Some(Err(Error::Msg("PrefetchLoader: a worker thread died before producing the batch".into())));
            }
            g = self.shared.cv.wait_timeout(g, std::time::Duration::from_millis(50)).unwrap().0;
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total - self.pos, Some(self.total - self.pos))
    }
}

impl ExactSizeIterator for PrefetchIter {}

impl Drop for PrefetchIter {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ------------------------------------------------------------------------------------------------ streaming shards

/// Datasets too large to index randomly: a list of shards (each read front to back), shuffled by shard order and by a
/// bounded shuffle buffer, optionally split across `world_size` ranks. Every epoch's order is a pure function of
/// `(seed, epoch)`; [`epoch_from`](Self::epoch_from) resumes by replaying and discarding the batches already consumed.
pub struct StreamingLoader {
    shards: Vec<Arc<dyn Dataset + Send + Sync>>,
    batch_size: usize,
    buffer: usize,
    seed: u64,
    rank: usize,
    world: usize,
    drop_last: bool,
}

impl StreamingLoader {
    /// `buffer` samples are held for shuffling (1 disables intra-stream shuffling).
    pub fn new(shards: Vec<Arc<dyn Dataset + Send + Sync>>, batch_size: usize, buffer: usize, seed: u64) -> Self {
        Self { shards, batch_size: batch_size.max(1), buffer: buffer.max(1), seed, rank: 0, world: 1, drop_last: false }
    }
    /// This process reads shards `i` of the epoch's shard order with `i % world_size == rank`.
    pub fn rank(mut self, rank: usize, world_size: usize) -> Self {
        self.world = world_size.max(1);
        self.rank = rank.min(self.world - 1);
        self
    }
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }
    pub fn epoch(&self, epoch: u64) -> StreamIter {
        self.epoch_from(epoch, 0)
    }
    pub fn epoch_from(&self, epoch: u64, start_batch: usize) -> StreamIter {
        let order = epoch_order(self.shards.len(), true, self.seed, epoch);
        let queue: std::collections::VecDeque<usize> = order.into_iter().enumerate().filter(|(i, _)| i % self.world == self.rank).map(|(_, s)| s).collect();
        let mut it = StreamIter {
            shards: self.shards.clone(),
            queue,
            cur: None,
            buf: Vec::new(),
            cap: self.buffer,
            batch: self.batch_size,
            drop_last: self.drop_last,
            rng: self.seed ^ epoch.wrapping_mul(0xA24B_AED4_963E_E407) ^ ((self.rank as u64) << 48),
            failed: false,
        };
        for _ in 0..start_batch {
            if it.next().is_none() {
                break;
            }
        }
        it
    }
}

pub struct StreamIter {
    shards: Vec<Arc<dyn Dataset + Send + Sync>>,
    queue: std::collections::VecDeque<usize>,
    cur: Option<(usize, usize)>,
    buf: Vec<(Tensor, Tensor)>,
    cap: usize,
    batch: usize,
    drop_last: bool,
    rng: u64,
    failed: bool,
}

impl StreamIter {
    fn read_next(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        loop {
            if self.cur.is_none() {
                match self.queue.pop_front() {
                    Some(s) => self.cur = Some((s, 0)),
                    None => return Ok(None),
                }
            }
            let (s, pos) = self.cur.unwrap();
            if pos >= self.shards[s].len() {
                self.cur = None;
                continue;
            }
            self.cur = Some((s, pos + 1));
            return self.shards[s].get(pos).map(Some);
        }
    }

    /// The next sample after the shuffle buffer: top it up, then swap a random element out for the next incoming one.
    fn next_sample(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        while self.buf.len() < self.cap {
            match self.read_next()? {
                Some(x) => self.buf.push(x),
                None => break,
            }
        }
        if self.buf.is_empty() {
            return Ok(None);
        }
        let j = (splitmix(&mut self.rng) % self.buf.len() as u64) as usize;
        Ok(Some(match self.read_next()? {
            Some(incoming) => std::mem::replace(&mut self.buf[j], incoming),
            None => self.buf.swap_remove(j),
        }))
    }
}

impl Iterator for StreamIter {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        if self.failed {
            return None;
        }
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        while xs.len() < self.batch {
            match self.next_sample() {
                Ok(Some((x, y))) => {
                    xs.push(x);
                    ys.push(y);
                }
                Ok(None) => break,
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
        if xs.is_empty() || (self.drop_last && xs.len() < self.batch) {
            return None;
        }
        Some(Tensor::stack(&xs, 0).and_then(|x| Tensor::stack(&ys, 0).map(|y| (x, y))))
    }
}

// ------------------------------------------------------------------------------------------------ on-disk arrays

fn read_at(f: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
            }
            done += n;
        }
        Ok(())
    }
}

struct NpyFile {
    file: std::fs::File,
    shape: Vec<usize>,
    row_elems: usize,
    esize: usize,
    kind: (u8, usize),
    data_offset: u64,
}

impl NpyFile {
    fn open(path: &std::path::Path) -> Result<Self> {
        let io = |e: std::io::Error| Error::Msg(format!("NpyDataset: {}: {e}", path.display()));
        let file = std::fs::File::open(path).map_err(io)?;
        let mut head = vec![0u8; 4096];
        let mut f2 = &file;
        let n = std::io::Read::read(&mut f2, &mut head).map_err(io)?;
        head.truncate(n);
        let info = npy_header(&head)?;
        if info.fortran || info.descr.starts_with('>') {
            return Err(Error::Msg("NpyDataset: only little-endian C-order arrays are supported".into()));
        }
        let kind = info.descr.as_bytes()[1];
        let esize: usize = info.descr[2..].parse().map_err(|_| Error::Msg("NpyDataset: unsupported dtype".into()))?;
        if info.shape.is_empty() {
            return Err(Error::Msg("NpyDataset: the array needs at least one dimension".into()));
        }
        let row_elems = info.shape[1..].iter().product::<usize>().max(1);
        Ok(Self { file, shape: info.shape, row_elems, esize, kind: (kind, esize), data_offset: info.data_offset as u64 })
    }

    fn rows(&self) -> usize {
        self.shape[0]
    }

    fn row(&self, i: usize) -> Result<Tensor> {
        if i >= self.rows() {
            return Err(Error::Msg(format!("NpyDataset: row {i} out of range for {} rows", self.rows())));
        }
        let mut buf = vec![0u8; self.row_elems * self.esize];
        let at = self.data_offset + (i * buf.len()) as u64;
        read_at(&self.file, &mut buf, at).map_err(|e| Error::Msg(format!("NpyDataset: {e}")))?;
        let shape = &self.shape[1..];
        match self.kind {
            (b'f', 4) => Tensor::from_slice(&buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), shape),
            (b'f', 8) => Tensor::from_slice(&buf.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), shape),
            (b'i', 8) => Tensor::from_slice(&buf.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), shape),
            (b'i', 4) => Tensor::from_slice(&buf.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>(), shape),
            (b'u', 1) => Tensor::from_slice(&buf, shape),
            other => Err(Error::Msg(format!("NpyDataset: unsupported element type {other:?}"))),
        }
    }
}

/// Features and targets stored as `.npy` files; only the requested rows are read from disk.
pub struct NpyDataset {
    x: NpyFile,
    y: NpyFile,
}

impl NpyDataset {
    pub fn open(features: impl AsRef<std::path::Path>, targets: impl AsRef<std::path::Path>) -> Result<Self> {
        let (x, y) = (NpyFile::open(features.as_ref())?, NpyFile::open(targets.as_ref())?);
        if x.rows() != y.rows() {
            return Err(Error::Msg(format!("NpyDataset: {} feature rows vs {} target rows", x.rows(), y.rows())));
        }
        Ok(Self { x, y })
    }
}

impl Dataset for NpyDataset {
    fn len(&self) -> usize {
        self.x.rows()
    }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.x.row(index)?, self.y.row(index)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;
    use crate::serialize::npy::save_npy;

    fn dataset(n: usize) -> TensorDataset {
        let x: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32).collect();
        TensorDataset::new(Tensor::from_slice(&x, &[n, 3]).unwrap(), Tensor::from_slice(&y, &[n, 1]).unwrap())
    }

    fn ids(it: impl Iterator<Item = Batch>) -> Vec<Vec<i64>> {
        it.map(|b| b.unwrap().1.to_vec::<f32>().unwrap().into_iter().map(|v| v as i64).collect()).collect()
    }

    #[test]
    fn order_is_independent_of_worker_count_and_covers_every_sample_once() {
        let ds = Arc::new(dataset(103));
        let one = ids(PrefetchLoader::new(ds.clone(), 10, true, 7).workers(1).epoch(0));
        let many = ids(PrefetchLoader::new(ds.clone(), 10, true, 7).workers(5).prefetch(2).epoch(0));
        assert_eq!(one, many);
        assert_eq!(one.len(), 11);
        let mut all: Vec<i64> = one.concat();
        all.sort_unstable();
        assert_eq!(all, (0..103).collect::<Vec<i64>>());
        // another epoch has another order; the same epoch always the same
        let e1 = ids(PrefetchLoader::new(ds.clone(), 10, true, 7).epoch(1));
        assert_ne!(one, e1);
        assert_eq!(e1, ids(PrefetchLoader::new(ds.clone(), 10, true, 7).workers(3).epoch(1)));
        // no shuffle: the natural order; drop_last removes the short batch
        let plain = ids(PrefetchLoader::new(ds.clone(), 10, false, 0).drop_last(true).epoch(0));
        assert_eq!(plain.len(), 10);
        assert_eq!(plain[3], (30..40).collect::<Vec<i64>>());
    }

    #[test]
    fn resume_continues_where_the_epoch_stopped_and_dropping_early_is_safe() {
        let ds = Arc::new(dataset(60));
        let loader = PrefetchLoader::new(ds, 8, true, 3).workers(4);
        let full = ids(loader.epoch(2));
        let tail = ids(loader.epoch_from(2, 3));
        assert_eq!(tail, full[3..].to_vec());
        // abandoning an iterator mid-epoch joins the workers without hanging
        let mut it = loader.epoch(0);
        it.next().unwrap().unwrap();
        drop(it);
        assert_eq!(loader.epoch(0).len(), full.len());
    }

    #[test]
    fn streaming_shards_cover_everything_once_shuffle_deterministically_split_by_rank_and_resume() {
        let shards: Vec<Arc<dyn Dataset + Send + Sync>> = (0..5)
            .map(|s| {
                let x: Vec<f32> = (0..7).map(|i| (s * 7 + i) as f32).collect();
                Arc::new(TensorDataset::new(Tensor::from_slice(&x, &[7, 1]).unwrap(), Tensor::from_slice(&x, &[7, 1]).unwrap())) as Arc<dyn Dataset + Send + Sync>
            })
            .collect();
        let ids = |it: StreamIter| -> Vec<i64> { it.flat_map(|b| b.unwrap().1.to_vec::<f32>().unwrap().into_iter().map(|v| v as i64)).collect() };
        let l = StreamingLoader::new(shards.clone(), 4, 10, 9);
        let e0 = ids(l.epoch(0));
        let mut sorted = e0.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..35).collect::<Vec<i64>>()); // nothing lost, nothing repeated
        assert_ne!(e0, sorted); // but shuffled
        assert_eq!(e0, ids(l.epoch(0))); // reproducible
        assert_ne!(e0, ids(l.epoch(1)));
        // two ranks partition the shards: disjoint and together complete
        let r0 = ids(StreamingLoader::new(shards.clone(), 4, 10, 9).rank(0, 2).epoch(3));
        let r1 = ids(StreamingLoader::new(shards.clone(), 4, 10, 9).rank(1, 2).epoch(3));
        assert!(r0.iter().all(|v| !r1.contains(v)));
        let mut all: Vec<i64> = [r0, r1].concat();
        all.sort_unstable();
        assert_eq!(all, (0..35).collect::<Vec<i64>>());
        // resume at batch 3 gives the tail of the full epoch; drop_last drops the short batch
        let full: Vec<Vec<i64>> = l.epoch(2).map(|b| b.unwrap().1.to_vec::<f32>().unwrap().into_iter().map(|v| v as i64).collect()).collect();
        let tail: Vec<Vec<i64>> = l.epoch_from(2, 3).map(|b| b.unwrap().1.to_vec::<f32>().unwrap().into_iter().map(|v| v as i64).collect()).collect();
        assert_eq!(tail, full[3..].to_vec());
        assert_eq!(full.len(), 9);
        assert_eq!(StreamingLoader::new(shards, 4, 10, 9).drop_last(true).epoch(2).count(), 8);
    }

    #[test]
    fn npy_rows_are_read_lazily_from_disk() {
        let dir = std::env::temp_dir().join("oxideon_npy_ds_test");
        std::fs::create_dir_all(&dir).unwrap();
        let (xp, yp) = (dir.join("x.npy"), dir.join("y.npy"));
        let x = Tensor::from_slice(&(0..40).map(|i| i as f32 * 0.5).collect::<Vec<_>>(), &[10, 4]).unwrap();
        let y = Tensor::from_slice(&(0..10).map(|i| i as i64).collect::<Vec<_>>(), &[10]).unwrap();
        save_npy(&x, &xp).unwrap();
        save_npy(&y, &yp).unwrap();
        let ds = NpyDataset::open(&xp, &yp).unwrap();
        assert_eq!(ds.len(), 10);
        let (a, b) = ds.get(7).unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![14.0, 14.5, 15.0, 15.5]);
        assert_eq!(b.item::<i64>().unwrap(), 7);
        assert!(ds.get(10).is_err());
        let batch = ds.get_batch(&[0, 9]).unwrap();
        assert_eq!(batch.0.shape(), [2, 4]);
        // and it works behind the prefetching loader
        let n: usize = PrefetchLoader::new(Arc::new(ds), 4, false, 0).epoch(0).map(|b| b.unwrap().0.shape()[0]).sum();
        assert_eq!(n, 10);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
