//! Real process groups: ranks that are threads of one process ([`LocalTransport`]) or separate processes / machines
//! ([`TcpTransport`]), with ring collectives (all-reduce, reduce-scatter, all-gather), broadcast, barrier and
//! point-to-point messages, plus [`DistributedDataParallel`] (parameter broadcast and bucketed gradient all-reduce).
//!
//! The collectives work on `f32` / `f64` slices and on CPU tensors. The ring algorithms move `2 (w - 1) / w` of the data per
//! rank, the same volume as NCCL's ring all-reduce.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, no_grad};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

pub use crate::distributed::ReduceOp;

/// Reliable, ordered, per-peer message passing. `send` never waits for the receiver to call `recv`.
pub trait Transport: Send + Sync {
    fn send(&self, to: usize, data: Vec<u8>) -> Result<()>;
    fn recv(&self, from: usize) -> Result<Vec<u8>>;
}

/// In-process transport: one channel per ordered pair of ranks.
pub struct LocalTransport {
    senders: Vec<Option<Sender<Vec<u8>>>>,
    receivers: Vec<Option<Mutex<Receiver<Vec<u8>>>>>,
}

impl LocalTransport {
    /// Transports for `world` ranks, index = rank.
    pub fn world(world: usize) -> Vec<LocalTransport> {
        let mut ts: Vec<LocalTransport> = (0..world).map(|_| LocalTransport { senders: (0..world).map(|_| None).collect(), receivers: (0..world).map(|_| None).collect() }).collect();
        for i in 0..world {
            for j in 0..world {
                if i != j {
                    let (tx, rx) = channel();
                    ts[i].senders[j] = Some(tx);
                    ts[j].receivers[i] = Some(Mutex::new(rx));
                }
            }
        }
        ts
    }
}

impl Transport for LocalTransport {
    fn send(&self, to: usize, data: Vec<u8>) -> Result<()> {
        self.senders.get(to).and_then(|s| s.as_ref()).ok_or_else(|| Error::Msg(format!("comm: no route to rank {to}")))?.send(data).map_err(|_| Error::Msg(format!("comm: rank {to} has gone away")))
    }
    fn recv(&self, from: usize) -> Result<Vec<u8>> {
        let rx = self.receivers.get(from).and_then(|s| s.as_ref()).ok_or_else(|| Error::Msg(format!("comm: no route from rank {from}")))?;
        rx.lock().unwrap().recv().map_err(|_| Error::Msg(format!("comm: rank {from} has gone away")))
    }
}

/// TCP transport: a full mesh of connections, set up through a rendezvous at rank 0's `master_addr`.
/// A reader thread per peer drains its socket, so concurrent large sends in both directions cannot deadlock.
pub struct TcpTransport {
    writers: Vec<Option<Mutex<TcpStream>>>,
    queues: Vec<Option<Mutex<Receiver<Vec<u8>>>>>,
}

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error::Msg(format!("comm: {e}")))
}

fn write_frame(s: &mut TcpStream, data: &[u8]) -> Result<()> {
    io(s.write_all(&(data.len() as u64).to_le_bytes()))?;
    io(s.write_all(data))
}

fn read_frame(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 8];
    s.read_exact(&mut len)?;
    let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

impl TcpTransport {
    /// Joins the group. Rank 0 listens on `master_addr` (`host:port`, port may be fixed); every rank binds a data listener on
    /// `host` (the host part of `master_addr`) and the mesh is built from the addresses rank 0 collects.
    pub fn connect(rank: usize, world: usize, master_addr: &str) -> Result<Self> {
        if rank >= world {
            return Err(Error::Msg("comm: rank out of range".into()));
        }
        let host = master_addr.rsplit_once(':').map(|(h, _)| h).unwrap_or("127.0.0.1").to_string();
        let data_listener = io(TcpListener::bind(format!("{host}:0")))?;
        let my_addr = io(data_listener.local_addr())?.to_string();
        // rendezvous
        let table: Vec<String> = if rank == 0 {
            let master = io(TcpListener::bind(master_addr))?;
            let mut table = vec![String::new(); world];
            table[0] = my_addr.clone();
            let mut conns = Vec::new();
            for _ in 1..world {
                let (mut s, _) = io(master.accept())?;
                let msg = io(read_frame(&mut s))?;
                let text = String::from_utf8_lossy(&msg).to_string();
                let (r, a) = text.split_once(' ').ok_or_else(|| Error::Msg("comm: bad rendezvous message".into()))?;
                table[r.parse::<usize>().map_err(|_| Error::Msg("comm: bad rank".into()))?] = a.to_string();
                conns.push(s);
            }
            let joined = table.join("\n");
            for mut s in conns {
                write_frame(&mut s, joined.as_bytes())?;
            }
            table
        } else {
            let mut s = None;
            for _ in 0..600 {
                match TcpStream::connect(master_addr) {
                    Ok(c) => {
                        s = Some(c);
                        break;
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
                }
            }
            let mut s = s.ok_or_else(|| Error::Msg(format!("comm: cannot reach the master at {master_addr}")))?;
            write_frame(&mut s, format!("{rank} {my_addr}").as_bytes())?;
            let reply = io(read_frame(&mut s))?;
            String::from_utf8_lossy(&reply).split('\n').map(str::to_string).collect()
        };
        // mesh: connect to higher ranks, accept from lower ones
        let mut streams: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
        let accept = std::thread::spawn(move || -> Result<Vec<(usize, TcpStream)>> {
            let mut got = Vec::new();
            for _ in 0..rank {
                let (mut s, _) = io(data_listener.accept())?;
                let mut id = [0u8; 8];
                io(s.read_exact(&mut id))?;
                got.push((u64::from_le_bytes(id) as usize, s));
            }
            Ok(got)
        });
        for (peer, addr) in table.iter().enumerate().skip(rank + 1) {
            let mut s = io(TcpStream::connect(addr))?;
            io(s.write_all(&(rank as u64).to_le_bytes()))?;
            streams[peer] = Some(s);
        }
        for (peer, s) in accept.join().map_err(|_| Error::Msg("comm: accept thread panicked".into()))?? {
            streams[peer] = Some(s);
        }
        let mut writers = Vec::new();
        let mut queues = Vec::new();
        for s in streams {
            match s {
                None => {
                    writers.push(None);
                    queues.push(None);
                }
                Some(s) => {
                    io(s.set_nodelay(true))?;
                    let mut reader = io(s.try_clone())?;
                    let (tx, rx) = channel();
                    std::thread::spawn(move || {
                        while let Ok(frame) = read_frame(&mut reader) {
                            if tx.send(frame).is_err() {
                                break;
                            }
                        }
                    });
                    writers.push(Some(Mutex::new(s)));
                    queues.push(Some(Mutex::new(rx)));
                }
            }
        }
        Ok(Self { writers, queues })
    }
}

impl Transport for TcpTransport {
    fn send(&self, to: usize, data: Vec<u8>) -> Result<()> {
        let w = self.writers.get(to).and_then(|w| w.as_ref()).ok_or_else(|| Error::Msg(format!("comm: no connection to rank {to}")))?;
        write_frame(&mut w.lock().unwrap(), &data)
    }
    fn recv(&self, from: usize) -> Result<Vec<u8>> {
        let q = self.queues.get(from).and_then(|q| q.as_ref()).ok_or_else(|| Error::Msg(format!("comm: no connection from rank {from}")))?;
        q.lock().unwrap().recv().map_err(|_| Error::Msg(format!("comm: connection to rank {from} closed")))
    }
}

/// Element types the collectives move and reduce.
pub trait Elem: Copy + Send + Sync + 'static {
    const SIZE: usize;
    fn put(self, out: &mut Vec<u8>);
    fn get(b: &[u8]) -> Self;
    fn reduce(self, o: Self, op: ReduceOp) -> Self;
    fn scale(self, n: usize) -> Self;
}

macro_rules! elem {
    ($t:ty) => {
        impl Elem for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().unwrap())
            }
            fn reduce(self, o: Self, op: ReduceOp) -> Self {
                match op {
                    ReduceOp::Sum | ReduceOp::Mean => self + o,
                    ReduceOp::Max => if self.is_nan() || o.is_nan() { self + o } else if self >= o { self } else { o },
                    ReduceOp::Min => if self.is_nan() || o.is_nan() { self + o } else if self <= o { self } else { o },
                }
            }
            fn scale(self, n: usize) -> Self {
                self / n as $t
            }
        }
    };
}
elem!(f32);
elem!(f64);

fn encode<T: Elem>(v: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * T::SIZE);
    for &x in v {
        x.put(&mut out);
    }
    out
}

fn decode<T: Elem>(b: &[u8]) -> Result<Vec<T>> {
    if b.len() % T::SIZE != 0 {
        return Err(Error::Msg("comm: message length is not a multiple of the element size".into()));
    }
    Ok(b.chunks_exact(T::SIZE).map(T::get).collect())
}

/// A rank's view of the group.
#[derive(Clone)]
pub struct ProcessGroup {
    rank: usize,
    world: usize,
    transport: Arc<dyn Transport>,
}

impl ProcessGroup {
    pub fn new(rank: usize, world: usize, transport: Arc<dyn Transport>) -> Self {
        Self { rank, world, transport }
    }

    /// Joins a TCP group (rank 0 listens on `master_addr`).
    pub fn tcp(rank: usize, world: usize, master_addr: &str) -> Result<Self> {
        Ok(Self::new(rank, world, Arc::new(TcpTransport::connect(rank, world, master_addr)?)))
    }

    /// Initializes a [`ProcessGroup`] from standard distributed environment variables
    /// (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), falling back to a single-rank local group.
    pub fn from_env() -> Result<Self> {
        let rank: usize = std::env::var("RANK")
            .ok()
            .map(|v| v.parse())
            .transpose()
            .map_err(|e| Error::Msg(format!("comm: invalid RANK env var: {e}")))?
            .unwrap_or(0);
        let world: usize = std::env::var("WORLD_SIZE")
            .ok()
            .map(|v| v.parse())
            .transpose()
            .map_err(|e| Error::Msg(format!("comm: invalid WORLD_SIZE env var: {e}")))?
            .unwrap_or(1);
        if world == 0 || rank >= world {
            return Err(Error::Msg(format!("comm: invalid rank {rank} for world_size {world}")));
        }
        if world == 1 {
            let t = LocalTransport::world(1)
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("comm: failed to create local transport".into()))?;
            return Ok(Self::new(0, 1, Arc::new(t)));
        }
        let addr = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = std::env::var("MASTER_PORT").unwrap_or_else(|_| "29500".to_string());
        Self::tcp(rank, world, &format!("{addr}:{port}"))
    }

    /// Runs `f(group)` on `world` threads that form an in-process group and returns their results in rank order.
    pub fn run_local<R: Send + 'static>(world: usize, f: impl Fn(ProcessGroup) -> R + Send + Sync + 'static) -> Vec<R> {
        let f = Arc::new(f);
        let handles: Vec<_> = LocalTransport::world(world)
            .into_iter()
            .enumerate()
            .map(|(rank, t)| {
                let (f, pg) = (f.clone(), ProcessGroup::new(rank, world, Arc::new(t)));
                std::thread::spawn(move || f(pg))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world_size(&self) -> usize {
        self.world
    }

    pub fn send_bytes(&self, to: usize, data: Vec<u8>) -> Result<()> {
        self.transport.send(to, data)
    }
    pub fn recv_bytes(&self, from: usize) -> Result<Vec<u8>> {
        self.transport.recv(from)
    }

    /// Blocks until every rank has arrived.
    pub fn barrier(&self) -> Result<()> {
        let mut x = [0f32];
        self.all_reduce(&mut x, ReduceOp::Sum)
    }

    /// Copies `buf` of rank `root` to every rank.
    pub fn broadcast<T: Elem>(&self, buf: &mut [T], root: usize) -> Result<()> {
        if self.world == 1 {
            return Ok(());
        }
        if self.rank == root {
            let bytes = encode(buf);
            for r in (0..self.world).filter(|&r| r != root) {
                self.transport.send(r, bytes.clone())?;
            }
        } else {
            let v: Vec<T> = decode(&self.transport.recv(root)?)?;
            if v.len() != buf.len() {
                return Err(Error::Msg("comm: broadcast buffers differ in length between ranks".into()));
            }
            buf.copy_from_slice(&v);
        }
        Ok(())
    }

    fn chunk(&self, len: usize, i: usize) -> std::ops::Range<usize> {
        (i * len / self.world)..((i + 1) * len / self.world)
    }

    /// Ring reduce-scatter; afterwards rank r owns the fully reduced chunk `(r + shift + 1) mod w`.
    fn ring_reduce_scatter<T: Elem>(&self, buf: &mut [T], op: ReduceOp, shift: usize) -> Result<()> {
        let (w, r) = (self.world, self.rank);
        let (next, prev) = ((r + 1) % w, (r + w - 1) % w);
        for s in 0..w - 1 {
            let send_idx = (r + shift + w - s) % w;
            let recv_idx = (r + shift + w - s - 1) % w;
            self.transport.send(next, encode(&buf[self.chunk(buf.len(), send_idx)]))?;
            let got: Vec<T> = decode(&self.transport.recv(prev)?)?;
            let range = self.chunk(buf.len(), recv_idx);
            if got.len() != range.len() {
                return Err(Error::Msg("comm: ranks disagree on the buffer length".into()));
            }
            for (d, g) in buf[range].iter_mut().zip(got) {
                *d = d.reduce(g, op);
            }
        }
        Ok(())
    }

    fn ring_all_gather<T: Elem>(&self, buf: &mut [T], owned: usize) -> Result<()> {
        let (w, r) = (self.world, self.rank);
        let (next, prev) = ((r + 1) % w, (r + w - 1) % w);
        for s in 0..w - 1 {
            let send_idx = (owned + w - s) % w;
            let recv_idx = (owned + w - s - 1) % w;
            self.transport.send(next, encode(&buf[self.chunk(buf.len(), send_idx)]))?;
            let got: Vec<T> = decode(&self.transport.recv(prev)?)?;
            let range = self.chunk(buf.len(), recv_idx);
            if got.len() != range.len() {
                return Err(Error::Msg("comm: ranks disagree on the buffer length".into()));
            }
            buf[range].copy_from_slice(&got);
        }
        Ok(())
    }

    /// Reduces `buf` across all ranks in place (ring reduce-scatter followed by ring all-gather).
    pub fn all_reduce<T: Elem>(&self, buf: &mut [T], op: ReduceOp) -> Result<()> {
        let w = self.world;
        if w > 1 {
            if buf.len() < w {
                // too small to split: exchange whole buffers
                let all = self.all_gather(buf)?;
                let n = buf.len();
                for (i, d) in buf.iter_mut().enumerate() {
                    let mut acc = all[i];
                    for r in 1..w {
                        acc = acc.reduce(all[r * n + i], op);
                    }
                    *d = acc;
                }
            } else {
                self.ring_reduce_scatter(buf, op, 0)?;
                self.ring_all_gather(buf, (self.rank + 1) % w)?;
            }
        }
        if op == ReduceOp::Mean {
            for d in buf.iter_mut() {
                *d = d.scale(w);
            }
        }
        Ok(())
    }

    /// Reduces across ranks and returns this rank's share of the result (chunk `rank` of the buffer).
    pub fn reduce_scatter<T: Elem>(&self, buf: &[T], op: ReduceOp) -> Result<Vec<T>> {
        let mut work = buf.to_vec();
        if self.world > 1 {
            if buf.len() < self.world {
                self.all_reduce(&mut work, if op == ReduceOp::Mean { ReduceOp::Sum } else { op })?;
            } else {
                self.ring_reduce_scatter(&mut work, if op == ReduceOp::Mean { ReduceOp::Sum } else { op }, self.world - 1)?;
            }
        }
        let mut mine = work[self.chunk(buf.len(), self.rank)].to_vec();
        if op == ReduceOp::Mean {
            for d in &mut mine {
                *d = d.scale(self.world);
            }
        }
        Ok(mine)
    }

    /// Concatenates every rank's `local` (all the same length) in rank order.
    pub fn all_gather<T: Elem>(&self, local: &[T]) -> Result<Vec<T>> {
        let (w, n) = (self.world, local.len());
        let mut out = vec![local.first().copied().unwrap_or_else(|| T::get(&vec![0u8; T::SIZE])); n * w];
        if n == 0 {
            return Ok(out);
        }
        out[self.rank * n..(self.rank + 1) * n].copy_from_slice(local);
        let (next, prev) = ((self.rank + 1) % w, (self.rank + w - 1) % w);
        for s in 0..w.saturating_sub(1) {
            let send_idx = (self.rank + w - s) % w;
            let recv_idx = (self.rank + w - s - 1) % w;
            self.transport.send(next, encode(&out[send_idx * n..(send_idx + 1) * n]))?;
            let got: Vec<T> = decode(&self.transport.recv(prev)?)?;
            if got.len() != n {
                return Err(Error::Msg("comm: all_gather inputs differ in length between ranks".into()));
            }
            out[recv_idx * n..(recv_idx + 1) * n].copy_from_slice(&got);
        }
        Ok(out)
    }

    // ---- tensors (CPU, F32 / F64) ----

    /// All-reduces a tensor and returns the result as a new tensor.
    pub fn all_reduce_tensor(&self, t: &Tensor, op: ReduceOp) -> Result<Tensor> {
        match t.dtype() {
            DType::F32 => {
                let mut v = t.to_vec::<f32>()?;
                self.all_reduce(&mut v, op)?;
                Tensor::from_slice(&v, t.shape())
            }
            DType::F64 => {
                let mut v = t.to_vec::<f64>()?;
                self.all_reduce(&mut v, op)?;
                Tensor::from_slice(&v, t.shape())
            }
            d => Err(Error::Msg(format!("comm: all_reduce supports F32 and F64 tensors, not {d:?}"))),
        }
    }

    /// Overwrites `t` on every rank with rank `root`'s values.
    pub fn broadcast_tensor(&self, t: &Tensor, root: usize) -> Result<()> {
        let new = match t.dtype() {
            DType::F32 => {
                let mut v = t.to_vec::<f32>()?;
                self.broadcast(&mut v, root)?;
                Tensor::from_slice(&v, t.shape())?
            }
            DType::F64 => {
                let mut v = t.to_vec::<f64>()?;
                self.broadcast(&mut v, root)?;
                Tensor::from_slice(&v, t.shape())?
            }
            d => return Err(Error::Msg(format!("comm: broadcast supports F32 and F64 tensors, not {d:?}"))),
        };
        t.assign(&new)
    }

    /// Gathers `local` tensors from every rank and concatenates them along `dim`.
    pub fn all_gather_tensor(&self, local: &Tensor, dim: isize) -> Result<Tensor> {
        let c = local.contiguous()?;
        let n = c.numel();
        let pieces: Vec<Tensor> = match c.dtype() {
            DType::F32 => {
                let gathered = self.all_gather(&c.to_vec::<f32>()?)?;
                (0..self.world)
                    .map(|r| Tensor::from_slice(&gathered[r * n..(r + 1) * n], c.shape()))
                    .collect::<Result<Vec<_>>>()?
            }
            DType::F64 => {
                let gathered = self.all_gather(&c.to_vec::<f64>()?)?;
                (0..self.world)
                    .map(|r| Tensor::from_slice(&gathered[r * n..(r + 1) * n], c.shape()))
                    .collect::<Result<Vec<_>>>()?
            }
            d => return Err(Error::Msg(format!("comm: all_gather_tensor supports F32 and F64 tensors, not {d:?}"))),
        };
        Tensor::cat(&pieces, dim)
    }

    /// Reduces `full` across all ranks and scatters equal slices, returning this rank's shard.
    pub fn reduce_scatter_tensor(&self, full: &Tensor, op: ReduceOp) -> Result<Tensor> {
        let c = full.contiguous()?;
        let shape = c.shape();
        let out_shape: Vec<usize> = if !shape.is_empty() && shape[0] % self.world == 0 {
            let mut s = shape.to_vec();
            s[0] /= self.world;
            s
        } else if c.numel() % self.world == 0 {
            vec![c.numel() / self.world]
        } else {
            return Err(Error::Msg(format!(
                "comm: reduce_scatter_tensor numel {} not divisible by world_size {}",
                c.numel(),
                self.world
            )));
        };
        match c.dtype() {
            DType::F32 => {
                let shard = self.reduce_scatter(&c.to_vec::<f32>()?, op)?;
                Tensor::from_slice(&shard, &out_shape)
            }
            DType::F64 => {
                let shard = self.reduce_scatter(&c.to_vec::<f64>()?, op)?;
                Tensor::from_slice(&shard, &out_shape)
            }
            d => Err(Error::Msg(format!("comm: reduce_scatter_tensor supports F32 and F64 tensors, not {d:?}"))),
        }
    }
}

/// Handle for an asynchronous collective communication operation.
pub struct CommHandle {
    handle: Option<std::thread::JoinHandle<Result<()>>>,
}

impl CommHandle {
    /// Blocks until the asynchronous communication finishes and returns its status.
    pub fn wait(mut self) -> Result<()> {
        match self.handle.take() {
            Some(h) => h.join().map_err(|_| Error::Msg("comm: async collective thread panicked".into()))?,
            None => Ok(()),
        }
    }
}

/// Data-parallel training over a [`ProcessGroup`]: every rank holds a full replica, parameters start identical (broadcast from
/// rank 0) and gradients are averaged across ranks in fixed-size buckets before the optimizer step.
#[derive(Clone)]
pub struct DistributedDataParallel {
    params: Vec<Tensor>,
    pg: ProcessGroup,
    bucket_elems: usize,
}

impl DistributedDataParallel {
    /// `bucket_bytes` is the target size of one all-reduce message (PyTorch's default is 25 MB).
    pub fn new(params: Vec<Tensor>, pg: ProcessGroup, bucket_bytes: usize) -> Result<Self> {
        let ddp = Self { params, pg, bucket_elems: (bucket_bytes / 4).max(1) };
        ddp.broadcast_parameters(0)?;
        Ok(ddp)
    }

    pub fn broadcast_parameters(&self, root: usize) -> Result<()> {
        for p in &self.params {
            self.pg.broadcast_tensor(p, root)?;
        }
        Ok(())
    }

    /// Replaces every parameter's gradient by its mean across ranks (a missing gradient counts as zero on that rank).
    pub fn sync_gradients(&self) -> Result<()> {
        // gradients of one dtype are flattened into buckets; each bucket is one all-reduce
        for dtype in [DType::F32, DType::F64] {
            let idx: Vec<usize> = (0..self.params.len()).filter(|&i| self.params[i].dtype() == dtype).collect();
            let mut start = 0;
            while start < idx.len() {
                let mut end = start;
                let mut elems = 0;
                while end < idx.len() && (elems == 0 || elems + self.params[idx[end]].numel() <= self.bucket_elems) {
                    elems += self.params[idx[end]].numel();
                    end += 1;
                }
                let members = &idx[start..end];
                if dtype == DType::F32 {
                    self.sync_bucket::<f32>(members)?;
                } else {
                    self.sync_bucket::<f64>(members)?;
                }
                start = end;
            }
        }
        Ok(())
    }

    /// Launches [`Self::sync_gradients`] on a background thread so communication can overlap with compute.
    pub fn sync_gradients_async(&self) -> Result<CommHandle> {
        let ddp = self.clone();
        let handle = std::thread::spawn(move || ddp.sync_gradients());
        Ok(CommHandle { handle: Some(handle) })
    }

    fn sync_bucket<T: Elem + crate::dtype::Element>(&self, members: &[usize]) -> Result<()> {
        let mut flat: Vec<T> = Vec::new();
        for &i in members {
            let p = &self.params[i];
            match p.grad() {
                Some(g) => flat.extend(g.contiguous()?.to_vec::<T>()?),
                None => flat.extend(std::iter::repeat_n(T::get(&vec![0u8; T::SIZE]), p.numel())),
            }
        }
        self.pg.all_reduce(&mut flat, ReduceOp::Mean)?;
        let mut off = 0;
        for &i in members {
            let p = &self.params[i];
            let n = p.numel();
            p.set_grad(Some(Tensor::from_slice(&flat[off..off + n], p.shape())?));
            off += n;
        }
        Ok(())
    }
}

/// ZeRO stage 1 over a real [`ProcessGroup`]: gradients are reduce-scattered, each rank runs AdamW on its own `1 / world`
/// slice of the flattened parameters (so Adam's two moment buffers are sharded too), and the updated slices are all-gathered.
/// Compute is `f64`; parameters keep their dtype.
pub struct ShardedAdamW {
    params: Vec<Tensor>,
    pg: ProcessGroup,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    padded: usize,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl ShardedAdamW {
    pub fn new(params: Vec<Tensor>, pg: ProcessGroup, lr: f64, weight_decay: f64) -> Result<Self> {
        let total: usize = params.iter().map(Tensor::numel).sum();
        let w = pg.world_size();
        let padded = total.div_ceil(w) * w;
        let shard = padded / w;
        let me = Self { params, pg, lr, betas: (0.9, 0.999), eps: 1e-8, weight_decay, padded, m: vec![0.0; shard], v: vec![0.0; shard], t: 0 };
        // every rank starts from rank 0's parameters
        for p in &me.params {
            me.pg.broadcast_tensor(p, 0)?;
        }
        Ok(me)
    }

    /// Optimizer-state elements held by this rank (versus `2 * numel` when replicated).
    pub fn state_elements(&self) -> usize {
        self.m.len() + self.v.len()
    }

    fn flat(&self, grads: bool) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(self.padded);
        for p in &self.params {
            let t = if grads {
                match p.grad() {
                    Some(g) => g.contiguous()?,
                    None => {
                        out.extend(std::iter::repeat_n(0.0, p.numel()));
                        continue;
                    }
                }
            } else {
                p.detach().contiguous()?
            };
            out.extend(t.to_dtype(DType::F64)?.to_vec::<f64>()?);
        }
        out.resize(self.padded, 0.0);
        Ok(out)
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// One update from the current gradients (each rank's own mini-batch gradient; they are averaged here).
    pub fn step(&mut self) -> Result<()> {
        self.t += 1;
        let g_shard = self.pg.reduce_scatter(&self.flat(true)?, ReduceOp::Mean)?;
        let mut full = self.flat(false)?;
        let shard = self.padded / self.pg.world_size();
        let lo = self.pg.rank() * shard;
        let (b1, b2) = self.betas;
        let (c1, c2) = (1.0 - b1.powi(self.t as i32), 1.0 - b2.powi(self.t as i32));
        let mut mine = full[lo..lo + shard].to_vec();
        for i in 0..shard {
            let g = g_shard[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            mine[i] -= self.lr * ((self.m[i] / c1) / ((self.v[i] / c2).sqrt() + self.eps) + self.weight_decay * mine[i]);
        }
        full = self.pg.all_gather(&mine)?;
        let mut off = 0;
        no_grad(|| -> Result<()> {
            for p in &self.params {
                let n = p.numel();
                let new = Tensor::from_slice(&full[off..off + n], p.shape())?.to_dtype(p.dtype())?;
                p.assign(&new)?;
                off += n;
            }
            Ok(())
        })
    }
}

/// ZeRO stage 3 / FSDP sharded parameter over a [`ProcessGroup`]: each rank stores only its `1 / world_size`
/// slice of the flattened parameter, materializing the full parameter on demand via [`Self::gather`] and
/// scattering gradients back to owning ranks via [`Self::reduce_scatter_grad`].
pub struct FullyShardedParam {
    shard: Tensor,
    full_shape: Vec<usize>,
    full_numel: usize,
    shard_len: usize,
    dtype: DType,
}

impl FullyShardedParam {
    /// Shards `full` across `pg`, keeping only the local rank's slice.
    pub fn from_tensor(full: &Tensor, pg: &ProcessGroup) -> Result<Self> {
        let world = pg.world_size();
        let rank = pg.rank();
        let full_shape = full.shape().to_vec();
        let full_numel = full.numel();
        let shard_len = full_numel.div_ceil(world).max(1);
        let padded = shard_len * world;
        let lo = rank * shard_len;
        let hi = lo + shard_len;
        let c = full.contiguous()?;
        let dtype = c.dtype();
        let shard = match dtype {
            DType::F32 => {
                let mut buf = c.to_vec::<f32>()?;
                buf.resize(padded, 0.0);
                Tensor::from_slice(&buf[lo..hi], &[shard_len])?
            }
            DType::F64 => {
                let mut buf = c.to_vec::<f64>()?;
                buf.resize(padded, 0.0);
                Tensor::from_slice(&buf[lo..hi], &[shard_len])?
            }
            d => return Err(Error::Msg(format!("comm: FullyShardedParam supports F32 and F64, not {d:?}"))),
        };
        Ok(Self { shard, full_shape, full_numel, shard_len, dtype })
    }

    pub fn shard(&self) -> &Tensor {
        &self.shard
    }

    pub fn shard_mut(&mut self) -> &mut Tensor {
        &mut self.shard
    }

    pub fn full_shape(&self) -> &[usize] {
        &self.full_shape
    }

    /// All-gathers every rank's shard and reconstructs the full unsharded parameter tensor.
    pub fn gather(&self, pg: &ProcessGroup) -> Result<Tensor> {
        match self.dtype {
            DType::F32 => {
                let full = pg.all_gather(&self.shard.to_vec::<f32>()?)?;
                Tensor::from_slice(&full[..self.full_numel], &self.full_shape)
            }
            DType::F64 => {
                let full = pg.all_gather(&self.shard.to_vec::<f64>()?)?;
                Tensor::from_slice(&full[..self.full_numel], &self.full_shape)
            }
            d => Err(Error::Msg(format!("comm: FullyShardedParam supports F32 and F64, not {d:?}"))),
        }
    }

    /// Reduce-scatters `full_grad` across `pg` and returns this rank's `[shard_len]` gradient shard.
    pub fn reduce_scatter_grad(&self, pg: &ProcessGroup, full_grad: &Tensor, op: ReduceOp) -> Result<Tensor> {
        if full_grad.numel() != self.full_numel {
            return Err(Error::Msg(format!(
                "comm: reduce_scatter_grad expected {} elements, got {}",
                self.full_numel,
                full_grad.numel()
            )));
        }
        let padded = self.shard_len * pg.world_size();
        let c = full_grad.contiguous()?.to_dtype(self.dtype)?;
        match self.dtype {
            DType::F32 => {
                let mut buf = c.to_vec::<f32>()?;
                buf.resize(padded, 0.0);
                let g_shard = pg.reduce_scatter(&buf, op)?;
                Tensor::from_slice(&g_shard, &[self.shard_len])
            }
            DType::F64 => {
                let mut buf = c.to_vec::<f64>()?;
                buf.resize(padded, 0.0);
                let g_shard = pg.reduce_scatter(&buf, op)?;
                Tensor::from_slice(&g_shard, &[self.shard_len])
            }
            d => Err(Error::Msg(format!("comm: FullyShardedParam supports F32 and F64, not {d:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(rank: usize, n: usize) -> Vec<f64> {
        (0..n).map(|i| (rank * 31 + i * 7 % 13) as f64 * 0.5 - 3.0).collect()
    }

    #[test]
    fn ring_collectives_match_a_serial_reference() {
        for world in [1usize, 2, 3, 5] {
            for n in [0usize, 1, 2, 7, 64, 101] {
                let res = ProcessGroup::run_local(world, move |pg| {
                    let mine = data(pg.rank(), n);
                    let mut sum = mine.clone();
                    pg.all_reduce(&mut sum, ReduceOp::Sum).unwrap();
                    let mut mx = mine.clone();
                    pg.all_reduce(&mut mx, ReduceOp::Max).unwrap();
                    let mut mean = mine.clone();
                    pg.all_reduce(&mut mean, ReduceOp::Mean).unwrap();
                    let gathered = pg.all_gather(&mine).unwrap();
                    let scattered = pg.reduce_scatter(&mine, ReduceOp::Sum).unwrap();
                    let mut b = if pg.rank() == world - 1 { mine.clone() } else { vec![0.0; n] };
                    pg.broadcast(&mut b, world - 1).unwrap();
                    pg.barrier().unwrap();
                    (sum, mx, mean, gathered, scattered, b)
                });
                let all: Vec<Vec<f64>> = (0..world).map(|r| data(r, n)).collect();
                let want_sum: Vec<f64> = (0..n).map(|i| all.iter().map(|v| v[i]).sum()).collect();
                let want_max: Vec<f64> = (0..n).map(|i| all.iter().map(|v| v[i]).fold(f64::MIN, f64::max)).collect();
                for (r, (sum, mx, mean, gathered, scattered, b)) in res.iter().enumerate() {
                    assert!(sum.iter().zip(&want_sum).all(|(a, b)| (a - b).abs() < 1e-9), "sum w={world} n={n}");
                    assert_eq!(mx, &want_max, "max w={world} n={n}");
                    assert!(mean.iter().zip(&want_sum).all(|(a, b)| (a - b / world as f64).abs() < 1e-9));
                    assert_eq!(gathered, &all.concat(), "gather w={world} n={n}");
                    let lo = r * n / world;
                    assert!(scattered.iter().zip(&want_sum[lo..]).all(|(a, b)| (a - b).abs() < 1e-9), "reduce_scatter w={world} n={n} r={r}");
                    assert_eq!(b, &all[world - 1], "broadcast");
                }
            }
        }
    }

    #[test]
    fn tcp_group_runs_the_same_collectives() {
        let world = 3;
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let addr = format!("127.0.0.1:{port}");
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let addr = addr.clone();
                std::thread::spawn(move || {
                    let pg = ProcessGroup::tcp(rank, world, &addr).unwrap();
                    let n = 100_000; // large enough that both directions of a socket are busy at once
                    let mut v: Vec<f32> = (0..n).map(|i| (i % 17) as f32 + rank as f32).collect();
                    pg.all_reduce(&mut v, ReduceOp::Sum).unwrap();
                    pg.barrier().unwrap();
                    v
                })
            })
            .collect();
        for h in handles {
            let v = h.join().unwrap();
            assert!(v.iter().enumerate().all(|(i, x)| (*x - (3.0 * (i % 17) as f32 + 3.0)).abs() < 1e-3));
        }
    }

    #[test]
    fn ddp_keeps_replicas_identical_and_averages_gradients() {
        let out = ProcessGroup::run_local(3, |pg| {
            let rank = pg.rank();
            let p = Tensor::from_slice(&[rank as f32 + 1.0; 4], &[2, 2]).unwrap().requires_grad_(true);
            let q = Tensor::from_slice(&[rank as f64; 3], &[3]).unwrap().requires_grad_(true);
            let ddp = DistributedDataParallel::new(vec![p.clone(), q.clone()], pg, 8).unwrap(); // tiny buckets
            // parameters were broadcast from rank 0
            assert_eq!(p.to_vec::<f32>().unwrap(), vec![1.0; 4]);
            p.set_grad(Some(Tensor::from_slice(&[rank as f32; 4], &[2, 2]).unwrap()));
            // q has no gradient on rank 1
            if rank != 1 {
                q.set_grad(Some(Tensor::from_slice(&[10.0f64 * (rank + 1) as f64; 3], &[3]).unwrap()));
            }
            ddp.sync_gradients().unwrap();
            (p.grad().unwrap().to_vec::<f32>().unwrap(), q.grad().unwrap().to_vec::<f64>().unwrap())
        });
        for (gp, gq) in out {
            assert!(gp.iter().all(|x| (*x - 1.0).abs() < 1e-6)); // mean of 0, 1, 2
            assert!(gq.iter().all(|x| (*x - 40.0 / 3.0).abs() < 1e-9)); // (10 + 0 + 30) / 3
        }
    }

    #[test]
    fn tensor_collectives_and_async_ddp() {
        let single = ProcessGroup::from_env().unwrap();
        assert_eq!(single.world_size(), 1);
        assert_eq!(single.rank(), 0);

        let out = ProcessGroup::run_local(3, |pg| {
            let rank = pg.rank();
            let local = Tensor::from_slice(&[rank as f32 + 1.0; 4], &[2, 2]).unwrap();
            let gathered = pg.all_gather_tensor(&local, 0).unwrap();
            assert_eq!(gathered.shape(), &[6, 2]);
            let scattered = pg.reduce_scatter_tensor(&gathered, ReduceOp::Sum).unwrap();
            assert_eq!(scattered.shape(), &[2, 2]);

            let p = Tensor::from_slice(&[rank as f32 + 1.0; 4], &[2, 2]).unwrap().requires_grad_(true);
            let ddp = DistributedDataParallel::new(vec![p.clone()], pg, 16).unwrap();
            p.set_grad(Some(Tensor::from_slice(&[rank as f32 * 3.0; 4], &[2, 2]).unwrap()));
            let handle = ddp.sync_gradients_async().unwrap();
            handle.wait().unwrap();
            (gathered.to_vec::<f32>().unwrap(), scattered.to_vec::<f32>().unwrap(), p.grad().unwrap().to_vec::<f32>().unwrap())
        });

        for (rank, (g, s, grad)) in out.iter().enumerate() {
            assert_eq!(g, &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
            let want = (rank as f32 + 1.0) * 3.0;
            assert!(s.iter().all(|x| (*x - want).abs() < 1e-6));
            // mean of 0, 3, 6 is 3.0
            assert!(grad.iter().all(|x| (*x - 3.0).abs() < 1e-6));
        }
    }

    #[test]
    fn fsdp_shards_gathers_and_reduce_scatters() {
        let out = ProcessGroup::run_local(3, |pg| {
            let rank = pg.rank();
            let full = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[7]).unwrap();
            let fsdp = FullyShardedParam::from_tensor(&full, &pg).unwrap();
            assert_eq!(fsdp.shard().shape(), &[3]); // ceil(7 / 3) = 3
            let gathered = fsdp.gather(&pg).unwrap();
            assert_eq!(gathered.shape(), &[7]);
            assert_eq!(gathered.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

            // Each rank produces a local gradient [rank+1; 7]; mean across ranks 0,1,2 is 2.0
            let full_grad = Tensor::from_slice(&[(rank + 1) as f32; 7], &[7]).unwrap();
            let g_shard = fsdp.reduce_scatter_grad(&pg, &full_grad, ReduceOp::Mean).unwrap();
            g_shard.to_vec::<f32>().unwrap()
        });

        assert_eq!(out[0], vec![2.0, 2.0, 2.0]);
        assert_eq!(out[1], vec![2.0, 2.0, 2.0]);
        // last shard has 1 real element + 2 zero-padded elements
        assert_eq!(out[2], vec![2.0, 0.0, 0.0]);
    }
}
