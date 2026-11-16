//! Training checkpoints that resume a run exactly: model tensors (sharded across files), optimizer state, the global RNG
//! state, and the position in training (epoch, step, batch within the epoch) plus free-form scalars.
//!
//! A checkpoint is a directory:
//! `model-00001-of-0000N.safetensors` â€¦ `model.index.json` Â· `optimizer.safetensors` Â· `state.txt`.
//! It is first written to `<dir>.tmp` and renamed into place, so a crash never leaves a half-written checkpoint visible.
//! Tensors are addressed by name, so a checkpoint written with one shard size loads with no regard to the sharding
//! (resharding is free), and [`CheckpointManager`] keeps the last N and finds the latest for `--resume`.

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::optim::{Optimizer, OptimizerState};
use crate::serialize::{load_safetensors, serialize_safetensors_with_metadata};
use crate::tensor::Tensor;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error::Msg(format!("checkpoint: {what}: {e}")))
}

/// Everything needed to continue a run.
#[derive(Clone, Default)]
pub struct TrainState {
    pub model: HashMap<String, Tensor>,
    pub optimizer: Option<OptimizerState>,
    pub rng: Vec<u8>,
    pub epoch: u64,
    pub step: u64,
    /// Batches of `epoch` already consumed (feed to `PrefetchLoader::epoch_from`).
    pub batch_in_epoch: u64,
    pub extra: BTreeMap<String, f64>,
}

impl TrainState {
    /// Snapshots the model, the optimizer (if any) and the global RNG. Tensors are copied, so training can go on while the
    /// state is written.
    pub fn capture(model: &dyn Module, optimizer: Option<&dyn Optimizer>, epoch: u64, step: u64, batch_in_epoch: u64) -> Result<Self> {
        let copy = |t: &Tensor| t.detach().contiguous()?.mul_scalar(1.0);
        let mut m = HashMap::new();
        for (k, t) in model.state_dict() {
            m.insert(k, if t.dtype().is_floating_point() { copy(&t)? } else { t.detach().contiguous()? });
        }
        let optimizer = match optimizer {
            Some(o) => {
                let mut s = o.state_dict();
                for t in s.tensors.values_mut() {
                    *t = copy(t)?;
                }
                Some(s)
            }
            None => None,
        };
        Ok(Self { model: m, optimizer, rng: crate::rng::get_rng_state(), epoch, step, batch_in_epoch, extra: BTreeMap::new() })
    }

    /// Puts the state back: model weights, optimizer, and the global RNG.
    pub fn restore(&self, model: &dyn Module, optimizer: Option<&mut dyn Optimizer>) -> Result<()> {
        model.load_state_dict(&self.model)?;
        if let (Some(o), Some(s)) = (optimizer, &self.optimizer) {
            o.load_state_dict(s)?;
        }
        if !self.rng.is_empty() {
            crate::rng::set_rng_state(&self.rng)?;
        }
        Ok(())
    }
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.numel() * t.dtype().size_in_bytes()
}

/// A serialized checkpoint, ready to be written (possibly from another thread).
pub struct EncodedCheckpoint {
    files: Vec<(String, Vec<u8>)>,
}

/// Serializes `state` into files, splitting the model into shards of at most `max_shard_bytes` (a single tensor larger
/// than that gets a shard of its own).
pub fn encode(state: &TrainState, max_shard_bytes: usize) -> Result<EncodedCheckpoint> {
    let mut names: Vec<&String> = state.model.keys().collect();
    names.sort();
    let mut shards: Vec<HashMap<String, Tensor>> = vec![HashMap::new()];
    let mut used = 0usize;
    for n in names {
        let t = &state.model[n];
        let sz = tensor_bytes(t);
        if used > 0 && used + sz > max_shard_bytes {
            shards.push(HashMap::new());
            used = 0;
        }
        shards.last_mut().unwrap().insert(n.clone(), t.clone());
        used += sz;
    }
    let total = shards.len();
    let mut files = Vec::new();
    let mut index = Vec::new();
    for (i, shard) in shards.iter().enumerate() {
        let fname = format!("model-{:05}-of-{:05}.safetensors", i + 1, total);
        let mut keys: Vec<&String> = shard.keys().collect();
        keys.sort();
        for k in keys {
            index.push(format!("    \"{}\": \"{}\"", k.replace('\\', "\\\\").replace('"', "\\\""), fname));
        }
        files.push((fname, serialize_safetensors_with_metadata(shard, None)?));
    }
    files.push(("model.index.json".into(), format!("{{\n  \"weight_map\": {{\n{}\n  }}\n}}\n", index.join(",\n")).into_bytes()));
    if let Some(o) = &state.optimizer {
        let mut all = o.tensors.clone();
        for (k, &v) in &o.scalars {
            all.insert(format!("__scalar__.{k}"), Tensor::from_slice(&[v], &[1])?);
        }
        files.push(("optimizer.safetensors".into(), serialize_safetensors_with_metadata(&all, None)?));
    }
    let mut txt = format!("epoch={}\nstep={}\nbatch_in_epoch={}\nrng={}\n", state.epoch, state.step, state.batch_in_epoch, hex(&state.rng));
    for (k, v) in &state.extra {
        txt.push_str(&format!("extra.{k}={}\n", v.to_bits()));
    }
    files.push(("state.txt".into(), txt.into_bytes()));
    Ok(EncodedCheckpoint { files })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn unhex(s: &str) -> Result<Vec<u8>> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| Error::Msg("checkpoint: bad rng hex".into()))).collect()
}

impl EncodedCheckpoint {
    /// Writes to `<dir>.tmp` then renames to `dir` (replacing an older checkpoint at that path).
    pub fn write(&self, dir: &Path) -> Result<()> {
        let tmp = PathBuf::from(format!("{}.tmp", dir.display()));
        if tmp.exists() {
            io(std::fs::remove_dir_all(&tmp), "clearing stale temp dir")?;
        }
        io(std::fs::create_dir_all(&tmp), "creating temp dir")?;
        for (name, bytes) in &self.files {
            io(std::fs::write(tmp.join(name), bytes), name)?;
        }
        if dir.exists() {
            io(std::fs::remove_dir_all(dir), "replacing old checkpoint")?;
        }
        io(std::fs::rename(&tmp, dir), "publishing checkpoint")
    }
}

/// Saves `state` to `dir` (blocking).
pub fn save_checkpoint(dir: impl AsRef<Path>, state: &TrainState, max_shard_bytes: usize) -> Result<()> {
    encode(state, max_shard_bytes)?.write(dir.as_ref())
}

/// Serializes now and writes on a background thread; join the handle to learn whether the write succeeded.
pub fn save_checkpoint_async(dir: impl AsRef<Path>, state: &TrainState, max_shard_bytes: usize) -> Result<std::thread::JoinHandle<Result<()>>> {
    let enc = encode(state, max_shard_bytes)?;
    let dir = dir.as_ref().to_path_buf();
    Ok(std::thread::spawn(move || enc.write(&dir)))
}

/// Loads a checkpoint directory.
pub fn load_checkpoint(dir: impl AsRef<Path>) -> Result<TrainState> {
    let dir = dir.as_ref();
    let mut st = TrainState::default();
    let mut shard_files: Vec<PathBuf> = io(std::fs::read_dir(dir), "reading directory")?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors")))
        .collect();
    shard_files.sort();
    if shard_files.is_empty() {
        return Err(Error::Msg(format!("checkpoint: no model shards in {}", dir.display())));
    }
    for f in shard_files {
        st.model.extend(load_safetensors(&f)?);
    }
    let opt = dir.join("optimizer.safetensors");
    if opt.exists() {
        st.optimizer = Some(OptimizerState::load(&opt)?);
    }
    let text = io(std::fs::read_to_string(dir.join("state.txt")), "state.txt")?;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        let num = |v: &str| v.parse::<u64>().map_err(|_| Error::Msg(format!("checkpoint: bad value for {k}")));
        match k {
            "epoch" => st.epoch = num(v)?,
            "step" => st.step = num(v)?,
            "batch_in_epoch" => st.batch_in_epoch = num(v)?,
            "rng" => st.rng = unhex(v)?,
            _ => {
                if let Some(name) = k.strip_prefix("extra.") {
                    st.extra.insert(name.to_string(), f64::from_bits(num(v)?));
                }
            }
        }
    }
    Ok(st)
}

/// Keeps `step-<n>` checkpoints under a root directory: saves, finds the latest, deletes all but the newest `keep`.
pub struct CheckpointManager {
    root: PathBuf,
    keep: usize,
    pending: Option<std::thread::JoinHandle<Result<()>>>,
    pub max_shard_bytes: usize,
}

impl CheckpointManager {
    pub fn new(root: impl AsRef<Path>, keep: usize) -> Result<Self> {
        io(std::fs::create_dir_all(root.as_ref()), "creating root")?;
        Ok(Self { root: root.as_ref().to_path_buf(), keep: keep.max(1), pending: None, max_shard_bytes: 1 << 30 })
    }

    fn path(&self, step: u64) -> PathBuf {
        self.root.join(format!("step-{step:010}"))
    }

    /// Waits for an outstanding async save.
    pub fn wait(&mut self) -> Result<()> {
        match self.pending.take() {
            Some(h) => h.join().map_err(|_| Error::Msg("checkpoint: writer thread panicked".into()))?,
            None => Ok(()),
        }
    }

    /// Saves `state` as step `state.step`; with `background` the write happens on another thread (the previous one is
    /// awaited first, so at most one is in flight).
    pub fn save(&mut self, state: &TrainState, background: bool) -> Result<()> {
        self.wait()?;
        let dir = self.path(state.step);
        if background {
            self.pending = Some(save_checkpoint_async(&dir, state, self.max_shard_bytes)?);
        } else {
            save_checkpoint(&dir, state, self.max_shard_bytes)?;
            self.prune()?;
        }
        Ok(())
    }

    /// Steps with a complete checkpoint on disk, ascending.
    pub fn steps(&self) -> Vec<u64> {
        let mut v: Vec<u64> = std::fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("step-")).and_then(|n| n.parse().ok()))
            .collect();
        v.sort_unstable();
        v
    }

    pub fn latest(&mut self) -> Result<Option<TrainState>> {
        self.wait()?;
        self.prune()?;
        match self.steps().last() {
            Some(&s) => Ok(Some(load_checkpoint(self.path(s))?)),
            None => Ok(None),
        }
    }

    fn prune(&self) -> Result<()> {
        let steps = self.steps();
        for s in steps.iter().take(steps.len().saturating_sub(self.keep)) {
            io(std::fs::remove_dir_all(self.path(*s)), "pruning")?;
        }
        Ok(())
    }
}

impl Drop for CheckpointManager {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::optim::Adam;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oxideon_ckpt_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn train_steps(m: &Linear, opt: &mut Adam, n: usize) -> Vec<f32> {
        let x = Tensor::from_slice(&[0.5f32, -1.0, 2.0, 0.25], &[2, 2]).unwrap();
        (0..n)
            .map(|_| {
                opt.zero_grad();
                let noise = Tensor::randn(&[2, 2]).unwrap().mul_scalar(0.01).unwrap(); // consumes the global RNG
                let loss = m.forward(&x.add(&noise).unwrap()).unwrap().square().unwrap().mean().unwrap();
                loss.backward().unwrap();
                opt.step().unwrap();
                loss.item::<f32>().unwrap()
            })
            .collect()
    }

    #[test]
    fn resumed_run_is_bit_identical_to_an_uninterrupted_one() {
        let make = || {
            crate::rng::manual_seed(11);
            let m = Linear::new(2, 2).unwrap();
            let opt = Adam::new(m.parameters(), 0.05);
            (m, opt)
        };
        // reference: 8 uninterrupted steps
        let (m, mut opt) = make();
        let reference = train_steps(&m, &mut opt, 8);
        let ref_w = m.state_dict()["weight"].to_vec::<f32>().unwrap();

        // 4 steps, checkpoint (tiny shards force several files), then a brand-new process-like restart
        let dir = tmp("resume");
        let (m, mut opt) = make();
        let first = train_steps(&m, &mut opt, 4);
        let st = TrainState::capture(&m, Some(&opt), 0, 4, 4).unwrap();
        save_checkpoint(&dir, &st, 8).unwrap();
        assert!(!PathBuf::from(format!("{}.tmp", dir.display())).exists());
        assert!(std::fs::read_dir(&dir).unwrap().count() >= 4); // 2+ shards, index, optimizer, state

        crate::rng::manual_seed(999); // the restarted run starts with unrelated RNG and weights
        let m2 = Linear::new(2, 2).unwrap();
        let mut opt2 = Adam::new(m2.parameters(), 0.05);
        let loaded = load_checkpoint(&dir).unwrap();
        assert_eq!((loaded.epoch, loaded.step, loaded.batch_in_epoch), (0, 4, 4));
        loaded.restore(&m2, Some(&mut opt2)).unwrap();
        let second = train_steps(&m2, &mut opt2, 4);
        assert_eq!([first, second].concat(), reference);
        assert_eq!(m2.state_dict()["weight"].to_vec::<f32>().unwrap(), ref_w);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manager_keeps_the_newest_finds_the_latest_and_saves_in_background() {
        let root = tmp("manager");
        let m = Linear::new(2, 2).unwrap();
        let mut mgr = CheckpointManager::new(&root, 2).unwrap();
        for step in [10u64, 20, 30] {
            let mut st = TrainState::capture(&m, None, 1, step, 0).unwrap();
            st.extra.insert("best_loss".into(), 0.125 * step as f64);
            mgr.save(&st, step == 30).unwrap(); // the last one goes through the async path
        }
        let latest = mgr.latest().unwrap().unwrap();
        assert_eq!(mgr.steps(), vec![20, 30]);
        assert_eq!(latest.step, 30);
        assert_eq!(latest.extra["best_loss"], 3.75);
        assert!(latest.optimizer.is_none());
        let empty = CheckpointManager::new(tmp("empty"), 1).unwrap().latest().unwrap();
        assert!(empty.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
