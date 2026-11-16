//! Resumable training checkpoints: model weights, optimizer state and a small metadata record per step, written
//! atomically into a directory, with `keep_last` rotation and [`CheckpointManager::latest`] for resuming.
//!
//! Layout: `<dir>/step-<N>/{model.safetensors, optimizer.safetensors, meta.txt}` (`meta.txt` holds `key=value` lines).

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::optim::{Optimizer, OptimizerState};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Metadata stored next to the tensors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckpointMeta {
    pub step: u64,
    pub epoch: u64,
    /// Free-form numeric values such as the best validation loss.
    pub values: BTreeMap<String, f64>,
}

pub struct CheckpointManager {
    dir: PathBuf,
    pub keep_last: usize,
}

impl CheckpointManager {
    pub fn new(dir: impl AsRef<Path>, keep_last: usize) -> Result<Self> {
        std::fs::create_dir_all(dir.as_ref()).map_err(|e| Error::Msg(format!("checkpoint dir: {e}")))?;
        Ok(Self { dir: dir.as_ref().to_path_buf(), keep_last: keep_last.max(1) })
    }

    fn step_dir(&self, step: u64) -> PathBuf {
        self.dir.join(format!("step-{step}"))
    }

    /// Steps present on disk, ascending (incomplete `.tmp` directories are ignored).
    pub fn steps(&self) -> Vec<u64> {
        let mut v: Vec<u64> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.strip_prefix("step-")?.parse().ok())
            .collect();
        v.sort_unstable();
        v
    }

    pub fn latest(&self) -> Option<u64> {
        self.steps().last().copied()
    }

    /// Writes a checkpoint for `meta.step` (into a temporary directory renamed on success) and prunes old ones.
    pub fn save(&self, model: &dyn Module, optimizer: &dyn Optimizer, meta: &CheckpointMeta) -> Result<PathBuf> {
        let tmp = self.dir.join(format!("step-{}.tmp", meta.step));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(|e| Error::Msg(format!("checkpoint: {e}")))?;
        crate::serialize::save_safetensors(&model.state_dict(), tmp.join("model.safetensors"))?;
        optimizer.state_dict().save(tmp.join("optimizer.safetensors"))?;
        let mut text = format!("step={}\nepoch={}\n", meta.step, meta.epoch);
        for (k, v) in &meta.values {
            if k.contains(['=', '\n']) {
                return Err(Error::Msg(format!("checkpoint: invalid meta key {k:?}")));
            }
            text.push_str(&format!("value.{k}={v:?}\n"));
        }
        std::fs::write(tmp.join("meta.txt"), text).map_err(|e| Error::Msg(format!("checkpoint: {e}")))?;
        let dest = self.step_dir(meta.step);
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::rename(&tmp, &dest).map_err(|e| Error::Msg(format!("checkpoint: {e}")))?;
        let steps = self.steps();
        for old in &steps[..steps.len().saturating_sub(self.keep_last)] {
            let _ = std::fs::remove_dir_all(self.step_dir(*old));
        }
        Ok(dest)
    }

    /// Restores `model` and `optimizer` from the checkpoint of `step` and returns its metadata.
    pub fn load(&self, step: u64, model: &dyn Module, optimizer: &mut dyn Optimizer) -> Result<CheckpointMeta> {
        let d = self.step_dir(step);
        if !d.is_dir() {
            return Err(Error::Msg(format!("checkpoint: step {step} not found in {}", self.dir.display())));
        }
        model.load_state_dict(&crate::serialize::load_safetensors(d.join("model.safetensors"))?)?;
        optimizer.load_state_dict(&OptimizerState::load(d.join("optimizer.safetensors"))?)?;
        let text = std::fs::read_to_string(d.join("meta.txt")).map_err(|e| Error::Msg(format!("checkpoint: {e}")))?;
        let mut meta = CheckpointMeta::default();
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            match k {
                "step" => meta.step = v.parse().map_err(|_| Error::Msg("checkpoint: bad step".into()))?,
                "epoch" => meta.epoch = v.parse().map_err(|_| Error::Msg("checkpoint: bad epoch".into()))?,
                _ => {
                    if let Some(name) = k.strip_prefix("value.") {
                        meta.values.insert(name.to_string(), v.parse().map_err(|_| Error::Msg(format!("checkpoint: bad value for {name}")))?);
                    }
                }
            }
        }
        Ok(meta)
    }

    /// Restores the newest checkpoint, if any.
    pub fn resume(&self, model: &dyn Module, optimizer: &mut dyn Optimizer) -> Result<Option<CheckpointMeta>> {
        match self.latest() {
            Some(s) => self.load(s, model, optimizer).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::optim::Adam;
    use crate::tensor::Tensor;

    fn train_step(model: &Linear, opt: &mut Adam, x: &Tensor) {
        opt.zero_grad();
        model.forward(x).unwrap().square().unwrap().mean().unwrap().backward().unwrap();
        opt.step().unwrap();
    }

    #[test]
    fn save_rotate_resume_continues_identically() {
        let dir = std::env::temp_dir().join(format!("oxideon-ckpt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mgr = CheckpointManager::new(&dir, 2).unwrap();
        let x = Tensor::randn(&[4, 3]).unwrap();
        let model = Linear::new(3, 2).unwrap();
        let mut opt = Adam::new(model.parameters(), 0.05);
        assert!(mgr.resume(&model, &mut opt).unwrap().is_none());
        for step in 1..=3u64 {
            train_step(&model, &mut opt, &x);
            let mut meta = CheckpointMeta { step, epoch: 0, ..Default::default() };
            meta.values.insert("loss".into(), 1.0 / step as f64);
            mgr.save(&model, &opt, &meta).unwrap();
        }
        assert_eq!(mgr.steps(), vec![2, 3], "keep_last = 2 prunes step 1");
        // one more step from the live run
        train_step(&model, &mut opt, &x);
        // a fresh model/optimizer resumed from step 3 must take the same step
        let model2 = Linear::new(3, 2).unwrap();
        let mut opt2 = Adam::new(model2.parameters(), 0.05);
        let meta = mgr.resume(&model2, &mut opt2).unwrap().unwrap();
        assert_eq!((meta.step, meta.values["loss"]), (3, 1.0 / 3.0));
        train_step(&model2, &mut opt2, &x);
        let (a, b) = (model.weight.to_vec::<f32>().unwrap(), model2.weight.to_vec::<f32>().unwrap());
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-6), "{a:?} vs {b:?}");
        assert!(mgr.load(99, &model2, &mut opt2).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
