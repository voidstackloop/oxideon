//! Module hooks (`register_forward_pre_hook` / `register_forward_hook`): wrap any [`Module`] in [`Hooked`] to inspect
//! or rewrite its input and output, for feature extraction, debugging and activation editing. [`FeatureRecorder`]
//! collects detached outputs.

use super::Module;
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Called with the input before the module runs; returning `Some(t)` replaces the input.
pub type PreHook = Box<dyn Fn(&Tensor) -> Result<Option<Tensor>> + Send + Sync>;
/// Called with `(input, output)` after the module runs; returning `Some(t)` replaces the output.
pub type PostHook = Box<dyn Fn(&Tensor, &Tensor) -> Result<Option<Tensor>> + Send + Sync>;

/// Identifies a registered hook for [`Hooked::remove_hook`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HookHandle(usize);

/// A module with forward hooks.
pub struct Hooked<M: Module> {
    pub inner: M,
    pre: Mutex<Vec<(usize, PreHook)>>,
    post: Mutex<Vec<(usize, PostHook)>>,
    next: AtomicUsize,
}

impl<M: Module> Hooked<M> {
    pub fn new(inner: M) -> Self {
        Self { inner, pre: Mutex::new(Vec::new()), post: Mutex::new(Vec::new()), next: AtomicUsize::new(0) }
    }

    pub fn register_forward_pre_hook(&self, hook: PreHook) -> HookHandle {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.pre.lock().expect("hook lock").push((id, hook));
        HookHandle(id)
    }

    pub fn register_forward_hook(&self, hook: PostHook) -> HookHandle {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.post.lock().expect("hook lock").push((id, hook));
        HookHandle(id)
    }

    /// Removes a hook; returns whether it existed.
    pub fn remove_hook(&self, handle: HookHandle) -> bool {
        let mut removed = false;
        let mut pre = self.pre.lock().expect("hook lock");
        let before = pre.len();
        pre.retain(|(i, _)| *i != handle.0);
        removed |= pre.len() != before;
        let mut post = self.post.lock().expect("hook lock");
        let before = post.len();
        post.retain(|(i, _)| *i != handle.0);
        removed |= post.len() != before;
        removed
    }
}

impl<M: Module> Module for Hooked<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (_, h) in self.pre.lock().expect("hook lock").iter() {
            if let Some(replacement) = h(&x)? {
                x = replacement;
            }
        }
        let mut out = self.inner.forward(&x)?;
        for (_, h) in self.post.lock().expect("hook lock").iter() {
            if let Some(replacement) = h(&x, &out)? {
                out = replacement;
            }
        }
        Ok(out)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.inner.named_parameters()
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.inner.named_buffers()
    }
    fn train(&self, mode: bool) {
        self.inner.train(mode);
    }
}

/// Collects (detached) module outputs through a forward hook.
#[derive(Clone, Default)]
pub struct FeatureRecorder {
    store: Arc<Mutex<Vec<Tensor>>>,
}

impl FeatureRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A hook that records every output.
    pub fn hook(&self) -> PostHook {
        let store = Arc::clone(&self.store);
        Box::new(move |_, out| {
            store.lock().expect("recorder lock").push(out.detach());
            Ok(None)
        })
    }

    /// The outputs recorded so far, oldest first.
    pub fn outputs(&self) -> Vec<Tensor> {
        self.store.lock().expect("recorder lock").clone()
    }

    pub fn clear(&self) {
        self.store.lock().expect("recorder lock").clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, ReLU, Sequential};

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    #[test]
    fn hooks_observe_and_rewrite() {
        let lin = Hooked::new(Linear::new(2, 2).unwrap());
        lin.inner.weight.assign(&Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap()).unwrap();
        if let Some(b) = &lin.inner.bias {
            b.assign(&Tensor::zeros_f32(&[2]).unwrap()).unwrap();
        }
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2]).unwrap();
        assert_eq!(v(&lin.forward(&x).unwrap()), vec![1.0, 2.0]);
        // a pre-hook doubles the input, a post-hook adds one
        let pre = lin.register_forward_pre_hook(Box::new(|x| Ok(Some(x.mul_scalar(2.0)?))));
        let post = lin.register_forward_hook(Box::new(|_, y| Ok(Some(y.add_scalar(1.0)?))));
        assert_eq!(v(&lin.forward(&x).unwrap()), vec![3.0, 5.0]);
        assert!(lin.remove_hook(pre));
        assert_eq!(v(&lin.forward(&x).unwrap()), vec![2.0, 3.0]);
        assert!(lin.remove_hook(post) && !lin.remove_hook(post));
        assert_eq!(v(&lin.forward(&x).unwrap()), vec![1.0, 2.0]);
        assert_eq!(lin.parameters().len(), 2);
    }

    #[test]
    fn recorder_collects_intermediate_features() {
        let rec = FeatureRecorder::new();
        let hidden = Hooked::new(Linear::new(3, 4).unwrap());
        hidden.register_forward_hook(rec.hook());
        let net = Sequential::new().add(hidden).add(ReLU).add(Linear::new(4, 1).unwrap());
        let x = Tensor::randn(&[5, 3]).unwrap();
        let out = net.forward(&x).unwrap();
        assert_eq!(out.shape(), [5, 1]);
        let feats = rec.outputs();
        assert_eq!(feats.len(), 1);
        assert_eq!(feats[0].shape(), [5, 4]);
        assert!(!feats[0].requires_grad());
        net.forward(&x).unwrap();
        assert_eq!(rec.outputs().len(), 2);
        rec.clear();
        assert!(rec.outputs().is_empty());
    }
}
