//! Neural network containers and structural combinators:
//! - [`Identity`] (`torch.nn.Identity`)
//! - [`LambdaModule`] (wraps any closure `Fn(&Tensor) -> Result<Tensor>` into a [`Module`])
//! - [`Residual`] (residual skip wrapper $x + f(x)$ or $g(x) + f(x)$)
//! - [`ModuleList`] (`torch.nn.ModuleList`)
//! - [`ModuleDict`] (`torch.nn.ModuleDict`)
//! - [`ParallelAdd`] and [`ParallelCat`] (multi-branch parallel execution with sum or concat merge)

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::Tensor;

/// Identity module (`torch.nn.Identity`): returns its input unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

/// Function closure signature stored in a [`LambdaModule`].
pub type LambdaFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

/// Wraps a stateless function or closure `Fn(&Tensor) -> Result<Tensor>` into a [`Module`].
pub struct LambdaModule {
    func: LambdaFn,
}

impl LambdaModule {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    {
        Self { func: Box::new(f) }
    }
}

impl Module for LambdaModule {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        (self.func)(input)
    }
}

/// Generic Residual Skip-Connection wrapper: computes `shortcut(x) + body(x)`
/// (or `x + body(x)` when `shortcut` is `None`).
pub struct Residual<M: Module> {
    pub body: M,
    pub shortcut: Option<Box<dyn Module + Send + Sync>>,
}

impl<M: Module> Residual<M> {
    pub fn new(body: M) -> Self {
        Self {
            body,
            shortcut: None,
        }
    }

    pub fn with_shortcut(body: M, shortcut: impl Module + Send + Sync + 'static) -> Self {
        Self {
            body,
            shortcut: Some(Box::new(shortcut)),
        }
    }
}

impl<M: Module> Module for Residual<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let fx = self.body.forward(input)?;
        match &self.shortcut {
            Some(sc) => sc.forward(input)?.add(&fx),
            None => input.add(&fx),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.body.named_parameters() {
            out.push((format!("body.{k}"), v));
        }
        if let Some(sc) = &self.shortcut {
            for (k, v) in sc.named_parameters() {
                out.push((format!("shortcut.{k}"), v));
            }
        }
        out
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.body.named_buffers() {
            out.push((format!("body.{k}"), v));
        }
        if let Some(sc) = &self.shortcut {
            for (k, v) in sc.named_buffers() {
                out.push((format!("shortcut.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        self.body.train(mode);
        if let Some(sc) = &self.shortcut {
            sc.train(mode);
        }
    }
}

/// Indexable list of neural network sub-modules (`torch.nn.ModuleList`) that registers
/// all parameters and propagates `.train(mode)` / `.eval()`.
#[derive(Default)]
pub struct ModuleList {
    modules: Vec<Box<dyn Module + Send + Sync>>,
}

impl ModuleList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, module: impl Module + Send + Sync + 'static) {
        self.modules.push(Box::new(module));
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, module: impl Module + Send + Sync + 'static) -> Self {
        self.push(module);
        self
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&(dyn Module + Send + Sync)> {
        self.modules.get(index).map(AsRef::as_ref)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(dyn Module + Send + Sync)> {
        self.modules.iter().map(AsRef::as_ref)
    }
}

impl Module for ModuleList {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for m in &self.modules {
            x = m.forward(&x)?;
        }
        Ok(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (i, m) in self.modules.iter().enumerate() {
            for (k, v) in m.named_parameters() {
                params.push((format!("{i}.{k}"), v));
            }
        }
        params
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, m) in self.modules.iter().enumerate() {
            for (k, v) in m.named_buffers() {
                out.push((format!("{i}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        for m in &self.modules {
            m.train(mode);
        }
    }
}

/// Ordered dictionary of named neural network sub-modules (`torch.nn.ModuleDict`).
#[derive(Default)]
pub struct ModuleDict {
    entries: Vec<(String, Box<dyn Module + Send + Sync>)>,
}

impl ModuleDict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, module: impl Module + Send + Sync + 'static) {
        let k = key.into();
        if let Some(slot) = self.entries.iter_mut().find(|(ek, _)| *ek == k) {
            slot.1 = Box::new(module);
        } else {
            self.entries.push((k, Box::new(module)));
        }
    }

    pub fn with(
        mut self,
        key: impl Into<String>,
        module: impl Module + Send + Sync + 'static,
    ) -> Self {
        self.insert(key, module);
        self
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn get(&self, key: &str) -> Option<&(dyn Module + Send + Sync)> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, m)| m.as_ref())
    }

    pub fn forward_key(&self, key: &str, input: &Tensor) -> Result<Tensor> {
        let m = self
            .get(key)
            .ok_or_else(|| Error::Msg(format!("ModuleDict: unknown key '{key}'")))?;
        m.forward(input)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Module for ModuleDict {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (_, m) in &self.entries {
            x = m.forward(&x)?;
        }
        Ok(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (name, m) in &self.entries {
            for (k, v) in m.named_parameters() {
                params.push((format!("{name}.{k}"), v));
            }
        }
        params
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (name, m) in &self.entries {
            for (k, v) in m.named_buffers() {
                out.push((format!("{name}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        for (_, m) in &self.entries {
            m.train(mode);
        }
    }
}

/// Multi-branch parallel container that evaluates all branches on `input` and sums their outputs:
/// $y = \sum_{b=1}^B f_b(x)$.
#[derive(Default)]
pub struct ParallelAdd {
    branches: Vec<Box<dyn Module + Send + Sync>>,
}

impl ParallelAdd {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, branch: impl Module + Send + Sync + 'static) -> Self {
        self.branches.push(Box::new(branch));
        self
    }
}

impl Module for ParallelAdd {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.branches.is_empty() {
            return Ok(input.clone());
        }
        let mut acc = self.branches[0].forward(input)?;
        for b in &self.branches[1..] {
            acc = acc.add(&b.forward(input)?)?;
        }
        Ok(acc)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.branches.iter().enumerate() {
            for (k, v) in b.named_parameters() {
                out.push((format!("branch_{i}.{k}"), v));
            }
        }
        out
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.branches.iter().enumerate() {
            for (k, v) in b.named_buffers() {
                out.push((format!("branch_{i}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        for b in &self.branches {
            b.train(mode);
        }
    }
}

/// Multi-branch parallel container that evaluates all branches on `input` and concatenates
/// their outputs along `axis` (Inception / DenseNet style).
pub struct ParallelCat {
    branches: Vec<Box<dyn Module + Send + Sync>>,
    pub axis: isize,
}

impl ParallelCat {
    pub fn new(axis: isize) -> Self {
        Self {
            branches: Vec::new(),
            axis,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, branch: impl Module + Send + Sync + 'static) -> Self {
        self.branches.push(Box::new(branch));
        self
    }
}

impl Module for ParallelCat {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.branches.is_empty() {
            return Ok(input.clone());
        }
        let mut outs = Vec::with_capacity(self.branches.len());
        for b in &self.branches {
            outs.push(b.forward(input)?);
        }
        Tensor::cat(&outs, self.axis)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.branches.iter().enumerate() {
            for (k, v) in b.named_parameters() {
                out.push((format!("branch_{i}.{k}"), v));
            }
        }
        out
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, b) in self.branches.iter().enumerate() {
            for (k, v) in b.named_buffers() {
                out.push((format!("branch_{i}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        for b in &self.branches {
            b.train(mode);
        }
    }
}
