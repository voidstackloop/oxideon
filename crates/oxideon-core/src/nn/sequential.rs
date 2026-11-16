//! Sequential module container.

use super::Module;
use crate::error::Result;
use crate::tensor::Tensor;

#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module + Send + Sync>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: impl Module + Send + Sync + 'static) -> Self {
        self.push(layer);
        self
    }

    pub fn push(&mut self, layer: impl Module + Send + Sync + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (name, p) in l.named_parameters() {
                params.push((format!("{i}.{name}"), p));
            }
        }
        params
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (name, b) in l.named_buffers() {
                out.push((format!("{i}.{name}"), b));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        self.layers.iter().for_each(|l| l.train(mode));
    }
}
