//! Multi-dimensional `StringTensor` for text, label, and object-like tensor workloads.

use crate::error::{Error, Result};

/// An N-dimensional tensor of UTF-8 strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringTensor {
    data: Vec<String>,
    shape: Vec<usize>,
}

impl StringTensor {
    pub fn from_vec(data: Vec<String>, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(Error::Msg(format!(
                "StringTensor: data len {} does not match shape {:?} (numel {})",
                data.len(),
                shape,
                numel
            )));
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn from_strs(data: &[&str], shape: &[usize]) -> Result<Self> {
        Self::from_vec(data.iter().map(|s| (*s).to_string()).collect(), shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[String] {
        &self.data
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let numel: usize = new_shape.iter().product();
        if numel != self.data.len() {
            return Err(Error::Msg(format!(
                "StringTensor::reshape: cannot reshape {} elements to {:?}",
                self.data.len(),
                new_shape
            )));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        })
    }

    pub fn eq_mask(&self, target: &str) -> Vec<bool> {
        self.data.iter().map(|s| s == target).collect()
    }

    pub fn contains_mask(&self, pat: &str) -> Vec<bool> {
        self.data.iter().map(|s| s.contains(pat)).collect()
    }

    pub fn map_str(&self, f: impl Fn(&str) -> String) -> Self {
        Self {
            data: self.data.iter().map(|s| f(s)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn get(&self, indices: &[usize]) -> Result<&str> {
        if indices.len() != self.shape.len() {
            return Err(Error::Msg("StringTensor::get: rank mismatch".into()));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (&idx, &dim) in indices.iter().rev().zip(self.shape.iter().rev()) {
            if idx >= dim {
                return Err(Error::Msg("StringTensor::get: index out of bounds".into()));
            }
            flat += idx * stride;
            stride *= dim;
        }
        Ok(&self.data[flat])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_tensor_ops() {
        let st = StringTensor::from_strs(&["cat", "dog", "bird", "fish"], &[2, 2]).unwrap();
        assert_eq!(st.get(&[1, 0]).unwrap(), "bird");
        assert_eq!(st.eq_mask("dog"), vec![false, true, false, false]);
        let upper = st.map_str(|s| s.to_uppercase());
        assert_eq!(upper.get(&[0, 1]).unwrap(), "DOG");
    }
}
