//! Name-based parameter selection for any [`Module`]: partial freezing (`freeze_matching`), per-submodule parameter
//! counts, and the standard "no weight decay on biases and norm scales" split for optimizer groups.

use super::Module;
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// Sets `requires_grad = !freeze` on every parameter whose name starts with `prefix`; returns how many changed.
pub fn freeze_matching(m: &dyn Module, prefix: &str, freeze: bool) -> usize {
    let mut n = 0;
    for (name, p) in m.named_parameters() {
        if name.starts_with(prefix) && p.set_requires_grad(!freeze).is_ok() {
            n += 1;
        }
    }
    n
}

/// Parameter counts grouped by the first `depth` dot-separated name components.
pub fn param_counts(m: &dyn Module, depth: usize) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (name, p) in m.named_parameters() {
        let key = name.split('.').take(depth).collect::<Vec<_>>().join(".");
        *out.entry(key).or_insert(0) += p.numel();
    }
    out
}

/// Splits parameters into `(decay, no_decay)`: biases, 1-D tensors (norm scales/shifts) and names containing
/// `norm`/`bias`/`embed` go to `no_decay`.
pub fn split_decay(m: &dyn Module) -> (Vec<Tensor>, Vec<Tensor>) {
    let (mut decay, mut none) = (Vec::new(), Vec::new());
    for (name, p) in m.named_parameters() {
        let l = name.to_lowercase();
        if p.ndim() <= 1 || l.contains("bias") || l.contains("norm") || l.contains("embed") {
            none.push(p);
        } else {
            decay.push(p);
        }
    }
    (decay, none)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, ReLU, Sequential};

    #[test]
    fn freeze_count_split() {
        let net = Sequential::new().add(Linear::new(3, 4).unwrap()).add(ReLU).add(Linear::new(4, 2).unwrap());
        let names: Vec<String> = net.named_parameters().into_iter().map(|(k, _)| k).collect();
        let first = names[0].split('.').next().unwrap().to_string();
        let counts = param_counts(&net, 1);
        assert_eq!(counts.values().sum::<usize>(), net.num_parameters());
        assert!(counts.len() >= 2, "{counts:?}");
        let k = freeze_matching(&net, &first, true);
        assert!(k >= 1 && net.num_trainable_parameters() < net.num_parameters());
        freeze_matching(&net, &first, false);
        assert_eq!(net.num_trainable_parameters(), net.num_parameters());
        let (decay, none) = split_decay(&net);
        assert_eq!(decay.len(), 2); // two weight matrices
        assert_eq!(none.len(), 2); // two biases
    }
}
