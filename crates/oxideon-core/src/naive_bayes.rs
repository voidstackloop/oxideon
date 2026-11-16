//! Count / binary-feature naive Bayes classifiers ([`MultinomialNb`], [`BernoulliNb`], [`ComplementNb`]) and a
//! ready-made [`TextClassifier`] that tokenizes strings, builds a vocabulary, and predicts class labels.

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::text_ir::tokenize;
use std::collections::BTreeMap;

fn check(x: &Matrix, y: &[usize], op: &str) -> Result<(usize, usize)> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.len() != y.len() || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite() || *v < 0.0)) {
        return Err(Error::Msg(format!("{op}: need a rectangular non-negative matrix with one label per row")));
    }
    Ok((d, y.iter().max().unwrap() + 1))
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b]).then(b.cmp(&a))).unwrap()
}

/// Multinomial naive Bayes for count features with Laplace/Lidstone smoothing `alpha`.
#[derive(Debug, Clone)]
pub struct MultinomialNb {
    log_prior: Vec<f64>,
    log_prob: Matrix, // [class][feature]
}

impl MultinomialNb {
    pub fn fit(x: &Matrix, y: &[usize], alpha: f64) -> Result<Self> {
        let (d, k) = check(x, y, "MultinomialNb")?;
        if alpha < 0.0 {
            return Err(Error::Msg("MultinomialNb: alpha must be non-negative".into()));
        }
        let n = x.len() as f64;
        let mut counts = vec![vec![0.0; d]; k];
        let mut n_class = vec![0.0; k];
        for (r, &c) in x.iter().zip(y) {
            n_class[c] += 1.0;
            r.iter().enumerate().for_each(|(j, v)| counts[c][j] += v);
        }
        let log_prior = n_class.iter().map(|&c| if c > 0.0 { (c / n).ln() } else { f64::NEG_INFINITY }).collect();
        let log_prob = counts
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum::<f64>() + alpha * d as f64;
                row.iter().map(|c| ((c + alpha) / total.max(1e-300)).ln()).collect()
            })
            .collect();
        Ok(Self { log_prior, log_prob })
    }

    /// Unnormalized log posterior per class for every row.
    pub fn joint_log_proba(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| (0..self.log_prior.len()).map(|c| self.log_prior[c] + r.iter().zip(&self.log_prob[c]).map(|(v, lp)| v * lp).sum::<f64>()).collect()).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.joint_log_proba(x).iter().map(|l| argmax(l)).collect()
    }

    /// Normalized class probabilities.
    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        self.joint_log_proba(x)
            .into_iter()
            .map(|l| {
                let m = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = l.iter().map(|v| (v - m).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            })
            .collect()
    }
}

/// Bernoulli naive Bayes: features are binarized at `threshold` and absent features contribute evidence too.
#[derive(Debug, Clone)]
pub struct BernoulliNb {
    log_prior: Vec<f64>,
    log_p: Matrix,
    log_not_p: Matrix,
    pub threshold: f64,
}

impl BernoulliNb {
    pub fn fit(x: &Matrix, y: &[usize], alpha: f64, threshold: f64) -> Result<Self> {
        let (d, k) = check(x, y, "BernoulliNb")?;
        let n = x.len() as f64;
        let mut present = vec![vec![0.0; d]; k];
        let mut n_class = vec![0.0; k];
        for (r, &c) in x.iter().zip(y) {
            n_class[c] += 1.0;
            r.iter().enumerate().filter(|(_, v)| **v > threshold).for_each(|(j, _)| present[c][j] += 1.0);
        }
        let p: Matrix = (0..k).map(|c| (0..d).map(|j| (present[c][j] + alpha) / (n_class[c] + 2.0 * alpha).max(1e-300)).collect()).collect();
        Ok(Self {
            log_prior: n_class.iter().map(|&c| if c > 0.0 { (c / n).ln() } else { f64::NEG_INFINITY }).collect(),
            log_p: p.iter().map(|r| r.iter().map(|v| v.max(1e-300).ln()).collect()).collect(),
            log_not_p: p.iter().map(|r| r.iter().map(|v| (1.0 - v).max(1e-300).ln()).collect()).collect(),
            threshold,
        })
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        x.iter()
            .map(|r| {
                let scores: Vec<f64> = (0..self.log_prior.len()).map(|c| self.log_prior[c] + r.iter().enumerate().map(|(j, &v)| if v > self.threshold { self.log_p[c][j] } else { self.log_not_p[c][j] }).sum::<f64>()).collect();
                argmax(&scores)
            })
            .collect()
    }
}

/// Complement naive Bayes (Rennie et al. 2003): estimates feature weights from the complement of each class, which
/// copes better with imbalanced text data.
#[derive(Debug, Clone)]
pub struct ComplementNb {
    weights: Matrix,
}

impl ComplementNb {
    /// `norm = false` matches scikit-learn's default (weights are the negated log complement probabilities); `true`
    /// applies the weight normalization of Rennie et al. (each class's weights divided by their absolute sum).
    pub fn fit(x: &Matrix, y: &[usize], alpha: f64, norm: bool) -> Result<Self> {
        let (d, k) = check(x, y, "ComplementNb")?;
        let mut total = vec![0.0; d];
        let mut per_class = vec![vec![0.0; d]; k];
        for (r, &c) in x.iter().zip(y) {
            for (j, v) in r.iter().enumerate() {
                per_class[c][j] += v;
                total[j] += v;
            }
        }
        let weights = (0..k)
            .map(|c| {
                let comp: Vec<f64> = (0..d).map(|j| total[j] - per_class[c][j] + alpha).collect();
                let s: f64 = comp.iter().sum();
                let logs: Vec<f64> = comp.iter().map(|v| (v / s).ln()).collect();
                let scale: f64 = if norm { logs.iter().map(|v| v.abs()).sum::<f64>().max(1e-300) } else { 1.0 };
                logs.iter().map(|v| -v / scale).collect() // negated: a document goes to the class whose complement fits it worst
            })
            .collect();
        Ok(Self { weights })
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        x.iter().map(|r| argmax(&self.weights.iter().map(|w| r.iter().zip(w).map(|(a, b)| a * b).sum::<f64>()).collect::<Vec<_>>())).collect()
    }
}

/// Bag-of-words text classifier built on [`MultinomialNb`]; class labels are arbitrary strings.
#[derive(Debug, Clone)]
pub struct TextClassifier {
    vocab: BTreeMap<String, usize>,
    classes: Vec<String>,
    model: MultinomialNb,
}

impl TextClassifier {
    pub fn fit<S: AsRef<str>, L: AsRef<str>>(docs: &[S], labels: &[L], alpha: f64) -> Result<Self> {
        if docs.is_empty() || docs.len() != labels.len() {
            return Err(Error::Msg("TextClassifier: need one label per document".into()));
        }
        let mut vocab: BTreeMap<String, usize> = BTreeMap::new();
        docs.iter().flat_map(|d| tokenize(d.as_ref())).for_each(|t| { let n = vocab.len(); vocab.entry(t).or_insert(n); });
        let mut classes: Vec<String> = labels.iter().map(|l| l.as_ref().to_string()).collect();
        classes.sort();
        classes.dedup();
        let y: Vec<usize> = labels.iter().map(|l| classes.binary_search(&l.as_ref().to_string()).unwrap()).collect();
        let me = Self { vocab, classes, model: MultinomialNb { log_prior: vec![], log_prob: vec![] } };
        let x: Matrix = docs.iter().map(|d| me.vectorize(d.as_ref())).collect();
        let model = MultinomialNb::fit(&x, &y, alpha)?;
        Ok(Self { model, ..me })
    }

    fn vectorize(&self, doc: &str) -> Vec<f64> {
        let mut v = vec![0.0; self.vocab.len()];
        tokenize(doc).iter().filter_map(|t| self.vocab.get(t)).for_each(|&i| v[i] += 1.0);
        v
    }

    pub fn predict(&self, doc: &str) -> &str {
        let l = &self.model.joint_log_proba(&vec![self.vectorize(doc)])[0];
        &self.classes[argmax(l)]
    }

    /// Class probabilities for a document as `(label, probability)`, most likely first.
    pub fn predict_proba(&self, doc: &str) -> Vec<(&str, f64)> {
        let p = &self.model.predict_proba(&vec![self.vectorize(doc)])[0];
        let mut out: Vec<(&str, f64)> = self.classes.iter().map(String::as_str).zip(p.iter().copied()).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multinomial_matches_a_hand_computed_example() {
        // Manning et al. IR book example: docs "Chinese Beijing Chinese" (c), "Chinese Chinese Shanghai" (c),
        // "Chinese Macao" (c), "Tokyo Japan Chinese" (j); test "Chinese Chinese Chinese Tokyo Japan" -> c
        // vocabulary order: Chinese, Beijing, Shanghai, Macao, Tokyo, Japan
        let x: Matrix = vec![vec![2.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]];
        let y = [0, 0, 0, 1];
        let nb = MultinomialNb::fit(&x, &y, 1.0).unwrap();
        // P(Chinese | c) = (5 + 1) / (8 + 6) = 3/7, P(Tokyo | c) = P(Japan | c) = 1/14
        assert!((nb.log_prob[0][0].exp() - 3.0 / 7.0).abs() < 1e-12 && (nb.log_prob[0][4].exp() - 1.0 / 14.0).abs() < 1e-12);
        // P(Chinese | j) = 2 / 9, P(Tokyo | j) = 2 / 9
        assert!((nb.log_prob[1][0].exp() - 2.0 / 9.0).abs() < 1e-12);
        let test = vec![vec![3.0, 0.0, 0.0, 0.0, 1.0, 1.0]];
        assert_eq!(nb.predict(&test), vec![0]);
        let p = nb.predict_proba(&test)[0].clone();
        assert!((p[0] - 0.6898).abs() < 5e-4 && (p[1] - 0.3102).abs() < 5e-4, "{p:?}"); // 0.0003 vs 0.0001 as in the book, exactly normalized
        assert!(MultinomialNb::fit(&x, &y, -1.0).is_err() && MultinomialNb::fit(&vec![vec![-1.0]], &[0], 1.0).is_err());
    }

    #[test]
    fn bernoulli_and_complement_classify_synthetic_data() {
        // class 0 uses words 0-1, class 1 uses words 2-3; every 15th row is flipped (noise), capping accuracy at 56/60
        let x: Matrix = (0..60).map(|i| { let c = i % 2; (0..4).map(|j| if (j / 2 == c) == (i % 15 != 0) { 1.0 + (i % 3) as f64 } else { 0.0 }).collect() }).collect();
        let y: Vec<usize> = (0..60).map(|i| i % 2).collect();
        let acc = |p: Vec<usize>| p.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / 60.0;
        assert!(acc(BernoulliNb::fit(&x, &y, 1.0, 0.0).unwrap().predict(&x)) > 0.9);
        assert!(acc(ComplementNb::fit(&x, &y, 1.0, false).unwrap().predict(&x)) > 0.9);
        assert!(acc(MultinomialNb::fit(&x, &y, 1.0).unwrap().predict(&x)) > 0.9);
    }

    #[test]
    fn text_classifier_end_to_end() {
        let docs = ["great movie loved the acting", "wonderful film great story", "terrible plot awful acting", "boring awful waste of time", "loved it wonderful", "awful boring terrible"];
        let labels = ["pos", "pos", "neg", "neg", "pos", "neg"];
        let clf = TextClassifier::fit(&docs, &labels, 1.0).unwrap();
        assert_eq!(clf.predict("a wonderful story, loved it"), "pos");
        assert_eq!(clf.predict("what an awful boring film"), "neg");
        let p = clf.predict_proba("great acting");
        assert_eq!(p[0].0, "pos");
        assert!((p.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-12);
        // unknown words fall back to the class prior (equal here), but must not panic
        assert!(["pos", "neg"].contains(&clf.predict("zzz qqq")));
        assert!(TextClassifier::fit(&docs, &labels[..2], 1.0).is_err());
    }
}
