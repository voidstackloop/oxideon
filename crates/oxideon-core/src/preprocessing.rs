//! scikit-learn style preprocessing on plain data: [`LabelEncoder`], [`OneHotEncoder`], [`SimpleImputer`],
//! [`RobustScaler`], [`PolynomialFeatures`], [`KBinsDiscretizer`], and row [`normalize`].

use crate::classical::Matrix;
use crate::error::{Error, Result};
use std::collections::BTreeMap;

/// Maps string categories to `0..k` (sorted order) and back.
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    pub classes: Vec<String>,
}

impl LabelEncoder {
    pub fn fit<S: AsRef<str>>(values: &[S]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::Msg("LabelEncoder: no values".into()));
        }
        let mut classes: Vec<String> = values.iter().map(|v| v.as_ref().to_string()).collect();
        classes.sort();
        classes.dedup();
        Ok(Self { classes })
    }

    pub fn transform<S: AsRef<str>>(&self, values: &[S]) -> Result<Vec<usize>> {
        values
            .iter()
            .map(|v| self.classes.binary_search_by(|c| c.as_str().cmp(v.as_ref())).map_err(|_| Error::Msg(format!("LabelEncoder: unseen label {:?}", v.as_ref()))))
            .collect()
    }

    pub fn inverse_transform(&self, codes: &[usize]) -> Result<Vec<String>> {
        codes.iter().map(|&c| self.classes.get(c).cloned().ok_or_else(|| Error::Msg(format!("LabelEncoder: code {c} out of range")))).collect()
    }
}

/// One-hot encoding of one categorical column; unseen categories encode as all zeros when `ignore_unknown`.
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    labels: LabelEncoder,
    pub ignore_unknown: bool,
}

impl OneHotEncoder {
    pub fn fit<S: AsRef<str>>(values: &[S], ignore_unknown: bool) -> Result<Self> {
        Ok(Self { labels: LabelEncoder::fit(values)?, ignore_unknown })
    }

    pub fn categories(&self) -> &[String] {
        &self.labels.classes
    }

    pub fn transform<S: AsRef<str>>(&self, values: &[S]) -> Result<Matrix> {
        let k = self.labels.classes.len();
        values
            .iter()
            .map(|v| {
                let mut row = vec![0.0; k];
                match self.labels.transform(&[v.as_ref()]) {
                    Ok(c) => row[c[0]] = 1.0,
                    Err(e) if !self.ignore_unknown => return Err(e),
                    Err(_) => {}
                }
                Ok(row)
            })
            .collect()
    }
}

/// Replaces `NaN` entries by a per-column statistic learned in `fit`.
#[derive(Debug, Clone)]
pub struct SimpleImputer {
    pub fill: Vec<f64>,
}

/// Statistic used by [`SimpleImputer`].
#[derive(Debug, Clone, Copy)]
pub enum ImputeStrategy {
    Mean,
    Median,
    MostFrequent,
    Constant(f64),
}

impl SimpleImputer {
    pub fn fit(x: &Matrix, strategy: ImputeStrategy) -> Result<Self> {
        let d = x.first().map_or(0, Vec::len);
        if x.is_empty() || d == 0 || x.iter().any(|r| r.len() != d) {
            return Err(Error::Msg("SimpleImputer: need a non-empty rectangular matrix".into()));
        }
        let fill = (0..d)
            .map(|j| {
                let mut col: Vec<f64> = x.iter().map(|r| r[j]).filter(|v| !v.is_nan()).collect();
                if let ImputeStrategy::Constant(c) = strategy {
                    return Ok(c);
                }
                if col.is_empty() {
                    return Err(Error::Msg(format!("SimpleImputer: column {j} is entirely missing")));
                }
                col.sort_by(f64::total_cmp);
                Ok(match strategy {
                    ImputeStrategy::Mean => col.iter().sum::<f64>() / col.len() as f64,
                    ImputeStrategy::Median => if col.len() % 2 == 1 { col[col.len() / 2] } else { 0.5 * (col[col.len() / 2 - 1] + col[col.len() / 2]) },
                    ImputeStrategy::MostFrequent => {
                        let mut best = (col[0], 0);
                        let mut i = 0;
                        while i < col.len() {
                            let run = col[i..].iter().take_while(|&&v| v == col[i]).count();
                            if run > best.1 {
                                best = (col[i], run);
                            }
                            i += run;
                        }
                        best.0
                    }
                    ImputeStrategy::Constant(c) => c,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { fill })
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, &v)| if v.is_nan() { self.fill[j] } else { v }).collect()).collect()
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Centres on the median and scales by the interquartile range (robust to outliers).
#[derive(Debug, Clone)]
pub struct RobustScaler {
    pub median: Vec<f64>,
    pub iqr: Vec<f64>,
}

impl RobustScaler {
    pub fn fit(x: &Matrix) -> Result<Self> {
        let d = x.first().map_or(0, Vec::len);
        if x.is_empty() || d == 0 || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) {
            return Err(Error::Msg("RobustScaler: need a non-empty rectangular matrix of finite values".into()));
        }
        let (mut median, mut iqr) = (Vec::new(), Vec::new());
        for j in 0..d {
            let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            col.sort_by(f64::total_cmp);
            median.push(quantile(&col, 0.5));
            let r = quantile(&col, 0.75) - quantile(&col, 0.25);
            iqr.push(if r < 1e-12 { 1.0 } else { r });
        }
        Ok(Self { median, iqr })
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| (v - self.median[j]) / self.iqr[j]).collect()).collect()
    }
}

/// All monomials of the input columns up to `degree` (`include_bias` adds the constant column).
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    pub degree: usize,
    pub include_bias: bool,
}

impl PolynomialFeatures {
    fn exponents(d: usize, degree: usize, include_bias: bool) -> Vec<Vec<usize>> {
        fn rec(d: usize, left: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if start == d {
                out.push(cur.clone());
                return;
            }
            for e in 0..=left {
                cur[start] = e;
                rec(d, left - e, start + 1, cur, out);
            }
            cur[start] = 0;
        }
        let mut all = Vec::new();
        rec(d, degree, 0, &mut vec![0; d], &mut all);
        all.retain(|e| {
            let t: usize = e.iter().sum();
            t >= usize::from(!include_bias)
        });
        all.sort_by_key(|e| (e.iter().sum::<usize>(), std::cmp::Reverse(e.clone())));
        all
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        let d = x.first().map_or(0, Vec::len);
        let exps = Self::exponents(d, self.degree, self.include_bias);
        x.iter().map(|r| exps.iter().map(|e| e.iter().zip(r).map(|(&p, v)| v.powi(p as i32)).product()).collect()).collect()
    }

    pub fn n_output_features(&self, d: usize) -> usize {
        Self::exponents(d, self.degree, self.include_bias).len()
    }
}

/// Bins each column into `bins` intervals (`uniform` width or `quantile` edges) and returns bin indices.
#[derive(Debug, Clone)]
pub struct KBinsDiscretizer {
    pub edges: Vec<Vec<f64>>,
}

impl KBinsDiscretizer {
    pub fn fit(x: &Matrix, bins: usize, quantile_strategy: bool) -> Result<Self> {
        let d = x.first().map_or(0, Vec::len);
        if x.is_empty() || bins < 2 || d == 0 || x.iter().any(|r| r.len() != d) {
            return Err(Error::Msg("KBinsDiscretizer: need data and bins >= 2".into()));
        }
        let edges = (0..d)
            .map(|j| {
                let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                col.sort_by(f64::total_cmp);
                (1..bins)
                    .map(|b| if quantile_strategy { quantile(&col, b as f64 / bins as f64) } else { col[0] + (col[col.len() - 1] - col[0]) * b as f64 / bins as f64 })
                    .collect()
            })
            .collect();
        Ok(Self { edges })
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| self.edges[j].iter().filter(|&&e| v >= &e).count() as f64).collect()).collect()
    }
}

/// Scales every row to unit `p`-norm (`p` = 1, 2 or `f64::INFINITY`); all-zero rows are left unchanged.
pub fn normalize(x: &Matrix, p: f64) -> Result<Matrix> {
    if p != 1.0 && p != 2.0 && !p.is_infinite() {
        return Err(Error::Msg("normalize: p must be 1, 2 or infinity".into()));
    }
    Ok(x.iter()
        .map(|r| {
            let n = if p == 1.0 { r.iter().map(|v| v.abs()).sum() } else if p == 2.0 { r.iter().map(|v| v * v).sum::<f64>().sqrt() } else { r.iter().fold(0.0, |m: f64, v| m.max(v.abs())) };
            if n == 0.0 { r.clone() } else { r.iter().map(|v| v / n).collect() }
        })
        .collect())
}

/// Counts of each category, sorted by key — handy for checking class balance.
pub fn value_counts<S: AsRef<str>>(values: &[S]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    values.iter().for_each(|v| *m.entry(v.as_ref().to_string()).or_insert(0) += 1);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoders() {
        let le = LabelEncoder::fit(&["b", "a", "c", "a"]).unwrap();
        assert_eq!(le.classes, vec!["a", "b", "c"]);
        assert_eq!(le.transform(&["c", "a"]).unwrap(), vec![2, 0]);
        assert_eq!(le.inverse_transform(&[1]).unwrap(), vec!["b"]);
        assert!(le.transform(&["z"]).is_err() && le.inverse_transform(&[9]).is_err());
        let oh = OneHotEncoder::fit(&["red", "green", "red"], false).unwrap();
        assert_eq!(oh.transform(&["red", "green"]).unwrap(), vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        assert!(oh.transform(&["blue"]).is_err());
        let lenient = OneHotEncoder::fit(&["red", "green"], true).unwrap();
        assert_eq!(lenient.transform(&["blue"]).unwrap(), vec![vec![0.0, 0.0]]);
        assert_eq!(value_counts(&["a", "b", "a"])["a"], 2);
    }

    #[test]
    fn imputer_and_scalers() {
        let x = vec![vec![1.0, f64::NAN], vec![3.0, 4.0], vec![f64::NAN, 4.0], vec![5.0, 10.0]];
        let mean = SimpleImputer::fit(&x, ImputeStrategy::Mean).unwrap();
        assert_eq!(mean.fill, vec![3.0, 6.0]);
        assert_eq!(SimpleImputer::fit(&x, ImputeStrategy::Median).unwrap().fill, vec![3.0, 4.0]);
        assert_eq!(SimpleImputer::fit(&x, ImputeStrategy::MostFrequent).unwrap().fill[1], 4.0);
        assert_eq!(SimpleImputer::fit(&x, ImputeStrategy::Constant(-1.0)).unwrap().transform(&x)[0][1], -1.0);
        assert!(mean.transform(&x).iter().flatten().all(|v| !v.is_nan()));
        assert!(SimpleImputer::fit(&vec![vec![f64::NAN]], ImputeStrategy::Mean).is_err());
        // one huge outlier barely moves the robust scaling of the rest
        let data: Matrix = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1000.0]];
        let r = RobustScaler::fit(&data).unwrap();
        assert_eq!((r.median[0], r.iqr[0]), (3.0, 2.0));
        assert_eq!(r.transform(&data)[0][0], -1.0);
    }

    #[test]
    fn polynomial_discretizer_normalizer() {
        let p = PolynomialFeatures { degree: 2, include_bias: true };
        assert_eq!(p.n_output_features(2), 6);
        let out = p.transform(&vec![vec![2.0, 3.0]]);
        let mut got = out[0].clone();
        got.sort_by(f64::total_cmp);
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 6.0, 9.0]); // 1, a, b, a^2, ab, b^2
        assert_eq!(PolynomialFeatures { degree: 3, include_bias: false }.n_output_features(2), 9);
        let x: Matrix = (0..10).map(|i| vec![i as f64]).collect();
        let kb = KBinsDiscretizer::fit(&x, 3, false).unwrap().transform(&x);
        assert_eq!(kb.iter().map(|r| r[0] as usize).collect::<Vec<_>>(), vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
        let q = KBinsDiscretizer::fit(&x, 2, true).unwrap().transform(&x);
        assert_eq!(q.iter().filter(|r| r[0] == 0.0).count(), 5);
        assert!(KBinsDiscretizer::fit(&x, 1, false).is_err());
        let n = normalize(&vec![vec![3.0, 4.0], vec![0.0, 0.0]], 2.0).unwrap();
        assert_eq!(n, vec![vec![0.6, 0.8], vec![0.0, 0.0]]);
        assert_eq!(normalize(&vec![vec![1.0, -3.0]], 1.0).unwrap(), vec![vec![0.25, -0.75]]);
        assert!(normalize(&vec![vec![1.0]], 3.0).is_err());
    }
}
