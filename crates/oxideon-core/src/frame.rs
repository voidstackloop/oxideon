//! A small columnar [`DataFrame`] with numeric and string columns: construction from a CSV [`Table`], `select`,
//! `filter`, `sort_by`, `group_by` aggregation, inner / left `join`, `describe`, and conversion to a matrix.

use crate::dataio::Table;
use crate::error::{Error, Result};
use std::collections::BTreeMap;

/// One column of a [`DataFrame`].
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Num(Vec<f64>),
    Str(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Num(v) => v.len(),
            Column::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::Num(v) => Column::Num(idx.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::Str(idx.iter().map(|&i| v[i].clone()).collect()),
        }
    }

    fn key(&self, i: usize) -> String {
        match self {
            Column::Num(v) => format!("{}", v[i]),
            Column::Str(v) => v[i].clone(),
        }
    }
}

/// Aggregation applied by [`DataFrame::group_by`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Agg {
    Count,
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DataFrame {
    names: Vec<String>,
    cols: Vec<Column>,
}

impl DataFrame {
    pub fn new(columns: Vec<(&str, Column)>) -> Result<Self> {
        let n = columns.first().map_or(0, |c| c.1.len());
        if columns.iter().any(|c| c.1.len() != n) {
            return Err(Error::Msg("DataFrame: all columns must have the same length".into()));
        }
        let mut names: Vec<String> = columns.iter().map(|c| c.0.to_string()).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        if names.len() != total {
            return Err(Error::Msg("DataFrame: duplicate column names".into()));
        }
        Ok(Self { names: columns.iter().map(|c| c.0.to_string()).collect(), cols: columns.into_iter().map(|c| c.1).collect() })
    }

    /// Builds a frame from a headered CSV table: columns that parse fully as numbers (empty = NaN) become numeric.
    pub fn from_table(t: &Table) -> Result<Self> {
        let header = t.header.as_ref().ok_or_else(|| Error::Msg("DataFrame::from_table: the table has no header".into()))?;
        let cols = (0..header.len())
            .map(|c| match t.numeric_column(c) {
                Ok(v) => Column::Num(v),
                Err(_) => Column::Str(t.rows.iter().map(|r| r[c].clone()).collect()),
            })
            .collect::<Vec<_>>();
        Self::new(header.iter().map(String::as_str).zip(cols).collect())
    }

    pub fn nrows(&self) -> usize {
        self.cols.first().map_or(0, Column::len)
    }

    pub fn columns(&self) -> &[String] {
        &self.names
    }

    pub fn col(&self, name: &str) -> Result<&Column> {
        self.names.iter().position(|n| n == name).map(|i| &self.cols[i]).ok_or_else(|| Error::Msg(format!("DataFrame: no column {name:?}")))
    }

    pub fn num(&self, name: &str) -> Result<&[f64]> {
        match self.col(name)? {
            Column::Num(v) => Ok(v),
            Column::Str(_) => Err(Error::Msg(format!("DataFrame: column {name:?} is not numeric"))),
        }
    }

    pub fn select(&self, names: &[&str]) -> Result<Self> {
        Self::new(names.iter().map(|n| Ok((*n, self.col(n)?.clone()))).collect::<Result<_>>()?)
    }

    fn take_rows(&self, idx: &[usize]) -> Self {
        Self { names: self.names.clone(), cols: self.cols.iter().map(|c| c.take(idx)).collect() }
    }

    /// Rows for which `pred(row_index)` holds.
    pub fn filter(&self, pred: impl Fn(usize) -> bool) -> Self {
        let idx: Vec<usize> = (0..self.nrows()).filter(|&i| pred(i)).collect();
        self.take_rows(&idx)
    }

    /// Stable sort by one column (numbers numerically, strings lexicographically).
    pub fn sort_by(&self, name: &str, descending: bool) -> Result<Self> {
        let col = self.col(name)?;
        let mut idx: Vec<usize> = (0..self.nrows()).collect();
        idx.sort_by(|&a, &b| {
            let o = match col {
                Column::Num(v) => v[a].total_cmp(&v[b]),
                Column::Str(v) => v[a].cmp(&v[b]),
            };
            if descending { o.reverse() } else { o }
        });
        Ok(self.take_rows(&idx))
    }

    /// Groups by `key` and aggregates the numeric `value` column; groups are returned sorted by key.
    pub fn group_by(&self, key: &str, value: &str, agg: Agg) -> Result<Self> {
        let (kc, vals) = (self.col(key)?, self.num(value)?);
        let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for i in 0..self.nrows() {
            groups.entry(kc.key(i)).or_default().push(vals[i]);
        }
        let (keys, out): (Vec<String>, Vec<f64>) = groups
            .into_iter()
            .map(|(k, v)| {
                let r = match agg {
                    Agg::Count => v.len() as f64,
                    Agg::Sum => v.iter().sum(),
                    Agg::Mean => v.iter().sum::<f64>() / v.len() as f64,
                    Agg::Min => v.iter().cloned().fold(f64::INFINITY, f64::min),
                    Agg::Max => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                };
                (k, r)
            })
            .unzip();
        let key_col = match kc {
            Column::Num(_) => Column::Num(keys.iter().map(|k| k.parse().unwrap_or(f64::NAN)).collect()),
            Column::Str(_) => Column::Str(keys),
        };
        Self::new(vec![(key, key_col), (value, Column::Num(out))])
    }

    /// Joins `other` on equal values of `on` (present in both frames). `left = true` keeps unmatched left rows,
    /// filling the right columns with `NaN` / empty strings. Duplicate right keys multiply rows.
    pub fn join(&self, other: &Self, on: &str, left: bool) -> Result<Self> {
        let (lk, rk) = (self.col(on)?, other.col(on)?);
        let mut index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        (0..other.nrows()).for_each(|j| index.entry(rk.key(j)).or_default().push(j));
        let (mut li, mut ri): (Vec<usize>, Vec<Option<usize>>) = (Vec::new(), Vec::new());
        for i in 0..self.nrows() {
            match index.get(&lk.key(i)) {
                Some(js) => js.iter().for_each(|&j| { li.push(i); ri.push(Some(j)); }),
                None if left => { li.push(i); ri.push(None); }
                None => {}
            }
        }
        let mut cols: Vec<(String, Column)> = self.names.iter().zip(&self.cols).map(|(n, c)| (n.clone(), c.take(&li))).collect();
        for (n, c) in other.names.iter().zip(&other.cols).filter(|(n, _)| n.as_str() != on) {
            let name = if self.names.contains(n) { format!("{n}_right") } else { n.clone() };
            let col = match c {
                Column::Num(v) => Column::Num(ri.iter().map(|j| j.map_or(f64::NAN, |j| v[j])).collect()),
                Column::Str(v) => Column::Str(ri.iter().map(|j| j.map_or(String::new(), |j| v[j].clone())).collect()),
            };
            cols.push((name, col));
        }
        Self::new(cols.iter().map(|(n, c)| (n.as_str(), c.clone())).collect())
    }

    /// Per numeric column: count, mean, std, min, max (NaNs ignored). Returns `(column, [count, mean, std, min, max])`.
    pub fn describe(&self) -> Vec<(String, [f64; 5])> {
        self.names
            .iter()
            .zip(&self.cols)
            .filter_map(|(n, c)| match c {
                Column::Num(v) => {
                    let ok: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
                    let cnt = ok.len() as f64;
                    let mean = ok.iter().sum::<f64>() / cnt;
                    let std = if cnt > 1.0 { (ok.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (cnt - 1.0)).sqrt() } else { f64::NAN };
                    Some((n.clone(), [cnt, mean, std, ok.iter().cloned().fold(f64::INFINITY, f64::min), ok.iter().cloned().fold(f64::NEG_INFINITY, f64::max)]))
                }
                Column::Str(_) => None,
            })
            .collect()
    }

    /// The listed numeric columns as a row-major matrix.
    pub fn to_matrix(&self, names: &[&str]) -> Result<Vec<Vec<f64>>> {
        let cols: Vec<&[f64]> = names.iter().map(|n| self.num(n)).collect::<Result<_>>()?;
        Ok((0..self.nrows()).map(|i| cols.iter().map(|c| c[i]).collect()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataio::parse_csv;

    fn sales() -> DataFrame {
        let t = parse_csv("region,product,units,price\nwest,a,10,2.5\neast,a,4,2.5\nwest,b,7,4.0\neast,b,,4.0\nwest,a,3,2.5\n", ',', true).unwrap();
        DataFrame::from_table(&t).unwrap()
    }

    #[test]
    fn construction_selection_and_filtering() {
        let df = sales();
        assert_eq!((df.nrows(), df.columns().len()), (5, 4));
        assert!(matches!(df.col("region").unwrap(), Column::Str(_)) && matches!(df.col("units").unwrap(), Column::Num(_)));
        assert!(df.num("units").unwrap()[3].is_nan(), "empty cells become NaN");
        assert!(df.num("region").is_err() && df.col("nope").is_err());
        let sel = df.select(&["product", "price"]).unwrap();
        assert_eq!(sel.columns(), &["product", "price"]);
        let west = df.filter(|i| matches!(df.col("region").unwrap(), Column::Str(v) if v[i] == "west"));
        assert_eq!(west.nrows(), 3);
        let sorted = df.sort_by("price", true).unwrap();
        assert_eq!(sorted.num("price").unwrap(), &[4.0, 4.0, 2.5, 2.5, 2.5]);
        assert!(DataFrame::new(vec![("a", Column::Num(vec![1.0])), ("b", Column::Num(vec![]))]).is_err());
        assert!(DataFrame::new(vec![("a", Column::Num(vec![1.0])), ("a", Column::Num(vec![2.0]))]).is_err());
    }

    #[test]
    fn grouping_describe_and_matrix() {
        let df = sales();
        let by_region = df.group_by("region", "price", Agg::Sum).unwrap();
        assert_eq!(by_region.col("region").unwrap(), &Column::Str(vec!["east".into(), "west".into()]));
        assert_eq!(by_region.num("price").unwrap(), &[6.5, 9.0]);
        let counts = df.group_by("product", "price", Agg::Count).unwrap();
        assert_eq!(counts.num("price").unwrap(), &[3.0, 2.0]);
        let mean = df.group_by("region", "price", Agg::Mean).unwrap();
        assert!((mean.num("price").unwrap()[1] - 3.0).abs() < 1e-12);
        let d = df.describe();
        let units = d.iter().find(|x| x.0 == "units").unwrap().1;
        assert_eq!(units[0], 4.0); // NaN ignored
        assert!((units[1] - 6.0).abs() < 1e-12 && units[3] == 3.0 && units[4] == 10.0);
        assert_eq!(df.to_matrix(&["price"]).unwrap().len(), 5);
        assert!(df.to_matrix(&["region"]).is_err());
    }

    #[test]
    fn joins() {
        let df = sales();
        let info = DataFrame::new(vec![("region", Column::Str(vec!["west".into(), "north".into()])), ("manager", Column::Str(vec!["ann".into(), "bob".into()])), ("price", Column::Num(vec![1.0, 2.0]))]).unwrap();
        let inner = df.join(&info, "region", false).unwrap();
        assert_eq!(inner.nrows(), 3, "only west rows match");
        assert_eq!(inner.col("manager").unwrap(), &Column::Str(vec!["ann".into(); 3]));
        assert!(inner.columns().contains(&"price_right".to_string()), "clashing names get a suffix");
        let outer = df.join(&info, "region", true).unwrap();
        assert_eq!(outer.nrows(), 5);
        let managers = match outer.col("manager").unwrap() { Column::Str(v) => v.clone(), _ => unreachable!() };
        assert_eq!(managers.iter().filter(|m| m.is_empty()).count(), 2, "east rows are unmatched");
        assert!(df.join(&info, "nope", false).is_err());
    }
}
