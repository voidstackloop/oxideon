//! Classical text retrieval and string similarity: [`tokenize`], [`TfidfVectorizer`], an Okapi [`Bm25`] index with
//! ranked search, [`cosine_sparse`], and string metrics ([`jaro_winkler`], [`soundex`], [`lcs_len`],
//! [`damerau_levenshtein`]).

use crate::error::{Error, Result};
use std::collections::{BTreeMap, HashMap};

/// Lower-cases and splits on non-alphanumeric characters.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(str::to_string).collect()
}

/// Sparse vector: sorted `(feature index, value)` pairs.
pub type SparseVec = Vec<(usize, f64)>;

/// Cosine similarity of two sparse vectors with sorted indices.
pub fn cosine_sparse(a: &SparseVec, b: &SparseVec) -> f64 {
    let (mut i, mut j, mut dot) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    let norm = |v: &SparseVec| v.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt();
    let d = norm(a) * norm(b);
    if d == 0.0 { 0.0 } else { dot / d }
}

/// TF-IDF with smoothed idf `ln((1 + n) / (1 + df)) + 1` and L2-normalized rows (scikit-learn defaults).
#[derive(Debug, Clone)]
pub struct TfidfVectorizer {
    pub vocab: BTreeMap<String, usize>,
    pub idf: Vec<f64>,
}

impl TfidfVectorizer {
    pub fn fit<S: AsRef<str>>(docs: &[S]) -> Result<Self> {
        if docs.is_empty() {
            return Err(Error::Msg("TfidfVectorizer: no documents".into()));
        }
        let mut df: BTreeMap<String, usize> = BTreeMap::new();
        for d in docs {
            let mut seen: Vec<String> = tokenize(d.as_ref());
            seen.sort();
            seen.dedup();
            seen.into_iter().for_each(|t| *df.entry(t).or_insert(0) += 1);
        }
        let n = docs.len() as f64;
        let vocab: BTreeMap<String, usize> = df.keys().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        let idf = df.values().map(|&d| ((1.0 + n) / (1.0 + d as f64)).ln() + 1.0).collect();
        Ok(Self { vocab, idf })
    }

    /// Sparse TF-IDF vector of one document (unknown words are ignored).
    pub fn transform_one(&self, doc: &str) -> SparseVec {
        let mut tf: BTreeMap<usize, f64> = BTreeMap::new();
        for t in tokenize(doc) {
            if let Some(&i) = self.vocab.get(&t) {
                *tf.entry(i).or_insert(0.0) += 1.0;
            }
        }
        let mut v: SparseVec = tf.into_iter().map(|(i, c)| (i, c * self.idf[i])).collect();
        let norm = v.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| x.1 /= norm);
        }
        v
    }

    pub fn transform<S: AsRef<str>>(&self, docs: &[S]) -> Vec<SparseVec> {
        docs.iter().map(|d| self.transform_one(d.as_ref())).collect()
    }
}

/// Okapi BM25 ranking over a fixed corpus.
#[derive(Debug, Clone)]
pub struct Bm25 {
    docs: Vec<HashMap<String, f64>>,
    lens: Vec<f64>,
    df: HashMap<String, f64>,
    avg_len: f64,
    pub k1: f64,
    pub b: f64,
}

impl Bm25 {
    pub fn new<S: AsRef<str>>(docs: &[S], k1: f64, b: f64) -> Result<Self> {
        if docs.is_empty() || k1 < 0.0 || !(0.0..=1.0).contains(&b) {
            return Err(Error::Msg("Bm25: need documents, k1 >= 0 and 0 <= b <= 1".into()));
        }
        let mut df: HashMap<String, f64> = HashMap::new();
        let mut tfs = Vec::new();
        let mut lens = Vec::new();
        for d in docs {
            let toks = tokenize(d.as_ref());
            lens.push(toks.len() as f64);
            let mut tf: HashMap<String, f64> = HashMap::new();
            toks.into_iter().for_each(|t| *tf.entry(t).or_insert(0.0) += 1.0);
            tf.keys().for_each(|t| *df.entry(t.clone()).or_insert(0.0) += 1.0);
            tfs.push(tf);
        }
        let avg_len = lens.iter().sum::<f64>() / lens.len() as f64;
        Ok(Self { docs: tfs, lens, df, avg_len, k1, b })
    }

    /// BM25 score of document `i` for the query.
    pub fn score(&self, query: &str, i: usize) -> f64 {
        let n = self.docs.len() as f64;
        tokenize(query)
            .iter()
            .map(|t| {
                let f = self.docs[i].get(t).copied().unwrap_or(0.0);
                if f == 0.0 {
                    return 0.0;
                }
                let df = self.df[t];
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                idf * f * (self.k1 + 1.0) / (f + self.k1 * (1.0 - self.b + self.b * self.lens[i] / self.avg_len.max(1e-9)))
            })
            .sum()
    }

    /// The `k` best documents as `(index, score)`, best first (zero-score documents are dropped).
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let mut s: Vec<(usize, f64)> = (0..self.docs.len()).map(|i| (i, self.score(query, i))).filter(|x| x.1 > 0.0).collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        s.truncate(k);
        s
    }
}

/// Jaro-Winkler similarity in `[0, 1]` (prefix scale 0.1, prefix capped at 4).
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let window = (a.len().max(b.len()) / 2).saturating_sub(1);
    let (mut ma, mut mb) = (vec![false; a.len()], vec![false; b.len()]);
    let mut matches = 0.0;
    for i in 0..a.len() {
        for j in i.saturating_sub(window)..(i + window + 1).min(b.len()) {
            if !mb[j] && a[i] == b[j] {
                ma[i] = true;
                mb[j] = true;
                matches += 1.0;
                break;
            }
        }
    }
    if matches == 0.0 {
        return 0.0;
    }
    let sa: Vec<char> = a.iter().zip(&ma).filter(|(_, m)| **m).map(|(c, _)| *c).collect();
    let sb: Vec<char> = b.iter().zip(&mb).filter(|(_, m)| **m).map(|(c, _)| *c).collect();
    let transpositions = sa.iter().zip(&sb).filter(|(x, y)| x != y).count() as f64 / 2.0;
    let jaro = (matches / a.len() as f64 + matches / b.len() as f64 + (matches - transpositions) / matches) / 3.0;
    let prefix = a.iter().zip(&b).take(4).take_while(|(x, y)| x == y).count() as f64;
    // Winkler's boost threshold: only reward a shared prefix when the strings are already similar.
    if jaro > 0.7 { jaro + prefix * 0.1 * (1.0 - jaro) } else { jaro }
}

/// American Soundex code (letter + 3 digits).
pub fn soundex(word: &str) -> String {
    let letters: Vec<char> = word.chars().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase()).collect();
    let Some(&first) = letters.first() else { return String::new() };
    let code = |c: char| match c {
        'B' | 'F' | 'P' | 'V' => '1',
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
        'D' | 'T' => '3',
        'L' => '4',
        'M' | 'N' => '5',
        'R' => '6',
        _ => '0',
    };
    let mut out = String::from(first);
    let mut last = code(first);
    for &c in &letters[1..] {
        let d = code(c);
        if d != '0' && d != last {
            out.push(d);
        }
        if c != 'H' && c != 'W' {
            last = d;
        }
        if out.len() == 4 {
            break;
        }
    }
    while out.len() < 4 {
        out.push('0');
    }
    out
}

/// Length of the longest common subsequence.
pub fn lcs_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev = vec![0; b.len() + 1];
    for x in a {
        let mut cur = vec![0; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { prev[j + 1].max(cur[j]) };
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Damerau-Levenshtein distance (optimal string alignment: adjacent transpositions cost 1).
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    (0..=a.len()).for_each(|i| d[i][0] = i);
    (0..=b.len()).for_each(|j| d[0][j] = j);
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    const DOCS: [&str; 4] = ["the cat sat on the mat", "the dog chased the cat", "dogs and cats are pets", "quantum field theory lectures"];

    #[test]
    fn tfidf_similarity() {
        assert_eq!(tokenize("Hello, World! It's 2024."), vec!["hello", "world", "it", "s", "2024"]);
        let tf = TfidfVectorizer::fit(&DOCS).unwrap();
        let v = tf.transform(&DOCS);
        // rows are unit length
        v.iter().for_each(|r| close(r.iter().map(|x| x.1 * x.1).sum::<f64>(), 1.0, 1e-12));
        // docs sharing "cat" are more similar than an unrelated one
        assert!(cosine_sparse(&v[0], &v[1]) > cosine_sparse(&v[0], &v[3]));
        close(cosine_sparse(&v[0], &v[0]), 1.0, 1e-12);
        close(cosine_sparse(&v[0], &v[3]), 0.0, 1e-12);
        // rare words weigh more than common ones: "the" (in 2 docs) < "mat" (in 1)
        let (the, mat) = (tf.vocab["the"], tf.vocab["mat"]);
        assert!(tf.idf[mat] > tf.idf[the]);
        assert!(tf.transform_one("zzz unknown").is_empty());
        assert!(TfidfVectorizer::fit::<&str>(&[]).is_err());
    }

    #[test]
    fn bm25_ranks_relevant_documents() {
        let bm = Bm25::new(&DOCS, 1.5, 0.75).unwrap();
        let hits = bm.search("cat mat", 3);
        assert_eq!(hits[0].0, 0, "{hits:?}");
        assert!(hits.iter().any(|h| h.0 == 1) && hits.iter().all(|h| h.0 != 3));
        assert!(bm.search("blockchain", 3).is_empty());
        assert_eq!(bm.search("quantum", 1)[0].0, 3);
        // term-frequency saturation: scores grow sublinearly in repeated terms
        let rep = Bm25::new(&["a b", "a a a a a a a a b", "c d"], 1.2, 0.0).unwrap();
        assert!(rep.score("a", 1) < 8.0 * rep.score("a", 0));
        assert!(Bm25::new(&DOCS, 1.5, 2.0).is_err());
    }

    #[test]
    fn string_similarity() {
        close(jaro_winkler("MARTHA", "MARHTA"), 0.961_111, 1e-5); // classic reference value
        close(jaro_winkler("DWAYNE", "DUANE"), 0.84, 1e-3);
        close(jaro_winkler("same", "same"), 1.0, 1e-12);
        close(jaro_winkler("abc", "xyz"), 0.0, 1e-12);
        // no prefix boost below the 0.7 similarity threshold (matches rapidfuzz / jellyfish)
        close(jaro_winkler("abcxxxxxxx", "abcyyyyyyy"), 0.533_333_333, 1e-8);
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
        assert_eq!(soundex("Tymczak"), "T522");
        assert_eq!(soundex("Pfister"), "P236");
        assert_eq!(soundex(""), "");
        assert_eq!(lcs_len(&"AGGTAB".chars().collect::<Vec<_>>(), &"GXTXAYB".chars().collect::<Vec<_>>()), 4);
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);
        assert_eq!(damerau_levenshtein("kitten", "sitting"), 3);
        assert_eq!(damerau_levenshtein("", "abc"), 3);
    }
}
