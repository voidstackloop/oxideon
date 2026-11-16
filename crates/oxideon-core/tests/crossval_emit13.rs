//! Cross-validation batch 13: string metrics and BM25 (compared by tools/crossval/check_batch13.py).
use oxideon_core::text_ir::*;

const PAIRS: [(&str, &str); 15] = [
    ("abcxxxxxxx", "abcyyyyyyy"),
    ("martha", "marhta"),
    ("dixon", "dicksonx"),
    ("jellyfish", "smellyfish"),
    ("abc", "xyz"),
    ("kitten", "sitting"),
    ("ca", "abc"),
    ("hello", "hello"),
    ("crate", "trace"),
    ("a", "b"),
    ("prefixmatch", "prefixmiss"),
    ("dwayne", "duane"),
    ("shackleford", "shackelford"),
    ("abcdef", "badcfe"),
    ("", "abc"),
];

#[test]
#[ignore]
fn emit() {
    for (i, (a, b)) in PAIRS.iter().enumerate() {
        println!("jw_{i} {:.12}", jaro_winkler(a, b));
        println!("dl_{i} {}", damerau_levenshtein(a, b));
        let (ca, cb): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
        println!("lcs_{i} {}", lcs_len(&ca, &cb));
    }
    let docs = [
        "the quick brown fox jumps over the lazy dog",
        "never jump over the lazy dog quickly",
        "a quick brown dog outpaces a quick fox",
        "lorem ipsum dolor sit amet",
        "the the the the fox",
    ];
    for (k1, b) in [(1.5, 0.75), (1.2, 0.3), (0.0, 0.5), (2.0, 1.0)] {
        let bm = Bm25::new(&docs, k1, b).unwrap();
        for (qi, q) in ["quick fox", "lazy dog the", "ipsum", "the fox fox"].iter().enumerate() {
            for i in 0..docs.len() {
                println!("bm25_{k1}_{b}_{qi}_{i} {:.12}", bm.score(q, i));
            }
        }
    }
}
