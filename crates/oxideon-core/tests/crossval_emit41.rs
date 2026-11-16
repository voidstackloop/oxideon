//! Cross-validation batch 41: Unicode general categories and NFC / NFD / NFKC / NFKD over every code point and over
//! pseudo-random sequences of marks and bases vs Python's unicodedata (tools/crossval/check_batch41.py).
use oxideon_core::tokenizer::unicode::*;

type Form = fn(&str) -> String;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
    *h ^= 0xff;
    *h = h.wrapping_mul(0x100000001b3);
}

#[test]
#[ignore]
fn emit_unicode() {
    let forms: [(&str, Form); 4] = [("nfc", nfc), ("nfd", nfd), ("nfkc", nfkc), ("nfkd", nfkd)];
    let mut hc = 0xcbf29ce484222325u64;
    let mut hs = [0xcbf29ce484222325u64; 4];
    let mut hq = [0xcbf29ce484222325u64; 4];
    let mut hcc = 0xcbf29ce484222325u64;
    for cp in 0..0x110000u32 {
        let Some(c) = char::from_u32(cp) else { continue };
        fnv(&mut hc, category(c).as_bytes());
        fnv(&mut hcc, &[combining_class(c)]);
        let s = c.to_string();
        for (k, (_, f)) in forms.iter().enumerate() {
            fnv(&mut hs[k], f(&s).as_bytes());
        }
    }
    // pool: bases, composites, many combining marks of different classes, Hangul jamo, compat characters
    let pool: Vec<char> = "aeiouAEIOUnNcCsSzZ\u{0301}\u{0300}\u{0308}\u{0323}\u{0327}\u{0328}\u{0331}\u{0345}\u{05B0}\u{064B}\u{0E38}\u{3099}\u{1100}\u{1161}\u{11A8}\u{AC00}\u{AC01}\u{1112}\u{11AB}ﬁ①㎏ǆǅÅÅẛ̣Ωö\u{1EA1}\u{0F71}\u{0F72}\u{0F74}\u{0F80}\u{1E0B}\u{0307}\u{0344}\u{0958}\u{09CB}\u{09BE}\u{0B3E}\u{0B47}"
        .chars()
        .collect();
    let mut x = 88172645463325252u64;
    for _ in 0..40000 {
        let mut s = String::new();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let mut y = x;
        for _ in 0..(1 + y % 7) {
            y = y.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s.push(pool[((y >> 33) as usize) % pool.len()]);
        }
        for (k, (_, f)) in forms.iter().enumerate() {
            fnv(&mut hq[k], f(&s).as_bytes());
        }
    }
    println!("cat {hc}");
    println!("ccc {hcc}");
    for (k, (n, _)) in forms.iter().enumerate() {
        println!("single_{n} {}", hs[k]);
        println!("seq_{n} {}", hq[k]);
    }
}
