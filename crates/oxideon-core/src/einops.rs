//! Einops-style `rearrange`, `reduce`, and `repeat` tensor pattern operations.
//!
//! Examples:
//! - `rearrange(&x, "b (h c) -> b h c", &[("h", 4)])`
//! - `rearrange(&x, "b c h w -> b h w c", &[])`
//! - `reduce(&x, "b c h w -> b c", ReduceKind::Mean, &[])`
//! - `repeat(&x, "h w -> b h w c", &[("b", 2), ("c", 3)])`
//! - `rearrange(&x, "b ... c -> b c ...", &[])` (one `...` stands for any number of axes) and `()` / `1` for unit axes

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone)]
enum Token {
    Single(String),
    Group(Vec<String>),
}

fn parse_side(side: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let bytes = side.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] == b'(' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b')' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(Error::Msg("einops: unclosed '(' in pattern".into()));
            }
            let inner = side[start..i].split_whitespace().map(str::to_string).collect::<Vec<_>>();
            // `()` is a unit axis (size 1)
            tokens.push(Token::Group(inner));
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'(' && bytes[i] != b')' {
                i += 1;
            }
            let name = &side[start..i];
            // a literal `1` is a unit axis too
            tokens.push(if name == "1" { Token::Group(Vec::new()) } else { Token::Single(name.to_string()) });
        }
    }
    Ok(tokens)
}

/// Replaces a top-level `...` with one generated axis name per dimension the other tokens of the left-hand side leave over
/// (`"b ... c -> ... b c"` on a rank-5 tensor becomes `"b _e0 _e1 _e2 c -> _e0 _e1 _e2 b c"`).
fn expand_ellipsis(pattern: &str, rank: usize) -> Result<String> {
    if !pattern.contains("...") {
        return Ok(pattern.to_string());
    }
    let (lhs, rhs) = pattern
        .split_once("->")
        .ok_or_else(|| Error::Msg(format!("einops: pattern '{pattern}' must contain '->'")))?;
    let tokens = parse_side(lhs)?;
    let ellipses = tokens.iter().filter(|t| matches!(t, Token::Single(n) if n == "...")).count();
    if ellipses != 1 || lhs.matches("...").count() != 1 {
        return Err(Error::Msg("einops: the left-hand side needs exactly one top-level '...'".into()));
    }
    let others = tokens.len() - 1;
    let k = rank.checked_sub(others).ok_or_else(|| Error::Msg(format!("einops: pattern needs at least {others} dimensions, tensor has {rank}")))?;
    let names = (0..k).map(|i| format!("_e{i}")).collect::<Vec<_>>().join(" ");
    Ok(format!("{}->{}", lhs.replace("...", &names), rhs.replace("...", &names)))
}

fn parse_pattern(pattern: &str) -> Result<(Vec<Token>, Vec<Token>)> {
    let (lhs, rhs) = pattern
        .split_once("->")
        .ok_or_else(|| Error::Msg(format!("einops: pattern '{pattern}' must contain '->'")))?;
    Ok((parse_side(lhs)?, parse_side(rhs)?))
}

fn resolve_lhs(
    lhs: &[Token],
    shape: &[usize],
    known: &mut HashMap<String, usize>,
) -> Result<Vec<(String, usize)>> {
    if lhs.len() != shape.len() {
        return Err(Error::Msg(format!(
            "einops: pattern lhs rank {} != tensor rank {}",
            lhs.len(),
            shape.len()
        )));
    }
    let mut flat_axes = Vec::new();
    for (tok, &dim) in lhs.iter().zip(shape.iter()) {
        match tok {
            Token::Single(name) => {
                if let Some(&prev) = known.get(name) {
                    if prev != dim {
                        return Err(Error::Msg(format!(
                            "einops: axis '{name}' size mismatch ({prev} vs {dim})"
                        )));
                    }
                } else {
                    known.insert(name.clone(), dim);
                }
                flat_axes.push((name.clone(), dim));
            }
            Token::Group(names) => {
                let mut prod = 1usize;
                let mut unknown = None;
                for name in names {
                    if let Some(&sz) = known.get(name) {
                        prod *= sz;
                    } else if unknown.is_none() {
                        unknown = Some(name.clone());
                    } else {
                        return Err(Error::Msg(format!(
                            "einops: cannot infer multiple unknown axes in group {names:?}"
                        )));
                    }
                }
                if let Some(u) = unknown {
                    if prod == 0 || dim % prod != 0 {
                        return Err(Error::Msg(format!(
                            "einops: dimension {dim} not divisible by {prod}"
                        )));
                    }
                    known.insert(u, dim / prod);
                } else if prod != dim {
                    return Err(Error::Msg(format!(
                        "einops: group {names:?} product {prod} != dim {dim}"
                    )));
                }
                for name in names {
                    flat_axes.push((name.clone(), known[name]));
                }
            }
        }
    }
    Ok(flat_axes)
}

/// Einops `rearrange`: splits, permutes, and merges axes according to `pattern`.
pub fn rearrange(t: &Tensor, pattern: &str, axes_lengths: &[(&str, usize)]) -> Result<Tensor> {
    let pattern = &expand_ellipsis(pattern, t.ndim())?;
    let (lhs, rhs) = parse_pattern(pattern)?;
    let mut known: HashMap<String, usize> =
        axes_lengths.iter().map(|&(k, v)| (k.to_string(), v)).collect();
    let flat_lhs = resolve_lhs(&lhs, t.shape(), &mut known)?;

    let split_shape: Vec<usize> = flat_lhs.iter().map(|(_, s)| *s).collect();
    let cur = if split_shape.as_slice() != t.shape() {
        t.reshape(&split_shape)?
    } else {
        t.clone()
    };

    let mut rhs_flat = Vec::new();
    let mut out_shape = Vec::new();
    for tok in &rhs {
        match tok {
            Token::Single(name) => {
                let sz = *known
                    .get(name)
                    .ok_or_else(|| Error::Msg(format!("einops: unknown axis '{name}' on rhs")))?;
                rhs_flat.push(name.clone());
                out_shape.push(sz);
            }
            Token::Group(names) => {
                let mut prod = 1usize;
                for name in names {
                    let sz = *known
                        .get(name)
                        .ok_or_else(|| Error::Msg(format!("einops: unknown axis '{name}' on rhs")))?;
                    rhs_flat.push(name.clone());
                    prod *= sz;
                }
                out_shape.push(prod);
            }
        }
    }

    if rhs_flat.len() != flat_lhs.len() {
        return Err(Error::Msg("einops::rearrange: axis count mismatch between lhs and rhs".into()));
    }
    let mut perm = Vec::with_capacity(rhs_flat.len());
    for name in &rhs_flat {
        let idx = flat_lhs
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| Error::Msg(format!("einops: rhs axis '{name}' not in lhs")))?;
        perm.push(idx);
    }
    let is_id = perm.iter().enumerate().all(|(i, &p)| i == p);
    let permuted = if is_id { cur } else { cur.permute(&perm)? };
    if permuted.shape() != out_shape.as_slice() {
        permuted.contiguous()?.reshape(&out_shape)
    } else {
        Ok(permuted)
    }
}

/// Einops `reduce`: reduces axes missing from `rhs`, then rearranges to `rhs`.
pub fn reduce(
    t: &Tensor,
    pattern: &str,
    op: ReduceKind,
    axes_lengths: &[(&str, usize)],
) -> Result<Tensor> {
    let pattern = &expand_ellipsis(pattern, t.ndim())?;
    let (lhs, rhs) = parse_pattern(pattern)?;
    let mut known: HashMap<String, usize> =
        axes_lengths.iter().map(|&(k, v)| (k.to_string(), v)).collect();
    let flat_lhs = resolve_lhs(&lhs, t.shape(), &mut known)?;
    let split_shape: Vec<usize> = flat_lhs.iter().map(|(_, s)| *s).collect();
    let mut cur = if split_shape.as_slice() != t.shape() {
        t.reshape(&split_shape)?
    } else {
        t.clone()
    };

    let mut rhs_names = Vec::new();
    for tok in &rhs {
        match tok {
            Token::Single(n) => rhs_names.push(n.clone()),
            Token::Group(ns) => rhs_names.extend(ns.iter().cloned()),
        }
    }

    let mut remaining_lhs = Vec::new();
    let mut reduce_axes = Vec::new();
    for (idx, (name, _)) in flat_lhs.iter().enumerate() {
        if rhs_names.contains(name) {
            remaining_lhs.push(name.clone());
        } else {
            reduce_axes.push(idx as isize);
        }
    }
    if !reduce_axes.is_empty() {
        cur = match op {
            ReduceKind::Sum => cur.sum_axes(&reduce_axes, false)?,
            ReduceKind::Mean => cur.mean_axes(&reduce_axes, false)?,
            ReduceKind::Max => {
                let mut tmp = cur;
                for &ax in reduce_axes.iter().rev() {
                    tmp = tmp.max_axis(ax, false)?;
                }
                tmp
            }
            ReduceKind::Min => {
                let mut tmp = cur;
                for &ax in reduce_axes.iter().rev() {
                    tmp = tmp.min_axis(ax, false)?;
                }
                tmp
            }
        };
    }
    let rhs_str = pattern.split_once("->").unwrap().1.trim();
    let new_pat = format!("{} -> {}", remaining_lhs.join(" "), rhs_str);
    rearrange(&cur, &new_pat, axes_lengths)
}

/// Einops `repeat`: expands and broadcasts new axes introduced on `rhs`.
pub fn repeat(t: &Tensor, pattern: &str, axes_lengths: &[(&str, usize)]) -> Result<Tensor> {
    let pattern = &expand_ellipsis(pattern, t.ndim())?;
    let (lhs, rhs) = parse_pattern(pattern)?;
    let mut known: HashMap<String, usize> =
        axes_lengths.iter().map(|&(k, v)| (k.to_string(), v)).collect();
    let flat_lhs = resolve_lhs(&lhs, t.shape(), &mut known)?;
    let split_shape: Vec<usize> = flat_lhs.iter().map(|(_, s)| *s).collect();
    let mut cur = if split_shape.as_slice() != t.shape() {
        t.reshape(&split_shape)?
    } else {
        t.clone()
    };

    let mut rhs_flat = Vec::new();
    for tok in &rhs {
        match tok {
            Token::Single(n) => rhs_flat.push(n.clone()),
            Token::Group(ns) => rhs_flat.extend(ns.iter().cloned()),
        }
    }

    // Insert size-1 axes for any new names on rhs, permute to rhs order, broadcast, then merge groups.
    let mut cur_names: Vec<String> = flat_lhs.iter().map(|(n, _)| n.clone()).collect();
    for name in &rhs_flat {
        if !cur_names.contains(name) {
            cur = cur.unsqueeze(cur_names.len() as isize)?;
            cur_names.push(name.clone());
        }
    }
    let mut perm = Vec::with_capacity(rhs_flat.len());
    let mut bcast_shape = Vec::with_capacity(rhs_flat.len());
    for name in &rhs_flat {
        let idx = cur_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::Msg(format!("einops::repeat: missing axis '{name}'")))?;
        perm.push(idx);
        let sz = *known
            .get(name)
            .ok_or_else(|| Error::Msg(format!("einops::repeat: unknown size for '{name}'")))?;
        bcast_shape.push(sz);
    }
    let permuted = cur.permute(&perm)?;
    let repeated = permuted.broadcast_to(&bcast_shape)?.contiguous()?;

    let mut out_shape = Vec::new();
    for tok in &rhs {
        match tok {
            Token::Single(n) => out_shape.push(known[n]),
            Token::Group(ns) => out_shape.push(ns.iter().map(|n| known[n]).product()),
        }
    }
    if repeated.shape() != out_shape.as_slice() {
        repeated.reshape(&out_shape)
    } else {
        Ok(repeated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn einops_rearrange_reduce_repeat() {
        let x = Tensor::from_slice(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]).unwrap();
        let r = rearrange(&x, "b c (h w) -> b h w c", &[("h", 2)]).unwrap();
        assert_eq!(r.shape(), &[2, 2, 2, 3]);

        let m = reduce(&r, "b h w c -> b c", ReduceKind::Sum, &[]).unwrap();
        assert_eq!(m.shape(), &[2, 3]);

        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let rep = repeat(&y, "h w -> b h w c", &[("b", 3), ("c", 2)]).unwrap();
        assert_eq!(rep.shape(), &[3, 2, 2, 2]);
    }
}
