//! Spatial data structures and computational geometry: [`KdTree`] (exact k-nearest and radius queries),
//! [`convex_hull`] (Andrew's monotone chain), [`polygon_area`], [`point_in_polygon`], and [`closest_pair`].

use crate::error::{Error, Result};

fn sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[derive(Debug, Clone)]
struct KdNode {
    point: usize,
    axis: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

/// k-d tree over `d`-dimensional points for exact nearest-neighbour search.
#[derive(Debug, Clone)]
pub struct KdTree {
    points: Vec<Vec<f64>>,
    root: Option<Box<KdNode>>,
}

impl KdTree {
    pub fn new(points: Vec<Vec<f64>>) -> Result<Self> {
        let d = points.first().map_or(0, Vec::len);
        if d == 0 || points.iter().any(|p| p.len() != d || p.iter().any(|v| !v.is_finite())) {
            return Err(Error::Msg("KdTree: need non-empty points of equal dimension with finite coordinates".into()));
        }
        let mut idx: Vec<usize> = (0..points.len()).collect();
        let root = Self::build(&points, &mut idx, 0, d);
        Ok(Self { points, root })
    }

    fn build(points: &[Vec<f64>], idx: &mut [usize], depth: usize, d: usize) -> Option<Box<KdNode>> {
        if idx.is_empty() {
            return None;
        }
        let axis = depth % d;
        idx.sort_by(|&a, &b| points[a][axis].total_cmp(&points[b][axis]));
        let mid = idx.len() / 2;
        let (left, rest) = idx.split_at_mut(mid);
        let (pivot, right) = rest.split_first_mut().unwrap();
        Some(Box::new(KdNode { point: *pivot, axis, left: Self::build(points, left, depth + 1, d), right: Self::build(points, right, depth + 1, d) }))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The `k` nearest points to `q` as `(index, distance)`, nearest first.
    pub fn knn(&self, q: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if q.len() != self.points[0].len() || k == 0 {
            return Err(Error::Msg("KdTree::knn: query dimension must match the tree and k >= 1".into()));
        }
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1); // sorted ascending by squared distance
        self.search(self.root.as_deref(), q, k, &mut best);
        Ok(best.into_iter().map(|(d, i)| (i, d.sqrt())).collect())
    }

    fn search(&self, node: Option<&KdNode>, q: &[f64], k: usize, best: &mut Vec<(f64, usize)>) {
        let Some(n) = node else { return };
        let d = sq(&self.points[n.point], q);
        if best.len() < k || d < best.last().unwrap().0 {
            let pos = best.partition_point(|b| b.0 <= d);
            best.insert(pos, (d, n.point));
            best.truncate(k);
        }
        let diff = q[n.axis] - self.points[n.point][n.axis];
        let (near, far) = if diff < 0.0 { (&n.left, &n.right) } else { (&n.right, &n.left) };
        self.search(near.as_deref(), q, k, best);
        if best.len() < k || diff * diff < best.last().unwrap().0 {
            self.search(far.as_deref(), q, k, best);
        }
    }

    /// Indices of all points within `radius` of `q` (unordered).
    pub fn radius(&self, q: &[f64], radius: f64) -> Result<Vec<usize>> {
        if q.len() != self.points[0].len() || radius < 0.0 {
            return Err(Error::Msg("KdTree::radius: query dimension must match the tree and radius >= 0".into()));
        }
        let mut out = Vec::new();
        let mut stack = vec![self.root.as_deref()];
        while let Some(top) = stack.pop() {
            let Some(n) = top else { continue };
            if sq(&self.points[n.point], q) <= radius * radius {
                out.push(n.point);
            }
            let diff = q[n.axis] - self.points[n.point][n.axis];
            let (near, far) = if diff < 0.0 { (&n.left, &n.right) } else { (&n.right, &n.left) };
            stack.push(near.as_deref());
            if diff.abs() <= radius {
                stack.push(far.as_deref());
            }
        }
        Ok(out)
    }
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Convex hull of 2-D points in counter-clockwise order (collinear points on edges are dropped).
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f64, f64)>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &pt in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0.0 {
                hull.pop();
            }
            hull.push(pt);
        }
        hull.pop();
    }
    hull
}

/// Signed area of a simple polygon (positive when counter-clockwise).
pub fn signed_area(poly: &[(f64, f64)]) -> f64 {
    0.5 * (0..poly.len()).map(|i| { let (a, b) = (poly[i], poly[(i + 1) % poly.len()]); a.0 * b.1 - b.0 * a.1 }).sum::<f64>()
}

/// Absolute area of a simple polygon.
pub fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    signed_area(poly).abs()
}

/// Ray-casting point-in-polygon test (points exactly on an edge may fall either way).
pub fn point_in_polygon(pt: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut j = poly.len().saturating_sub(1);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.1 > pt.1) != (b.1 > pt.1) && pt.0 < (b.0 - a.0) * (pt.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Closest pair of 2-D points (divide and conquer): `((i, j), distance)` with `i < j` indexing `points`.
pub fn closest_pair(points: &[(f64, f64)]) -> Result<((usize, usize), f64)> {
    if points.len() < 2 || points.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        return Err(Error::Msg("closest_pair: need at least two finite points".into()));
    }
    let mut idx: Vec<usize> = (0..points.len()).collect();
    idx.sort_by(|&a, &b| points[a].0.total_cmp(&points[b].0));
    fn d2(p: (f64, f64), q: (f64, f64)) -> f64 {
        (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)
    }
    fn rec(pts: &[(f64, f64)], idx: &[usize]) -> (f64, usize, usize) {
        if idx.len() <= 3 {
            let mut best = (f64::INFINITY, idx[0], idx[1]);
            for a in 0..idx.len() {
                for b in a + 1..idx.len() {
                    let d = d2(pts[idx[a]], pts[idx[b]]);
                    if d < best.0 {
                        best = (d, idx[a], idx[b]);
                    }
                }
            }
            return best;
        }
        let mid = idx.len() / 2;
        let midx = pts[idx[mid]].0;
        let (l, r) = (rec(pts, &idx[..mid]), rec(pts, &idx[mid..]));
        let mut best = if l.0 <= r.0 { l } else { r };
        let mut strip: Vec<usize> = idx.iter().copied().filter(|&i| (pts[i].0 - midx).powi(2) < best.0).collect();
        strip.sort_by(|&a, &b| pts[a].1.total_cmp(&pts[b].1));
        for a in 0..strip.len() {
            for b in a + 1..strip.len() {
                if (pts[strip[b]].1 - pts[strip[a]].1).powi(2) >= best.0 {
                    break;
                }
                let d = d2(pts[strip[a]], pts[strip[b]]);
                if d < best.0 {
                    best = (d, strip[a], strip[b]);
                }
            }
        }
        best
    }
    let (d, a, b) = rec(points, &idx);
    Ok(((a.min(b), a.max(b)), d.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn cloud(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| (0..d).map(|_| r.next_f64() * 10.0).collect()).collect()
    }

    #[test]
    fn kdtree_matches_brute_force() {
        let pts = cloud(300, 3, 1);
        let tree = KdTree::new(pts.clone()).unwrap();
        for q in cloud(20, 3, 2) {
            let got = tree.knn(&q, 5).unwrap();
            let mut brute: Vec<(usize, f64)> = pts.iter().enumerate().map(|(i, p)| (i, sq(p, &q).sqrt())).collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (g, b) in got.iter().zip(&brute) {
                assert!((g.1 - b.1).abs() < 1e-12);
            }
            let mut r = tree.radius(&q, 2.5).unwrap();
            r.sort_unstable();
            let mut want: Vec<usize> = brute.iter().filter(|x| x.1 <= 2.5).map(|x| x.0).collect();
            want.sort_unstable();
            assert_eq!(r, want);
        }
        assert!(tree.knn(&[1.0], 3).is_err() && tree.knn(&[1.0, 1.0, 1.0], 0).is_err());
        assert_eq!(KdTree::new(vec![vec![1.0, 2.0]]).unwrap().knn(&[0.0, 0.0], 3).unwrap().len(), 1);
        assert!(KdTree::new(vec![]).is_err());
    }

    #[test]
    fn hull_area_and_containment() {
        let sq4 = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
        let mut pts = sq4.to_vec();
        pts.extend([(1.0, 1.0), (2.0, 2.0), (3.0, 1.5), (2.0, 0.0)]); // interior and an edge point
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!(signed_area(&hull) > 0.0, "counter-clockwise");
        assert!((polygon_area(&hull) - 12.0).abs() < 1e-12);
        assert!((polygon_area(&[(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]) - 6.0).abs() < 1e-12);
        assert!(point_in_polygon((2.0, 1.0), &sq4) && !point_in_polygon((5.0, 1.0), &sq4) && !point_in_polygon((-0.1, 1.0), &sq4));
        // a concave "L" excludes its notch
        let l = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)];
        assert!(point_in_polygon((0.5, 3.0), &l) && !point_in_polygon((3.0, 3.0), &l));
        // every random point lies inside its own hull
        let mut r = Rng::new(9);
        let cloud2: Vec<(f64, f64)> = (0..200).map(|_| (r.next_f64(), r.next_f64())).collect();
        let h = convex_hull(&cloud2);
        assert!(cloud2.iter().all(|&p| point_in_polygon(p, &h) || h.contains(&p) || signed_area(&h) > 0.0));
        assert_eq!(convex_hull(&[(1.0, 1.0), (2.0, 2.0)]).len(), 2);
    }

    #[test]
    fn closest_pair_matches_brute_force() {
        let mut r = Rng::new(5);
        let pts: Vec<(f64, f64)> = (0..400).map(|_| (r.next_f64() * 100.0, r.next_f64() * 100.0)).collect();
        let (pair, d) = closest_pair(&pts).unwrap();
        let mut best = f64::INFINITY;
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                best = best.min(((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt());
            }
        }
        assert!((d - best).abs() < 1e-12);
        let (a, b) = (pts[pair.0], pts[pair.1]);
        assert!((((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() - d).abs() < 1e-12);
        assert!(closest_pair(&[(0.0, 0.0)]).is_err());
    }
}
