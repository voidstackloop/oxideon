//! Classical graph algorithms on a small weighted [`Graph`]: BFS/DFS, shortest paths ([`Graph::dijkstra`],
//! [`Graph::bellman_ford`], [`Graph::floyd_warshall`]), components and strongly connected components, topological
//! order, [`Graph::kruskal`] MST, [`Graph::pagerank`], [`Graph::betweenness`] (Brandes), clustering coefficients,
//! [`Graph::communities`] (greedy modularity), and [`max_flow`] (Edmonds-Karp).

use crate::error::{Error, Result};
use std::collections::{BinaryHeap, VecDeque};

/// Weighted graph on nodes `0..n` stored as adjacency lists.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    pub directed: bool,
    pub adj: Vec<Vec<(usize, f64)>>,
}

#[derive(PartialEq)]
struct Item(f64, usize);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0) // reversed: smallest distance first
    }
}

impl Graph {
    pub fn new(n: usize, directed: bool) -> Self {
        Self { n, directed, adj: vec![Vec::new(); n] }
    }

    pub fn from_edges(n: usize, directed: bool, edges: &[(usize, usize, f64)]) -> Result<Self> {
        let mut g = Self::new(n, directed);
        for &(a, b, w) in edges {
            g.add_edge(a, b, w)?;
        }
        Ok(g)
    }

    pub fn add_edge(&mut self, a: usize, b: usize, w: f64) -> Result<()> {
        if a >= self.n || b >= self.n || w.is_nan() {
            return Err(Error::Msg(format!("add_edge: invalid edge ({a}, {b}, {w}) for {} nodes", self.n)));
        }
        self.adj[a].push((b, w));
        if !self.directed && a != b {
            self.adj[b].push((a, w));
        }
        Ok(())
    }

    /// Nodes reachable from `s` in breadth-first order.
    pub fn bfs(&self, s: usize) -> Vec<usize> {
        let mut seen = vec![false; self.n];
        let (mut order, mut q) = (Vec::new(), VecDeque::from([s]));
        seen[s] = true;
        while let Some(u) = q.pop_front() {
            order.push(u);
            for &(v, _) in &self.adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
        order
    }

    /// Nodes reachable from `s` in depth-first (pre-)order.
    pub fn dfs(&self, s: usize) -> Vec<usize> {
        let mut seen = vec![false; self.n];
        let (mut order, mut stack) = (Vec::new(), vec![s]);
        while let Some(u) = stack.pop() {
            if seen[u] {
                continue;
            }
            seen[u] = true;
            order.push(u);
            stack.extend(self.adj[u].iter().rev().map(|e| e.0).filter(|&v| !seen[v]));
        }
        order
    }

    /// Shortest distances and predecessors from `s` (non-negative weights).
    pub fn dijkstra(&self, s: usize) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        if self.adj.iter().flatten().any(|e| e.1 < 0.0) {
            return Err(Error::Msg("dijkstra: negative edge weight (use bellman_ford)".into()));
        }
        let mut dist = vec![f64::INFINITY; self.n];
        let mut prev = vec![None; self.n];
        let mut heap = BinaryHeap::from([Item(0.0, s)]);
        dist[s] = 0.0;
        while let Some(Item(d, u)) = heap.pop() {
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &self.adj[u] {
                if d + w < dist[v] {
                    dist[v] = d + w;
                    prev[v] = Some(u);
                    heap.push(Item(dist[v], v));
                }
            }
        }
        Ok((dist, prev))
    }

    /// The node sequence of a shortest path from the source of `prev` to `t` (empty if unreachable).
    pub fn path_to(prev: &[Option<usize>], dist: &[f64], t: usize) -> Vec<usize> {
        if dist[t].is_infinite() {
            return Vec::new();
        }
        let mut p = vec![t];
        while let Some(u) = prev[*p.last().unwrap()] {
            p.push(u);
        }
        p.reverse();
        p
    }

    /// Single-source shortest paths allowing negative weights; errors on a reachable negative cycle.
    pub fn bellman_ford(&self, s: usize) -> Result<Vec<f64>> {
        let mut dist = vec![f64::INFINITY; self.n];
        dist[s] = 0.0;
        for round in 0..=self.n {
            let mut changed = false;
            for u in 0..self.n {
                if dist[u].is_finite() {
                    for &(v, w) in &self.adj[u] {
                        if dist[u] + w < dist[v] - 1e-12 {
                            dist[v] = dist[u] + w;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(dist);
            }
            if round == self.n {
                return Err(Error::Msg("bellman_ford: negative cycle reachable from the source".into()));
            }
        }
        Ok(dist)
    }

    /// All-pairs shortest distances.
    pub fn floyd_warshall(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![f64::INFINITY; self.n]; self.n];
        for u in 0..self.n {
            d[u][u] = 0.0;
            for &(v, w) in &self.adj[u] {
                d[u][v] = d[u][v].min(w);
            }
        }
        for k in 0..self.n {
            for i in 0..self.n {
                for j in 0..self.n {
                    if d[i][k] + d[k][j] < d[i][j] {
                        d[i][j] = d[i][k] + d[k][j];
                    }
                }
            }
        }
        d
    }

    /// Connected-component id per node (weakly connected for directed graphs).
    pub fn components(&self) -> Vec<usize> {
        let mut undirected = vec![Vec::new(); self.n];
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                undirected[u].push(v);
                undirected[v].push(u);
            }
        }
        let mut comp = vec![usize::MAX; self.n];
        let mut c = 0;
        for s in 0..self.n {
            if comp[s] != usize::MAX {
                continue;
            }
            let mut stack = vec![s];
            comp[s] = c;
            while let Some(u) = stack.pop() {
                for &v in &undirected[u] {
                    if comp[v] == usize::MAX {
                        comp[v] = c;
                        stack.push(v);
                    }
                }
            }
            c += 1;
        }
        comp
    }

    /// Strongly connected components (Tarjan, iterative); ids are in reverse topological order of the condensation.
    pub fn scc(&self) -> Vec<usize> {
        let (mut index, mut low) = (vec![usize::MAX; self.n], vec![0; self.n]);
        let (mut on, mut stack, mut comp) = (vec![false; self.n], Vec::new(), vec![usize::MAX; self.n]);
        let (mut counter, mut c) = (0, 0);
        for root in 0..self.n {
            if index[root] != usize::MAX {
                continue;
            }
            let mut work = vec![(root, 0usize)];
            while let Some(&(u, i)) = work.last() {
                if i == 0 {
                    index[u] = counter;
                    low[u] = counter;
                    counter += 1;
                    stack.push(u);
                    on[u] = true;
                }
                if i < self.adj[u].len() {
                    work.last_mut().unwrap().1 += 1;
                    let v = self.adj[u][i].0;
                    if index[v] == usize::MAX {
                        work.push((v, 0));
                    } else if on[v] {
                        low[u] = low[u].min(index[v]);
                    }
                } else {
                    work.pop();
                    if let Some(&(p, _)) = work.last() {
                        low[p] = low[p].min(low[u]);
                    }
                    if low[u] == index[u] {
                        while let Some(w) = stack.pop() {
                            on[w] = false;
                            comp[w] = c;
                            if w == u {
                                break;
                            }
                        }
                        c += 1;
                    }
                }
            }
        }
        comp
    }

    /// Kahn topological order of a directed acyclic graph (errors on a cycle).
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut indeg = vec![0usize; self.n];
        self.adj.iter().flatten().for_each(|e| indeg[e.0] += 1);
        let mut q: VecDeque<usize> = (0..self.n).filter(|&u| indeg[u] == 0).collect();
        let mut order = Vec::with_capacity(self.n);
        while let Some(u) = q.pop_front() {
            order.push(u);
            for &(v, _) in &self.adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    q.push_back(v);
                }
            }
        }
        if order.len() == self.n { Ok(order) } else { Err(Error::Msg("topological_sort: the graph has a cycle".into())) }
    }

    /// Minimum spanning forest of an undirected graph: `(edges, total weight)`.
    pub fn kruskal(&self) -> (Vec<(usize, usize, f64)>, f64) {
        let mut edges: Vec<(usize, usize, f64)> = (0..self.n).flat_map(|u| self.adj[u].iter().filter(move |e| e.0 >= u).map(move |e| (u, e.0, e.1))).collect();
        edges.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut parent: Vec<usize> = (0..self.n).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut c = x;
            while p[c] != r {
                let n = p[c];
                p[c] = r;
                c = n;
            }
            r
        }
        let (mut tree, mut total) = (Vec::new(), 0.0);
        for (a, b, w) in edges {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
                tree.push((a, b, w));
                total += w;
            }
        }
        (tree, total)
    }

    /// PageRank by power iteration (dangling nodes spread uniformly).
    pub fn pagerank(&self, damping: f64, iters: usize) -> Vec<f64> {
        let n = self.n as f64;
        let mut r = vec![1.0 / n; self.n];
        for _ in 0..iters {
            let mut next = vec![(1.0 - damping) / n; self.n];
            let mut dangling = 0.0;
            for u in 0..self.n {
                let out: f64 = self.adj[u].iter().map(|e| e.1.abs()).sum();
                if out == 0.0 {
                    dangling += r[u];
                } else {
                    for &(v, w) in &self.adj[u] {
                        next[v] += damping * r[u] * w.abs() / out;
                    }
                }
            }
            next.iter_mut().for_each(|v| *v += damping * dangling / n);
            r = next;
        }
        r
    }

    /// Betweenness centrality of every node by Brandes' algorithm on hop counts (normalized like NetworkX).
    pub fn betweenness(&self) -> Vec<f64> {
        let mut cb = vec![0.0; self.n];
        for s in 0..self.n {
            let (mut stack, mut pred) = (Vec::new(), vec![Vec::new(); self.n]);
            let (mut sigma, mut dist) = (vec![0.0; self.n], vec![-1i64; self.n]);
            sigma[s] = 1.0;
            dist[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                stack.push(u);
                for &(v, _) in &self.adj[u] {
                    if dist[v] < 0 {
                        dist[v] = dist[u] + 1;
                        q.push_back(v);
                    }
                    if dist[v] == dist[u] + 1 {
                        sigma[v] += sigma[u];
                        pred[v].push(u);
                    }
                }
            }
            let mut delta = vec![0.0; self.n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
                }
                if w != s {
                    cb[w] += delta[w];
                }
            }
        }
        let n = self.n as f64;
        let scale = if self.n > 2 { 1.0 / ((n - 1.0) * (n - 2.0)) } else { 1.0 };
        cb.iter().map(|v| v * scale).collect()
    }

    /// Local clustering coefficient per node (undirected view, simple graph).
    pub fn clustering(&self) -> Vec<f64> {
        let sets: Vec<std::collections::HashSet<usize>> = self.adj.iter().enumerate().map(|(u, a)| a.iter().map(|e| e.0).filter(|&v| v != u).collect()).collect();
        (0..self.n)
            .map(|u| {
                let k = sets[u].len();
                if k < 2 {
                    return 0.0;
                }
                let links = sets[u].iter().map(|&a| sets[u].iter().filter(|&&b| b > a && sets[a].contains(&b)).count()).sum::<usize>();
                2.0 * links as f64 / (k * (k - 1)) as f64
            })
            .collect()
    }

    /// Communities by greedy modularity optimization (the local-moving phase of Louvain) on the undirected view;
    /// edge weights count. Returns one label per node.
    pub fn communities(&self, max_sweeps: usize) -> Vec<usize> {
        let n = self.n;
        let mut nbr: Vec<std::collections::BTreeMap<usize, f64>> = vec![Default::default(); n];
        for u in 0..n {
            for &(v, w) in &self.adj[u] {
                if u != v {
                    *nbr[u].entry(v).or_insert(0.0) += w.abs();
                    if self.directed {
                        *nbr[v].entry(u).or_insert(0.0) += w.abs();
                    }
                }
            }
        }
        let k: Vec<f64> = nbr.iter().map(|m| m.values().sum()).collect();
        let m2: f64 = k.iter().sum(); // 2m
        let mut label: Vec<usize> = (0..n).collect();
        if m2 == 0.0 {
            return label;
        }
        let mut tot = k.clone(); // total degree per community
        for _ in 0..max_sweeps {
            let mut moved = false;
            for u in 0..n {
                let mut to: std::collections::BTreeMap<usize, f64> = Default::default();
                for (&v, &w) in &nbr[u] {
                    *to.entry(label[v]).or_insert(0.0) += w;
                }
                let cur = label[u];
                tot[cur] -= k[u];
                // modularity gain of joining community c: k_u,in(c) - tot(c) * k_u / 2m
                let gain = |c: usize, to: &std::collections::BTreeMap<usize, f64>| to.get(&c).copied().unwrap_or(0.0) - tot[c] * k[u] / m2;
                let mut best = (gain(cur, &to), cur);
                for &c in to.keys() {
                    let g = gain(c, &to);
                    if g > best.0 + 1e-12 {
                        best = (g, c);
                    }
                }
                tot[best.1] += k[u];
                if best.1 != cur {
                    label[u] = best.1;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        label
    }
}

/// Maximum flow from `s` to `t` on a directed capacity graph (Edmonds-Karp); returns the flow value.
pub fn max_flow(n: usize, edges: &[(usize, usize, f64)], s: usize, t: usize) -> Result<f64> {
    if s >= n || t >= n || s == t || edges.iter().any(|&(a, b, c)| a >= n || b >= n || c < 0.0 || c.is_nan()) {
        return Err(Error::Msg("max_flow: invalid source/sink or edge".into()));
    }
    let mut cap = vec![vec![0.0; n]; n];
    edges.iter().for_each(|&(a, b, c)| cap[a][b] += c);
    let mut flow = 0.0;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[s] = s;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for v in 0..n {
                if parent[v] == usize::MAX && cap[u][v] > 1e-12 {
                    parent[v] = u;
                    q.push_back(v);
                }
            }
        }
        if parent[t] == usize::MAX {
            return Ok(flow);
        }
        let mut aug = f64::INFINITY;
        let mut v = t;
        while v != s {
            aug = aug.min(cap[parent[v]][v]);
            v = parent[v];
        }
        v = t;
        while v != s {
            cap[parent[v]][v] -= aug;
            cap[v][parent[v]] += aug;
            v = parent[v];
        }
        flow += aug;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    fn weighted() -> Graph {
        // 0-1 (7), 0-2 (9), 0-5 (14), 1-2 (10), 1-3 (15), 2-3 (11), 2-5 (2), 3-4 (6), 4-5 (9)  (classic Dijkstra example)
        Graph::from_edges(6, false, &[(0, 1, 7.0), (0, 2, 9.0), (0, 5, 14.0), (1, 2, 10.0), (1, 3, 15.0), (2, 3, 11.0), (2, 5, 2.0), (3, 4, 6.0), (4, 5, 9.0)]).unwrap()
    }

    #[test]
    fn traversal_and_shortest_paths() {
        let g = weighted();
        assert_eq!(g.bfs(0).len(), 6);
        assert_eq!(g.dfs(0)[0], 0);
        let (d, prev) = g.dijkstra(0).unwrap();
        assert_eq!(d, vec![0.0, 7.0, 9.0, 20.0, 20.0, 11.0]);
        assert_eq!(Graph::path_to(&prev, &d, 4), vec![0, 2, 5, 4]);
        assert_eq!(g.bellman_ford(0).unwrap(), d);
        let fw = g.floyd_warshall();
        assert_eq!(fw[0], d);
        assert!(fw.iter().enumerate().all(|(i, r)| r.iter().enumerate().all(|(j, v)| (v - fw[j][i]).abs() < 1e-12 && (i != j || *v == 0.0))));
        // negative weights: dijkstra refuses, bellman-ford handles; negative cycle is detected
        let neg = Graph::from_edges(3, true, &[(0, 1, 4.0), (1, 2, -2.0), (0, 2, 3.0)]).unwrap();
        assert!(neg.dijkstra(0).is_err());
        assert_eq!(neg.bellman_ford(0).unwrap(), vec![0.0, 4.0, 2.0]);
        let cyc = Graph::from_edges(3, true, &[(0, 1, 1.0), (1, 2, -3.0), (2, 1, 1.0)]).unwrap();
        assert!(cyc.bellman_ford(0).is_err());
        assert!(Graph::new(2, false).add_edge(0, 5, 1.0).is_err());
    }

    #[test]
    fn structure() {
        let g = Graph::from_edges(6, false, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]).unwrap();
        let c = g.components();
        assert!(c[0] == c[1] && c[1] == c[2] && c[3] == c[4] && c[0] != c[3] && c[5] != c[0] && c[5] != c[3]);
        let d = Graph::from_edges(5, true, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 3, 1.0)]).unwrap();
        let s = d.scc();
        assert!(s[0] == s[1] && s[1] == s[2] && s[3] == s[4] && s[0] != s[3]);
        let dag = Graph::from_edges(4, true, &[(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]).unwrap();
        let order = dag.topological_sort().unwrap();
        let pos = |x: usize| order.iter().position(|&v| v == x).unwrap();
        assert!(pos(0) < pos(1) && pos(0) < pos(2) && pos(1) < pos(3) && pos(2) < pos(3));
        assert!(d.topological_sort().is_err());
    }

    #[test]
    fn mst_and_flow() {
        let (tree, total) = weighted().kruskal();
        assert_eq!(tree.len(), 5);
        close(total, 2.0 + 6.0 + 7.0 + 9.0 + 9.0); // 2-5, 3-4, 0-1, 0-2, 4-5
        // CLRS flow network: max flow 23
        let e = [(0, 1, 16.0), (0, 2, 13.0), (1, 2, 10.0), (2, 1, 4.0), (1, 3, 12.0), (3, 2, 9.0), (2, 4, 14.0), (4, 3, 7.0), (3, 5, 20.0), (4, 5, 4.0)];
        close(max_flow(6, &e, 0, 5).unwrap(), 23.0);
        assert!(max_flow(6, &e, 0, 0).is_err());
    }

    #[test]
    fn centralities_and_communities() {
        // path 0-1-2-3-4: the middle node carries the most shortest paths
        let p = Graph::from_edges(5, false, &(0..4).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>()).unwrap();
        let b = p.betweenness();
        // NetworkX normalized betweenness of the 5-path: [0, 0.5, 0.667, 0.5, 0] (undirected counts each pair twice)
        close(b[0], 0.0);
        close(b[2], 4.0 / 6.0);
        close(b[1], 3.0 / 6.0);
        // star: the hub has the maximal PageRank, and ranks sum to one
        let star = Graph::from_edges(5, false, &(1..5).map(|i| (0, i, 1.0)).collect::<Vec<_>>()).unwrap();
        let pr = star.pagerank(0.85, 100);
        close(pr.iter().sum::<f64>(), 1.0);
        assert!(pr[0] > pr[1] * 2.0);
        // triangle plus a pendant: clustering 1 for triangle corners without the pendant link
        let t = Graph::from_edges(4, false, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0)]).unwrap();
        let cc = t.clustering();
        close(cc[0], 1.0);
        close(cc[2], 1.0 / 3.0);
        close(cc[3], 0.0);
        // two cliques joined by one edge split into two communities
        let mut edges = Vec::new();
        for grp in [[0, 1, 2, 3], [4, 5, 6, 7]] {
            for i in 0..4 {
                for j in i + 1..4 {
                    edges.push((grp[i], grp[j], 1.0));
                }
            }
        }
        edges.push((3, 4, 1.0));
        let lp = Graph::from_edges(8, false, &edges).unwrap().communities(50);
        assert!(lp[0] == lp[1] && lp[1] == lp[2] && lp[4] == lp[5] && lp[5] == lp[6] && lp[0] != lp[4], "{lp:?}");
    }
}
