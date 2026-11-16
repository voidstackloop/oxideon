//! Prefix caching for the paged KV cache: full blocks of a finished (or running) prompt stay in memory, keyed by the
//! chain of token blocks that led to them, so a new request that starts with the same tokens (a system prompt, a chat
//! history) reuses those physical blocks instead of recomputing their keys and values.
//!
//! Sharing rides on the allocator's reference counts: the cache holds one reference per cached block and every sequence
//! that matches takes one more. Blocks that only the cache still holds are evicted least-recently-used first, leaves
//! before the blocks they extend.

use super::paged_kv::{BlockAllocator, PhysicalBlockId};
use std::collections::HashMap;

struct Entry {
    block: PhysicalBlockId,
    tokens: Vec<usize>,
    parent: Option<u64>,
    children: usize,
    last_used: u64,
}

#[derive(Default)]
pub struct PrefixCache {
    entries: HashMap<u64, Entry>,
    clock: u64,
    pub hits: usize,
    pub misses: usize,
}

fn chain_hash(parent: Option<u64>, tokens: &[usize]) -> u64 {
    let mut h = parent.unwrap_or(0xcbf2_9ce4_8422_2325);
    for &t in tokens {
        h = (h ^ t as u64).wrapping_mul(0x0100_0000_01b3);
        h ^= h >> 29;
    }
    h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The longest cached run of full blocks that `tokens` starts with. The caller receives one new reference on each
    /// returned block (release them with `BlockAllocator::free` when the sequence ends); the cached tokens are
    /// `blocks.len() * block_size`.
    pub fn match_prefix(&mut self, alloc: &mut BlockAllocator, tokens: &[usize]) -> Vec<PhysicalBlockId> {
        let bs = alloc.block_size;
        let mut out = Vec::new();
        let mut parent = None;
        for chunk in tokens.chunks_exact(bs) {
            let key = chain_hash(parent, chunk);
            match self.entries.get_mut(&key) {
                Some(e) if e.tokens == chunk && e.parent == parent => {
                    self.clock += 1;
                    e.last_used = self.clock;
                    alloc.inc_ref(e.block);
                    out.push(e.block);
                    parent = Some(key);
                }
                _ => break,
            }
        }
        if out.is_empty() { self.misses += 1 } else { self.hits += 1 }
        out
    }

    /// Registers the full blocks of a sequence (`blocks[i]` holds `tokens[i*bs..(i+1)*bs]`); blocks already cached are
    /// left alone. The cache takes its own reference on every block it adds.
    pub fn insert(&mut self, alloc: &mut BlockAllocator, tokens: &[usize], blocks: &[PhysicalBlockId]) {
        let bs = alloc.block_size;
        let mut parent = None;
        for (chunk, &block) in tokens.chunks_exact(bs).zip(blocks) {
            let key = chain_hash(parent, chunk);
            match self.entries.get(&key) {
                Some(e) if e.tokens == chunk && e.parent == parent => {}
                Some(_) => break, // a hash collision with different content: stop rather than alias blocks
                None => {
                    alloc.inc_ref(block);
                    self.clock += 1;
                    self.entries.insert(key, Entry { block, tokens: chunk.to_vec(), parent, children: 0, last_used: self.clock });
                    if let Some(p) = parent.and_then(|p| self.entries.get_mut(&p)) {
                        p.children += 1;
                    }
                }
            }
            parent = Some(key);
        }
    }

    /// Frees up to `want` blocks that no sequence uses any more. Returns how many were freed.
    pub fn evict(&mut self, alloc: &mut BlockAllocator, want: usize) -> usize {
        let mut freed = 0;
        while freed < want {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.children == 0 && alloc.ref_count(e.block) == 1)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(&k, _)| k);
            let Some(k) = victim else { break };
            let e = self.entries.remove(&k).unwrap();
            if let Some(p) = e.parent.and_then(|p| self.entries.get_mut(&p)) {
                p.children -= 1;
            }
            alloc.free(e.block);
            freed += 1;
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(alloc: &mut BlockAllocator, n_tokens: usize) -> Vec<PhysicalBlockId> {
        (0..n_tokens.div_ceil(alloc.block_size)).map(|_| alloc.allocate().unwrap()).collect()
    }

    #[test]
    fn shared_prefixes_reuse_blocks_diverging_ones_do_not_and_eviction_is_leaf_first() {
        let mut a = BlockAllocator::new(8, 4).unwrap();
        let mut c = PrefixCache::new();
        let sys: Vec<usize> = (0..10).collect(); // two full blocks and a partial one

        // sequence A computes everything, registers its full blocks, then finishes
        let a_blocks = seq(&mut a, sys.len());
        c.insert(&mut a, &sys, &a_blocks);
        assert_eq!(c.len(), 2);
        for &b in &a_blocks {
            a.free(b);
        }
        assert_eq!(a.num_free_blocks(), 8 - 2); // only the two cached blocks stay resident

        // sequence B starts with the same 8 tokens: both blocks are reused, nothing new is allocated
        let mut b_tokens = sys[..8].to_vec();
        b_tokens.extend([100, 101]);
        let shared = c.match_prefix(&mut a, &b_tokens);
        assert_eq!(shared, a_blocks[..2].to_vec());
        assert_eq!(a.ref_count(shared[0]), 2);
        assert_eq!(a.num_free_blocks(), 6);

        // sequence C shares only the first block (its second differs)
        let mut c_tokens = sys[..4].to_vec();
        c_tokens.extend([7, 7, 7, 7]);
        let shared_c = c.match_prefix(&mut a, &c_tokens);
        assert_eq!(shared_c, vec![a_blocks[0]]);
        assert_eq!(c.hits, 2);
        assert!(c.match_prefix(&mut a, &[9, 9, 9, 9]).is_empty());
        assert_eq!(c.misses, 1);

        // blocks in use by B and C cannot be evicted
        assert_eq!(c.evict(&mut a, 2), 0);
        for &b in shared.iter().chain(&shared_c) {
            a.free(b);
        }
        // now both are cache-only: the leaf (block 2) goes before its parent
        assert_eq!(c.evict(&mut a, 1), 1);
        assert_eq!(c.len(), 1);
        assert!(c.match_prefix(&mut a, &sys[..8]).len() == 1);
        a.free(a_blocks[0]);
        assert_eq!(c.evict(&mut a, 5), 1);
        assert!(c.is_empty());
        assert_eq!(a.num_free_blocks(), 8);
    }
}
