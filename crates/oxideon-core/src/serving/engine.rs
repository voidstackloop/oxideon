//! Continuous Batching Scheduler and LLM Serving Engine.
//!
//! Unlike static batching which pads to max sequence length and stalls until all requests finish,
//! continuous batching schedules requests at the iteration level: requests can join the running
//! batch during prefill and leave as soon as they generate an EOS token or hit `max_tokens`.

use std::collections::VecDeque;

use super::paged_kv::{BlockAllocator, BlockTable, PagedKVCache};
use super::prefix_cache::PrefixCache;
use crate::grammar::regex::{Regex, RegexConstraint};
use std::sync::Arc;
use crate::dtype::DType;
use crate::error::Result;

pub type RequestId = usize;

/// Client generation request submitted to the serving engine.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: RequestId,
    pub prompt_tokens: Vec<usize>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub eos_token_id: Option<usize>,
    pub sampling: crate::sampling::SamplingConfig,
    pub generated_tokens: Vec<usize>,
    pub block_table: BlockTable,
    pub is_finished: bool,
    /// Leading prompt tokens whose keys and values came from the prefix cache (the model need not recompute them).
    pub cached_prefix_tokens: usize,
    /// Optional output-format constraint; mask the logits with it, the engine advances it after every token.
    pub constraint: Option<RegexConstraint>,
}

impl BatchRequest {
    pub fn new(id: RequestId, prompt_tokens: Vec<usize>, max_new_tokens: usize) -> Self {
        let sampling = crate::sampling::SamplingConfig {
            max_new_tokens,
            ..Default::default()
        };
        Self {
            id,
            prompt_tokens,
            max_new_tokens,
            temperature: 1.0,
            eos_token_id: Some(2), // default EOS
            sampling,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(),
            is_finished: false,
            cached_prefix_tokens: 0,
            constraint: None,
        }
    }

    pub fn with_sampling(mut self, sampling: crate::sampling::SamplingConfig) -> Self {
        self.temperature = sampling.temperature;
        self.max_new_tokens = sampling.max_new_tokens;
        if let Some(&eos) = sampling.stop_tokens.first() {
            self.eos_token_id = Some(eos);
        }
        self.sampling = sampling;
        self
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
}

/// Dynamic iteration-level scheduler.
pub struct ContinuousBatchScheduler {
    pub max_batch_size: usize,
    pub chunked_prefill_size: usize,
    waiting: VecDeque<BatchRequest>,
    running: Vec<BatchRequest>,
    next_id: usize,
    /// When set, finished prompts leave their full blocks here and new prompts reuse them.
    pub prefix_cache: Option<PrefixCache>,
}

impl ContinuousBatchScheduler {
    pub fn new(max_batch_size: usize, chunked_prefill_size: usize) -> Self {
        Self {
            max_batch_size,
            chunked_prefill_size,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 1,
            prefix_cache: None,
        }
    }

    /// Submits a new prompt for generation, assigning an auto-incrementing `RequestId`.
    pub fn submit(&mut self, prompt: Vec<usize>, max_new_tokens: usize) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.waiting
            .push_back(BatchRequest::new(id, prompt, max_new_tokens));
        id
    }

    /// Submits a prompt with full [`crate::sampling::SamplingConfig`].
    pub fn submit_with_config(
        &mut self,
        prompt: Vec<usize>,
        config: crate::sampling::SamplingConfig,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        let max_new = config.max_new_tokens;
        let req = BatchRequest::new(id, prompt, max_new).with_sampling(config);
        self.waiting.push_back(req);
        id
    }

    /// Schedules the next step batch: promotes waiting requests into running if slots are available.
    pub fn schedule_step(&mut self, allocator: &mut BlockAllocator) -> Result<Vec<RequestId>> {
        if allocator.block_size == 0 {
            return Err(crate::error::Error::Msg(
                "schedule_step: allocator block_size must be positive".into(),
            ));
        }
        // Remove finished requests and free their physical memory
        let mut active = Vec::with_capacity(self.running.len());
        for req in self.running.drain(..) {
            if req.is_finished {
                if let Some(pc) = &mut self.prefix_cache {
                    pc.insert(allocator, &req.prompt_tokens, &req.block_table.physical_blocks);
                }
                for &b in &req.block_table.physical_blocks {
                    allocator.free(b);
                }
            } else {
                active.push(req);
            }
        }
        self.running = active;

        // Admit new requests from waiting queue up to max_batch_size
        while self.running.len() < self.max_batch_size && !self.waiting.is_empty() {
            let mut req = self.waiting.pop_front().unwrap();

            // Reuse cached prefix blocks (never the last prompt token: its logits must be computed), allocate the rest
            let bs = allocator.block_size;
            let total_blocks = req.prompt_tokens.len().div_ceil(bs).max(1);
            let shared = match &mut self.prefix_cache {
                Some(pc) => pc.match_prefix(allocator, &req.prompt_tokens[..req.prompt_tokens.len().saturating_sub(1)]),
                None => Vec::new(),
            };
            let needed_blocks = total_blocks - shared.len();
            if allocator.num_free_blocks() < needed_blocks
                && let Some(pc) = &mut self.prefix_cache
            {
                pc.evict(allocator, needed_blocks - allocator.num_free_blocks());
            }
            if allocator.num_free_blocks() < needed_blocks {
                // Not enough KV memory right now; give the shared references back and retry later
                for b in shared {
                    allocator.free(b);
                }
                self.waiting.push_front(req);
                break;
            }
            req.cached_prefix_tokens = shared.len() * bs;
            for b in shared {
                req.block_table.append_block(b);
            }
            for _ in 0..needed_blocks {
                let blk = allocator.allocate()?;
                req.block_table.append_block(blk);
            }
            self.running.push(req);
        }

        Ok(self.running.iter().map(|r| r.id).collect())
    }

    /// Gives every running request a block for the token it will write next; a request that cannot get one (KV memory
    /// exhausted even after evicting cached prefixes) is finished instead of corrupting another sequence's block.
    pub fn ensure_capacity(&mut self, allocator: &mut BlockAllocator) {
        let bs = allocator.block_size;
        for req in &mut self.running {
            while !req.is_finished && req.block_table.physical_blocks.len() < (req.total_tokens() + 1).div_ceil(bs) {
                if allocator.num_free_blocks() == 0
                    && let Some(pc) = &mut self.prefix_cache
                {
                    pc.evict(allocator, 1);
                }
                match allocator.allocate() {
                    Ok(b) => req.block_table.append_block(b),
                    Err(_) => req.is_finished = true,
                }
            }
        }
    }

    pub fn running_requests_mut(&mut self) -> &mut [BatchRequest] {
        &mut self.running
    }

    pub fn num_active(&self) -> usize {
        self.running.len() + self.waiting.len()
    }
}

/// End-to-end PagedAttention LLM Serving Engine.
pub struct ServingEngine {
    pub scheduler: ContinuousBatchScheduler,
    pub allocator: BlockAllocator,
    pub paged_kv: PagedKVCache,
    /// Token id → text, needed to advance regex constraints (set with [`ServingEngine::set_vocab`]).
    pub vocab: Option<Arc<Vec<String>>>,
}

impl ServingEngine {
    pub fn new(
        max_batch_size: usize,
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(crate::error::Error::Msg(
                "ServingEngine: block_size must be positive".into(),
            ));
        }
        crate::limits::check_at_most("num_blocks", num_blocks, 1 << 24)?;
        crate::limits::check_elements("KV cache", &[num_blocks, block_size, num_heads, head_dim])?;
        let scheduler = ContinuousBatchScheduler::new(max_batch_size, 512);
        let allocator = BlockAllocator::new(num_blocks, block_size)?;
        let paged_kv = PagedKVCache::new(num_blocks, block_size, num_heads, head_dim, DType::F32)?;
        Ok(Self {
            scheduler,
            allocator,
            paged_kv,
            vocab: None,
        })
    }

    /// Turns on prefix caching: prompts that start with tokens of an earlier request reuse its KV blocks.
    pub fn enable_prefix_cache(&mut self) {
        self.scheduler.prefix_cache.get_or_insert_with(PrefixCache::new);
    }

    pub fn set_vocab(&mut self, vocab: Vec<String>) {
        self.vocab = Some(Arc::new(vocab));
    }

    /// Submits a request whose output must match `pattern` (it finishes once an EOS is sampled after a full match).
    /// Requires [`set_vocab`](Self::set_vocab); the forward callback masks logits via `req.constraint`.
    pub fn submit_constrained(&mut self, prompt: Vec<usize>, config: crate::sampling::SamplingConfig, pattern: Arc<Regex>) -> Result<RequestId> {
        if self.vocab.is_none() {
            return Err(crate::error::Error::Msg("submit_constrained: call set_vocab first".into()));
        }
        let id = self.scheduler.submit_with_config(prompt, config);
        let c = RegexConstraint::new(pattern);
        if let Some(r) = self.scheduler.waiting.iter_mut().find(|r| r.id == id) {
            r.constraint = Some(c);
        }
        Ok(id)
    }

    /// Submits a request and returns its assigned ID.
    pub fn submit(&mut self, prompt: Vec<usize>, max_new_tokens: usize) -> RequestId {
        self.scheduler.submit(prompt, max_new_tokens)
    }

    /// Submits a request with custom [`crate::sampling::SamplingConfig`].
    pub fn submit_with_config(
        &mut self,
        prompt: Vec<usize>,
        config: crate::sampling::SamplingConfig,
    ) -> RequestId {
        self.scheduler.submit_with_config(prompt, config)
    }

    /// Executes one iteration of generation across all active running requests.
    /// Passes next token step to `forward_step` callback.
    pub fn step<F>(&mut self, mut forward_step: F) -> Result<Vec<(RequestId, usize)>>
    where
        F: FnMut(&[BatchRequest]) -> Result<Vec<usize>>,
    {
        let _active_ids = self.scheduler.schedule_step(&mut self.allocator)?;
        if self.scheduler.running_requests_mut().is_empty() {
            return Ok(Vec::new());
        }

        // Call model forward function to get the next token for each running request
        let next_tokens = forward_step(self.scheduler.running_requests_mut())?;
        let running = self.scheduler.running_requests_mut().len();
        if next_tokens.len() != running {
            return Err(crate::error::Error::Msg(format!(
                "engine step: forward returned {} tokens for {} running requests",
                next_tokens.len(),
                running
            )));
        }

        let mut completed_step = Vec::new();
        for (req, &tok) in self
            .scheduler
            .running_requests_mut()
            .iter_mut()
            .zip(next_tokens.iter())
        {
            req.generated_tokens.push(tok);
            completed_step.push((req.id, tok));
            if let (Some(c), Some(v)) = (&mut req.constraint, &self.vocab)
                && req.eos_token_id != Some(tok)
                && let Some(text) = v.get(tok)
            {
                c.advance(text);
            }

            // Check stopping conditions
            if req.generated_tokens.len() >= req.max_new_tokens || req.eos_token_id == Some(tok) {
                req.is_finished = true;
            }
        }

        self.scheduler.ensure_capacity(&mut self.allocator);
        Ok(completed_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingConfig;
    use crate::tensor::Tensor;

    fn engine(blocks: usize) -> ServingEngine {
        ServingEngine::new(4, blocks, 4, 1, 2).unwrap()
    }

    #[test]
    fn a_second_request_reuses_the_first_requests_prompt_blocks() {
        let mut e = engine(8);
        e.enable_prefix_cache();
        let prompt: Vec<usize> = (10..20).collect(); // 2 full blocks + 2 tokens
        e.submit(prompt.clone(), 1);
        e.step(|r| Ok(vec![5; r.len()])).unwrap(); // finishes after one token
        let mut p2 = prompt[..8].to_vec();
        p2.extend([77, 78, 79]);
        e.submit(p2, 1);
        let mut seen = Vec::new();
        e.step(|r| {
            seen.extend(r.iter().map(|q| q.cached_prefix_tokens));
            Ok(vec![5; r.len()])
        })
        .unwrap();
        assert_eq!(seen, vec![8]);
        // the shared blocks are held by the cache and the new request, the rest was allocated fresh
        let free_before = e.allocator.num_free_blocks();
        e.step(|r| Ok(vec![5; r.len()])).unwrap(); // request 2 is finished and released here
        assert!(e.allocator.num_free_blocks() > free_before - 1);
    }

    #[test]
    fn running_out_of_kv_memory_ends_the_request_instead_of_overwriting_blocks() {
        let mut e = engine(2); // 8 token slots in total
        e.submit(vec![10, 11, 12], 100);
        let mut steps = 0;
        while steps < 50 {
            if e.step(|r| Ok(vec![5; r.len()])).unwrap().is_empty() {
                break;
            }
            steps += 1;
        }
        assert!(steps < 50, "the request must stop when memory runs out");
        assert!(steps <= 6, "{steps}");
    }

    #[test]
    fn constrained_requests_only_ever_emit_text_that_matches_the_pattern() {
        let vocab: Vec<String> = ["<pad>", "<unk>", "<eos>", "a", "b"].iter().map(|s| s.to_string()).collect();
        let mut e = engine(8);
        e.set_vocab(vocab.clone());
        let re = Arc::new(Regex::new("ab+").unwrap());
        let cfg = SamplingConfig { max_new_tokens: 4, ..SamplingConfig::greedy() };
        let id = e.submit_constrained(vec![9], cfg, re.clone()).unwrap();
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 3.0, 2.0], &[5]).unwrap(); // the "model" always prefers a > b > eos
        let mut text = String::new();
        loop {
            let out = e
                .step(|reqs| {
                    reqs.iter()
                        .map(|r| {
                            let mut c = r.constraint.clone().unwrap();
                            let m = c.mask_logits(&logits, &vocab, Some(2)).unwrap().to_vec::<f32>().unwrap();
                            Ok((0..m.len()).max_by(|&a, &b| m[a].partial_cmp(&m[b]).unwrap()).unwrap())
                        })
                        .collect()
                })
                .unwrap();
            if out.is_empty() {
                break;
            }
            for (rid, tok) in out {
                assert_eq!(rid, id);
                text.push_str(&vocab[tok]);
            }
        }
        assert_eq!(text, "abbb");
        assert!(re.is_match(&text));
        assert!(e.submit_constrained(vec![1], SamplingConfig::greedy(), re).is_ok());
        let mut bare = engine(2);
        assert!(bare.submit_constrained(vec![1], SamplingConfig::greedy(), Arc::new(Regex::new("a").unwrap())).is_err());
    }
}
