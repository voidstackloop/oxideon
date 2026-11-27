# Oxideon API Stability Statement

**Version:** `0.1.1` (pre-1.0: no surface is frozen yet).
**Last updated:** 2026-11-10.

This document states which parts of the public Python API (`oxideon`
package, typed stubs in `python/oxideon/__init__.pyi`) can be relied on,
how stability is earned, and what compatibility users can expect.

## Tier definitions

| Tier | Name | Meaning |
|---|---|---|
| 1 | Hardened | Covered by dedicated bug-hunt regression batches (`tests/test_bug_hunt_*.py`, fail-pre-fix) plus oracle parity tests against PyTorch/NumPy/SciPy where applicable. Behavior changes here are treated as bugs until proven otherwise. |
| 2 | Covered | Exercised by `tests/` and pinned in the generated stubs, but without a dedicated adversarial hunt. Tested behavior holds; untested corners may still be fixed (including behavior fixes) without a deprecation period. |
| 3 | Provisional | New, fully tested API whose *shape* may still evolve based on user feedback. Behavior is tested; names and signatures carry no compatibility promise yet. |
| 4 | Experimental | May change or disappear without notice. Do not build durable dependencies on it. |

Anything starting with `_`, plus `tools/` and `tests/`, is internal and
carries no compatibility promise at any tier.

## Module classification

**Tier 1 — Hardened** (dedicated regression batches in `BUG_HUNT_2026-11-10.md`,
plus the Rust core behind `cargo test`):

- `ox.optim`, `ox.optim.lr_scheduler` (Batch CT: param validation,
  `load_state_dict` live-param mapping, LBFGS/NAdam correctness,
  per-group schedulers)
- `ox.serialization` (Batch CT: strict-error propagation, pickle fallback rules)
- `ox.vision.ops`, `ox.vision.transforms` (Batch CP, plus CQ validation follow-ups)
- `ox.fx` (Batches CQ/CS: tracing, node/proxy guards)
- `ox.distributions`, `ox.linalg` (Batches CQ/CR, incl. no-SciPy LU fallback)
- `ox.losses`, `ox.init`, `ox.kernels`, `ox.spatial` (Batch CR)
- `ox.ao.quantization`, `ox.quant_compression`, `ox.bitnet`, `ox.checkpoint`
  (Batch CS, incl. QAT borrow-and-restore and checkpoint RNG restore)
- `ox.distributed`, `ox.memory` (Batch CU: collective op/dtype validation,
  reduce-scatter semantics, parallel-container gather paths, KVMemoryBank
  empty-read grad connectivity)
- `ox.nn` normalization/exotic layers + `ox.functional` pooling/pad/LRN
  (Batch CV: BatchNorm/GroupNorm/margin-loss/PReLU/adaptive-max/LPPool/
  pad/LRN gradient flow restored, Flatten dims, CTCLoss plumbing —
  all torch-parity verified)
- Rust tensor/autograd core (`cargo test --workspace`: 2013 tests green)

Earlier dated `BUG_HUNT_*.md` logs record additional hardened surface from
previous campaigns; when in doubt, the per-batch regression files
(`tests/test_bug_hunt_*.py`) are the authoritative record.

**Tier 3 — Provisional** (new, scipy-oracle tested, shape may evolve):

- `ox.numerical` (quadrature, interpolation, scalar optimization, curve fitting)
- `ox.stats` (classical hypothesis tests, ECDF, p-value adjustment)
- `ox.physics` (wave/diffusion PDEs, N-body, symplectic integration)

**Tier 4 — Experimental:**

- `crates/oxideon-engines` surface and `docs/ENGINES_CATALOG.md` engines
- Singularity-milestone showcases and `browser_demo/`
- Anything marked experimental in its docstring or docs page

**Tier 2 — Covered:** every other public module: pinned in stubs
(`tools/gen_stubs.py` is idempotent; 2584 symbols) and exercised by the
suite, hunts pending.

## Guarantees (all tiers)

1. **Error contract.** Library code raises descriptive `ValueError` /
   `TypeError` / `RuntimeError`; no bare `assert`s, no bare `IndexError`
   from validated entry points. Error *types* for documented misuse are
   part of the contract; messages may improve.
2. **Oracle parity intent.** Where PyTorch/NumPy/SciPy define the
   semantics (optimizers, schedulers, linalg, distributions, stats),
   matching them is the spec; divergence is a bug (file it with a
   reproducer against the oracle).
3. **Serialization durability.** Files written by `ox.save` /
   `ox.save_file` remain loadable by later versions; the safetensors
   subset is additionally constrained by the safetensors format itself.
4. **Stubs are the surface record.** The generated `__init__.pyi` is the
   authoritative public-symbol list; CI-gate idempotency
   (`tools/gen_stubs.py` runs clean twice in a row) keeps it honest.

## Change policy (pre-1.0)

- Tier 1 behavior fixes land as bug fixes with a regression test; if a
  fix must change long-standing behavior, it is called out in
  `CHANGELOG.md` under `Unreleased`.
- Tier 2→1 promotion happens by running a bug-hunt batch (prompt:
  `BUG_HUNT_PROMPT.md`) and logging it; Tier 3→2 promotion happens by
  freezing the API after one release of real use with no shape changes.
- After 1.0, Tier 1/2 removals and signature breaks will follow
  deprecation warnings over one minor release; Tier 3/4 keep the
  pre-1.0 rules.

## What this does NOT promise

- Numerical bit-identity across devices/backends or versions (tolerances
  in tests, e.g. `atol`/`rtol` against oracles, are the promise).
- Performance characteristics or absence of slowness in any path.
- That Tier 2/4 code is bug-free — only that Tier 1 has survived an
  adversarial hunt and Tier 3 matches its oracles.
