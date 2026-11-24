## Summary
<!-- Briefly describe what this PR does and link any related issue (e.g., Fixes #4). -->

## Verification
- [ ] `cargo test -p oxideon-core --lib`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `pytest tests/ -q` (if Python bindings or Python code changed)
- [ ] `python tools/gen_stubs.py` (if Python signatures/docstrings changed)
