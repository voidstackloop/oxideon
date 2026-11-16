//! Upper bounds for user-chosen work sizes. A typo (or `2**63`) should be an error, not a call that never returns or
//! an allocation that takes the machine down.

use crate::error::{Error, Result};

/// Most loop iterations / time steps / series terms one call may be asked for (about a second for a cheap step).
/// Longer runs: call again from the previous state.
pub const MAX_ITERATIONS: usize = 100_000_000;

/// Most repeated structural units (layers, blocks, experts, table rows) a constructor accepts. Real models stay far below
/// it (the deepest published transformers have a few hundred layers), and each unit costs real memory even when tiny.
pub const MAX_REPEATS: usize = 1 << 16;

/// Most elements a single user-sized buffer may hold (2^31: 8 GiB of `f32`). Larger requests are refused up front;
/// smaller ones that do not fit in memory still fail through the fallible allocation helpers in `layout`.
pub const MAX_ELEMENTS: usize = 1 << 31;

/// Returns `n` if it is at most `max`, otherwise an error that names the argument.
pub fn check_at_most(what: &str, n: usize, max: usize) -> Result<usize> {
    if n > max {
        return Err(Error::Msg(format!(
            "{what} = {n} is above the limit of {max}"
        )));
    }
    Ok(n)
}

/// Returns `n` if it is at most [`MAX_ITERATIONS`], otherwise an error that names the argument.
pub fn check_iterations(what: &str, n: usize) -> Result<usize> {
    check_at_most(what, n, MAX_ITERATIONS)
        .map_err(|e| Error::Msg(format!("{e} per call; split the work into several calls")))
}

/// Returns `n` if it is at most [`MAX_REPEATS`], otherwise an error that names the argument.
pub fn check_repeats(what: &str, n: usize) -> Result<usize> {
    check_at_most(what, n, MAX_REPEATS)
}

/// Element count of a buffer with these `dims`: an error if it overflows `usize` or exceeds [`MAX_ELEMENTS`].
pub fn check_elements(what: &str, dims: &[usize]) -> Result<usize> {
    let n = crate::layout::checked_numel(dims)
        .map_err(|_| Error::Msg(format!("{what}: size {dims:?} overflows")))?;
    check_at_most(what, n, MAX_ELEMENTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_accept_the_boundary_and_reject_beyond() {
        assert_eq!(
            check_iterations("steps", MAX_ITERATIONS).unwrap(),
            MAX_ITERATIONS
        );
        assert!(
            check_iterations("steps", MAX_ITERATIONS + 1)
                .unwrap_err()
                .to_string()
                .contains("steps")
        );
        assert_eq!(check_repeats("depth", 12).unwrap(), 12);
        assert!(check_repeats("depth", MAX_REPEATS + 1).is_err());
        assert!(check_repeats("depth", usize::MAX).is_err());
    }
}
