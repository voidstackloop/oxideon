//! Reynolds-O'Hearn Separation Logic, Separating Conjunction $P * Q$ & Local Frame Rule (O'Hearn, Reynolds & Yang, 2001; Reynolds, 2002).
//!
//! ### 1. Disjoint Heaplet Union $h = h_1 \uplus h_2$ & Frame Rule Soundness
//! In Separation Logic, $h \models P * Q$ holds iff $h$ splits into domain-disjoint subheaps $h_1 \uplus h_2$ ($\mathrm{dom}(h_1) \cap \mathrm{dom}(h_2) = \emptyset$)
//! with $h_1 \models P$ and $h_2 \models Q$, enabling modular local reasoning via $\frac{\{P\} C \{Q\}}{\{P * R\} C \{Q * R\}}$ when $\mathrm{mod}(C) \cap \mathrm{fv}(R) = \emptyset$.

use crate::error::Result;

/// Reynolds-O'Hearn Separation Logic Disjoint Heaplet & Frame Rule Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReynoldsSeparationLogicHeap;

impl ReynoldsSeparationLogicHeap {
    /// Creates a new Reynolds Separation Logic Heap engine.
    pub fn new() -> Self {
        Self
    }

    /// Checks whether two address-domain bitmasks $D_1, D_2$ are disjoint ($D_1 \cap D_2 = \emptyset$) and whether frame $D_R$ is untouched by command footprint $M_C$, returning $(D_1 \cup D_2, \text{valid})$.
    pub fn check_separating_conjunction_and_frame(
        dom1_mask: u64,
        dom2_mask: u64,
        frame_mask_r: u64,
        modified_mask_c: u64,
    ) -> Result<(u64, bool, bool)> {
        let disjoint_heaps = (dom1_mask & dom2_mask) == 0;
        let frame_preserved = (frame_mask_r & modified_mask_c) == 0;
        let combined = dom1_mask | dom2_mask | frame_mask_r;
        Ok((combined, disjoint_heaps, disjoint_heaps && frame_preserved))
    }
}
