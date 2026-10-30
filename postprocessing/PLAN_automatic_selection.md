# CT-guided automatic selection v1

Approved scope: rank existing predicted components using the original CT and predicted volume; never read ground truth during selection. Evaluate only on the existing 13 validation patients, final checkpoint. Do not process the 32 test patients in this experiment.

Implementation plan:
1. Test lung-air extraction on synthetic CT, physical spacing, correct selection of a small nearby candidate, ambiguity and unreliable-ROI fallback.
2. Estimate a coarse pulmonary ROI using enclosed low-HU air inside slice-wise body masks. Fill holes, retain up to two dominant air components, verify volume plausibility.
3. Score whole 26-connected prediction components by fraction within a 10 mm margin, fraction in the ROI, and relative volume. Preserve all input components if the estimate is unreliable, no candidate is credible, or scores are ambiguous.
4. Write masks to automatic_ct_v1 with immutable input hashes and decision manifest. Preserve original NIfTI geometry. Configuration is frozen in automatic_selection_v1.json before evaluation.
5. Reuse evaluation and paired comparisons against raw, largest-component and oracle outputs. Report failures, not only mean gains. No tuning on the 13 results in this version.

This ROI is a heuristic, not a validated anatomical segmentation. Even a unique score can select the wrong component. This version cannot refine boundaries or recover a missed target. Config choices are experimental developer defaults approved as a first test, not clinical thresholds.
