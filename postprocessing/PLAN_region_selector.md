# Oracle-supervised region selector v1

Approved scope: CPU prototype trained on oracle labels for 64 candidates from 13 best-checkpoint validation patients. Volumes, normalized RAS positions and voxel probability statistics are inputs. No annotation-derived metric or patient ID may be a feature.

1. Generate oracle targets again on validation_best_probabilities, not on final-checkpoint outputs. Verify source hashes and component IDs against probability statistics.
2. Fixed model: StandardScaler + L2 LogisticRegression, C=0.1, balanced classes, lbfgs, max_iter=2000, seed=12345. Equal base patient weight before sklearn class balancing (effective totals vary after class weighting). No tuning against this experiment's validation results.
3. Leave-one-patient-out: fit scaler/classifier only on the 12 other patients; rank the held-out patient's components; keep one entire maximum-score region. Training labels for the held-out patient must never reach fit. Empty predictions remain empty.
4. Export OOF masks separately, record each fit's train/test patient IDs and selected component scores; evaluate with the existing metrics. Compare to raw and oracle best on identical patients. This is development validation, not an independent test because the backbone checkpoint was selected on these 13 patients.
5. Save one final fitted model trained on all 13 patients for future patients, never use its resubstitution scores as validation results. Provide separate inference entry point with no oracle or reference-label input and refusal of known training patients by default.
6. Test feature isolation, axis-normalized positions, patient separation, deterministic predictions and integrity of exported masks. Perform a read-only review before reporting conclusions.

Outputs outside Git: Datasets/nnUNet/experiments/region_selector_v1. Code stays under postprocessing. No nnU-Net retraining or test-set inference.
