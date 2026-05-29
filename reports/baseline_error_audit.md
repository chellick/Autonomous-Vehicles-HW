# Baseline Error Audit

This document is the synthesis of seminar 4. It treats the baseline (`models/best_yolo_auto_trasnport.pt`, the YOLOv8 detector trained on the Udacity Self-Driving-Car v3 dataset) as a measured decision system and records its current behavior as a frozen reference. Every claim here is backed by one of the five artifact files in this seminar (`score_and_decision_rule.md`, `threshold_sweep.csv`, `operating_point.md`, `error_analysis.md`, `worst_slice_analysis.md`) plus the existing data audit and stress test from earlier seminars.

## What the baseline ranks well

The model ranks correctly in the high-confidence regime of common classes. At score `>= 0.7` the per-bin accuracy is 0.97 or higher, and at score `>= 0.8` it is essentially 1.0. For the four classes that dominate the dataset by box count (`car`, `pedestrian`, `trafficLight-Red`, `trafficLight-Green`) this means the model is reliable when it is confident. The ranking signal is monotonic — every populated reliability bin shows accuracy that increases with mean confidence — and this is what justifies using `tau` as a cutoff at all.

## Where ranking breaks

Ranking degrades on three axes that the seminar 2 and 3 audits already flagged and that this seminar measures directly.

The first is small-object classes. Pedestrian, biker, and every traffic-light state class have F1 below 0.72 at the framework default and below 0.74 at the chosen operating point. The reliability is not catastrophic but the recall ceiling is structural: the model does not emit boxes for many small-object instances at any threshold. The mechanism (the 1920×1200 to 512×512 stretch resize) is documented in `data_audit.md` section 1.6 and reproduced in section 5 of `error_analysis.md`.

The second is the bare `trafficLight` annotation class. It overlaps semantically with the colored state subclasses by construction (`data_audit.md` section 1.5), and the wrong-class confusion matrix in `error_analysis.md` shows the model reproducing the annotation inconsistency rather than learning a wrong concept. Ranking on this class is not the model's failure but the dataset's; it cannot be improved by training changes alone.

The third is the rare traffic-light states (Yellow, YellowLeft, GreenLeft). At 6, 63, and 76 ground-truth boxes respectively in the test set, the per-class metrics are statistical noise rather than measurements. This is risk R4 from the data audit, confirmed numerically in `worst_slice_analysis.md`.

## Sensitivity to operating point

The baseline is moderately sensitive to `tau` in the operationally interesting range. Across `tau in [0.10, 0.30]` precision moves from 0.67 to 0.85, recall moves from 0.77 to 0.70, and uniform-cost risk has a flat minimum near `tau = 0.14`. F1 has a separate, also flat, minimum at `tau = 0.30`. Outside this range the model degrades quickly: above `tau = 0.50` recall drops below 0.60 and F1 drops below 0.72; below `tau = 0.10` precision drops below 0.65 and the false-positive rate dominates the risk.

The flat plateaus mean the choice of `tau` inside `[0.10, 0.30]` is not very consequential for boundary errors — most of the recoverable trade-off has already happened by the time `tau` hits 0.20. Outside that range the choice matters a lot, and any future change should require an explicit cost-weight justification.

## Where the working operating point is and why

The operating point is `tau = 0.14`, the minimizer of the uniform-cost risk function `R(tau) = c_FP * FP(tau) + c_FN * FN(tau)` with `c_FN = 5, c_FP = 1`. The full justification, including the per-class-cost sensitivity check at `tau = 0.12`, is in `operating_point.md`. The short version is that the deployment scenario assigns higher cost to missed detections than to false alarms (road-scene detection, where downstream consumers can filter spurious boxes but cannot recover missed objects), and the F1-optimal threshold is silent about this asymmetry.

The framework default of `tau = 0.25` from earlier seminars sits just outside the defensible range `[0.10, 0.20]` and is preserved in the project only as a reference for legacy artifacts.

## Boundary errors recoverable by threshold movement

The error analysis identifies the boundary set as the false positives in the `tau = 0.14` to `tau = 0.30` range — about 1,400 extra emissions at the lower end. These are not concentrated on a fixed set of scenes; they are distributed across the test set and represent the model's marginal "speculative" detections. Moving `tau` up trades these away for missed detections at the same rate as the F1 curve predicts. The choice of `tau = 0.14` accepts these as the operational price of the recall floor.

## Errors not recoverable by threshold movement

Three error families are structural and do not move with `tau`:

- Small-object misses on pedestrians, bikers, and small traffic lights — the model emits no box at any threshold for many of these instances.
- Wrong-class confusion in the trafficLight family — the dataset has overlapping class semantics and the model reproduces them.
- Duplicate-bounding-box precision loss in dense car scenes — the ground truth has duplicate annotations and the model is penalized on both.

None of the three is fixable by re-tuning `tau`. The first needs a preprocessing change (preserve aspect ratio at resize, or run inference at higher resolution). The second needs a class-taxonomy revision in the dataset. The third needs the duplicate annotations to be deduplicated.

## Limitations of the baseline visible at this stage

Beyond the per-error-family limitations above, three system-level limitations are visible.

The score is not a calibrated probability. Every populated mid-confidence bin shows accuracy systematically above mean confidence by 14 to 26 points (see `error_analysis.md` calibration section and `reports/figures/reliability_diagram.png`). The model is operationally under-confident — the safer of the two failure modes for a driving detector — but any downstream system that treats `s` as `P(correct)` will be wrong. This is risk R5-equivalent at the score-interpretation level rather than at the class-label level.

The decision rule is set-valued (`y_tau(x)` is a set of boxes, not a label). The seminar's bin-of-binary metric framing (TP/FP/FN/TN) had to be adapted to a per-box matching scheme with scene-level TN, which loses some symmetry. The `accuracy` and `fpr` columns in `threshold_sweep.csv` use scene-level TN for completeness, but they are less stable indicators than precision/recall for a detection task and should not be the criterion for any further operating-point decision.

The baseline numbers were measured under the group-split protocol that closes the twin-frame leakage from the data audit. The image-split numbers (which have 74.3% twin-frame overlap with train) would be different and are not the canonical reference; this is documented in `data_audit.md` section 6.

## Where to act next

The actions visible from this audit split cleanly into three groups.

Data-side actions that this baseline cannot improve from: collect more examples of the rare traffic-light states (R4 closure), de-duplicate the ground-truth boxes in the dataset (R6 closure), and resolve the `trafficLight` versus `trafficLight-<state>` overlap by either removing the bare class or annotating the state on every traffic-light instance (R5 closure). All three require dataset modifications outside the model.

Model-side actions: change the input pipeline to preserve aspect ratio at the YOLO resize, or train a variant at higher input resolution; both directly attack the recall ceiling on small-object classes that this audit identifies as the dominant structural failure. A calibration step (temperature scaling on a held-out set) would make the score interpretable as a probability and fix the under-confidence pattern, but does not change ranking or detection counts.

Decision-policy actions: use the `tau = 0.14` operating point in any future evaluation as the canonical reference, and treat the `[0.10, 0.20]` band as the allowed range for any application-specific re-tuning. Any threshold choice outside this range needs its own explicit cost-weight justification.

This audit does not propose any of these as concrete next steps for this seminar. It records the baseline as it is. Subsequent course iterations will use this record as the basis for any modification of data, model, or decision policy.
