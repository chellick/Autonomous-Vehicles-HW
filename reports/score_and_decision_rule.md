# Score and Decision Rule of the Baseline

This document fixes what the baseline detector outputs as a numerical signal, what discrete decision is taken from that signal, and what threshold is used by default. Everything in the rest of the seminar's artifacts (threshold sweep, operating point, error analysis, calibration) refers back to this definition.

## What the model outputs

The baseline is a YOLOv8 detector (`models/best_yolo_auto_trasnport.pt`, 73 layers, ~3 million parameters, 8.1 GFLOPs) trained on 11 classes of road objects. For an input RGB image at 640×640 pixels (after Ultralytics' resize), the model's raw output is a tensor of shape `(1, 4 + 11, A)` where `A` is the number of anchor positions across the three detection heads. The first four channels per anchor are the bounding-box coordinates after the box-head transform; the remaining eleven are per-class scores.

The class scores reach the user as sigmoid-activated values in the open interval (0, 1). For each anchor position `i`, the per-class score is

$$ s_{i,c} = \sigma(z_{i,c}) $$

where `z_{i,c}` is the raw logit produced by the classification head for class `c` at that anchor. This score is what Ultralytics calls the "confidence" of a detection. There is no separate objectness score in YOLOv8 — the maximum class score per anchor functions as both objectness and class probability.

The score is not a calibrated probability. Section 6 of `data_audit.md` and section 5 of `stress_test.md` both showed that the model is systematically under-confident in the operationally relevant confidence range. The score should be read as a learned ranking signal that monotonically tracks correctness on average but does not interpret as `P(correct | s)`.

## Decision rule

Two thresholds turn the per-anchor scores into a discrete prediction. The first is the per-anchor cutoff `tau`: an anchor `i` survives if `max_c s_{i,c} >= tau`. The second is non-maximum suppression (NMS): for each class `c`, surviving boxes are sorted by score and pruned if they overlap a higher-scoring box of the same class with IoU above `iou_nms` (the baseline uses `iou_nms = 0.6`). The output is the set of surviving (box, class, score) triples.

The decision rule, written in the form the seminar requires, is

$$ \hat y_\tau(x) = \{ (b_i, c_i, s_i) \;:\; s_i \ge \tau \;\wedge\; \text{NMS}(b_i, c_i, s_i, \text{iou\_nms}) \} $$

This is a per-image set-valued decision, not a single label, because object detection produces zero or more boxes per image. When this document refers to a single-class binary view (true positive, false positive, etc.) for the threshold sweep and risk function, it means the per-box decision: a candidate detection is "emitted" if its score is at least `tau` and it survives NMS, "suppressed" otherwise.

The matching that turns these emitted detections into TP/FP/FN counts is fixed at IoU >= 0.5 against ground truth, with the additional class-correctness requirement encoded in `src/box_metrics.py`. A correctly-localized box with the wrong class is recorded separately as `wrong_class`; for cost and metric accounting it counts as both a missed ground-truth box (FN for the GT class) and a spurious emission (FP for the predicted class).

## Default threshold

The baseline runs with `tau = 0.25`. This value is the Ultralytics framework default and was used unchanged in every training, validation, and inference command in this project. It was not chosen by an explicit engineering criterion; it is the framework's out-of-the-box value for the `conf` argument of `model.predict()` and `model.val()`.

This means `tau = 0.25` is a technical default, not an engineered operating point. Section "Operating point" in `operating_point.md` records the explicit choice that replaces it after the threshold sweep. The default is documented here only so that any earlier metric reported with `tau = 0.25` (the data audit, the seminar 3 stress test, the existing `runs/` directory) can be located on the threshold curve in `reports/threshold_sweep.csv`.

## What "positive" and "negative" mean for this task

Object detection does not have a clean true-negative concept. The framing used throughout the seminar 4 artifacts is:

- True positive: an emitted box matches a ground-truth box at IoU >= 0.5 with the same class.
- False positive: an emitted box has no ground-truth match at IoU >= 0.5, or matches with a different class. (Class mismatches are the `wrong_class` row in `box_metrics.py`.)
- False negative: a ground-truth box has no matching emitted box at IoU >= 0.5 with the correct class.
- True negative: a scene that has no ground-truth boxes and on which the model emits no boxes at the chosen threshold. This is a scene-level rather than box-level concept.

The TN scene count is tracked in the threshold sweep CSV but it is not symmetric with the other three counts: TP/FP/FN are per-box, TN is per-scene. This asymmetry is unavoidable for object detection and is the reason why the seminar's `accuracy` and `fpr` columns use the scene-level TN; precision, recall, and F1 do not depend on TN and are unaffected.
