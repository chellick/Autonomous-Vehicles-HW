# Worst-Slice Analysis

This document examines whether the baseline's quality is uniform across the data or whether some subdomains carry disproportionate risk. The analysis runs at the default operating point `tau = 0.25` on the full 2,971-frame group-split test set, using the matched ground truth from `runs/20260426_135344_groupsplit_test/`. Numbers are sourced from `reports/slice_metrics.csv`, produced by `src/slice_metrics.py`.

## Slice definitions

The slices come in two families. Class slices group every emitted-or-missed box by its ground-truth class label; the support column is the number of ground-truth boxes of that class in the test set. Scene slices group the 2,971 test frames by properties of the scene as a whole; the support column is the number of frames in the slice.

The class slices are the eleven training classes. The scene slices are the six engineering subdomains carried over from `data_audit.md` section 2.6 (`empty`, `dense_car` for at least five cars, `tl_state` for any active traffic-light class, `rare_tl` for the Yellow / GreenLeft / YellowLeft classes, `ped_biker`, `truck`) plus three additional ones added specifically for this analysis: `size_small`, `size_medium`, `size_large` partition frames by mean ground-truth box area as a fraction of the frame (cutoffs at 0.5% and 5% of frame area), and `count_sparse`, `count_medium`, `count_dense` partition frames by ground-truth box count (cutoffs at 2 and 8). These three families let us test directly whether failure correlates with object size and scene density rather than with object semantics.

A frame can belong to multiple scene slices; a ground-truth box can belong to only one class slice. Slices were chosen because each one is something the deployment scenario is likely to ask about ("is the model good at small objects" or "does it handle dense traffic"), and because they cover the three most plausible failure axes — object class, object size, and scene density — without inventing a slice that the data cannot define.

## Results

The table below lists every slice ranked by F1 ascending. Slices with an undefined metric (empty scenes have no ground-truth boxes, so precision and recall are 0/0) are reported as F1 = 0 and excluded from the worst-slice ranking.

| Slice | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| scene:empty | 381 | — | — | — |
| class:trafficLight-YellowLeft | 6 | 0.000 | 0.000 | 0.000 |
| class:trafficLight-Yellow | 63 | 0.587 | 0.429 | 0.495 |
| class:trafficLight-GreenLeft | 76 | 0.623 | 0.500 | 0.555 |
| class:pedestrian | 2,205 | 0.770 | 0.502 | 0.608 |
| class:biker | 388 | 0.822 | 0.513 | 0.632 |
| class:trafficLight-Green | 1,081 | 0.780 | 0.623 | 0.693 |
| class:truck | 756 | 0.740 | 0.660 | 0.698 |
| scene:size_small | 969 | 0.796 | 0.640 | 0.709 |
| scene:rare_tl | 84 | 0.760 | 0.665 | 0.710 |
| class:trafficLight | 410 | 0.731 | 0.710 | 0.720 |
| class:trafficLight-RedLeft | 325 | 0.765 | 0.702 | 0.732 |
| class:trafficLight-Red | 1,278 | 0.821 | 0.676 | 0.742 |
| scene:ped_biker | 833 | 0.825 | 0.675 | 0.742 |
| scene:count_dense | 893 | 0.827 | 0.686 | 0.750 |
| scene:tl_state | 844 | 0.818 | 0.694 | 0.751 |
| scene:truck | 533 | 0.808 | 0.722 | 0.762 |
| scene:dense_car | 1,258 | 0.830 | 0.712 | 0.766 |
| scene:count_medium | 1,310 | 0.819 | 0.765 | 0.791 |

(The full table including the better-performing slices is in `reports/slice_metrics.csv`.)

## What the worst slices share

The bottom of the ranking is dominated by per-class slices, not by per-scene slices. The four lowest non-trivial F1 values are all single-class — the four rarest or smallest-object classes (the three rare traffic-light states plus `pedestrian`). The lowest scene-level slice (`size_small`) ranks below the worst-class scene level but above the worst per-class entries. This means the worst-case behavior of the baseline is captured better by asking "which class" than by asking "which scene"; per-class operational risk is the right framing.

Two distinct mechanisms produce the worst slices.

The `trafficLight-YellowLeft` row is statistical noise. With six ground-truth boxes in the entire test set, the F1 of 0.000 means the model missed four of them and misclassified the other two; any single-image change would shift the metric by 0.16. Risk R4 in the data audit explicitly recorded that per-class metrics on classes with this much support are not measurable, and this row confirms that. The `trafficLight-Yellow` and `trafficLight-GreenLeft` rows have the same problem at slightly larger scale (63 and 76 ground-truth boxes); their F1 of around 0.5 is more meaningful but should still not be used to compare model variants without bootstrap confidence intervals.

The `pedestrian` and `biker` rows are real signals. With 2,205 and 388 ground-truth boxes respectively, their low recall (0.50 and 0.51) is statistically meaningful and matches the structural finding from the data audit's section 5: pedestrians and bikers are tall narrow objects, and the 1920×1200 to 512×512 stretch resize compresses vertical extent more than horizontal, so these classes lose the most discriminative pixels before the model sees them. The `size_small` scene slice (F1 0.709) confirms the same pattern at the scene level: frames where the average box covers less than 0.5% of the frame area are systematically harder for the model to detect.

## Is the worst slice a threshold problem or a structural one

The threshold-sensitivity question for each slice is whether moving `tau` redistributes the slice's errors in a useful way. For the small-object classes the answer is no. Their failure mode is missed detections (recall 0.50 on pedestrian, 0.51 on biker, 0.43 on Yellow), and lowering `tau` cannot create boxes that the model never produced. The score histogram for these classes is shifted left: the model is not just under-confident on small objects, it does not see them at all in many cases. This is consistent with what the operating-point sensitivity analysis in `stress_test.md` section 4 showed under blur shift, where recall could not be recovered by re-tuning `tau`.

For the rare traffic-light classes the operating-point question does not apply for a different reason: the support is too small for any threshold choice to be statistically defensible. Re-tuning `tau` on six ground-truth examples is hyperparameter overfitting to the test set.

For the larger-object classes that rank in the middle of the ranking (`trafficLight-Red`, `truck`, `trafficLight-Green`), the precision is consistently higher than the recall, which suggests there is some room for a lower-threshold operating point if recall is the priority on those classes. The exact trade-off is in `reports/threshold_sweep.csv` and the global decision is recorded in `reports/operating_point.md`.

The conclusion is that the baseline is not uniformly degraded across slices — it is structurally weak on small objects (pedestrian, biker, all traffic-light states) and effectively unmeasured on rare traffic-light states. Neither of these is a global operating-point problem. The first is fixable only by changing the input pipeline (preserving aspect ratio at resize, or running the detector at higher resolution); the second is fixable only by collecting more data for the rare states. Both are recorded as open data-side actions in the synthesis at the end of `baseline_error_audit.md`.
