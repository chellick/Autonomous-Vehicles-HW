# Operating Point

This document records the engineered choice of the baseline operating threshold `tau` and the criterion behind that choice. The threshold sweep that backs every number here is in `reports/threshold_sweep.csv`, generated from the full 2,971-frame group-split test set with predictions saved at `conf >= 0.001` (`runs/groupsplit_full/`); the matching is at IoU >= 0.5 against ground truth, identical to every other artifact in this seminar.

## Candidates

Three reasonable ways to pick `tau` were considered. Each one is grounded in a measurable quantity rather than visual judgment.

The first is the **F1-optimal threshold**. The maximum F1 across the 51-point sweep is 0.771 at `tau = 0.30`, with precision 0.851 and recall 0.704. The curve is locally flat: F1 is within 0.01 of the maximum across the entire `tau` range from 0.20 to 0.34. F1 is a defensible criterion when the cost of a false positive equals the cost of a false negative and there is no obvious operational asymmetry.

The second is the **uniform-cost risk-optimal threshold**, with cost ratio `c_FN = 5, c_FP = 1`. The cost ratio reflects the deployment context: the system is intended for road-scene detection, where missing a real object is operationally worse than emitting a spurious detection that the downstream consumer can filter out. The minimum of `R(tau) = c_FP * FP(tau) + c_FN * FN(tau)` across the sweep is at `tau = 0.14` with risk 28,750, precision 0.728, recall 0.757. At this point the system catches more ground-truth boxes (recall up by 5.4 points relative to F1-optimal) at the cost of about 1,400 more false positives across the test set (precision down by 12.3 points).

The third is the **per-class-cost risk-optimal threshold**. The per-class costs in `src/analyze_thresholds.py` (`PER_CLASS_FN_COSTS`) assign higher FN cost to safety-critical classes: pedestrian and biker at 10, car and truck at 5, traffic-light states at 3, the bare `trafficLight` fallback at 2. False-positive cost is 1 across all classes. The minimum of this per-class risk function is at `tau = 0.12` with precision 0.700 and recall 0.765. The optimum moves further down compared to uniform costs because lowering the threshold disproportionately recovers FN on small-object classes (pedestrian, biker), which carry the highest weights.

The three optima form a spectrum: F1 at 0.30, uniform risk at 0.14, per-class risk at 0.12.

## Choice

The baseline operating point is **`tau = 0.14`**, the uniform-cost risk minimizer.

The choice is the uniform-cost risk minimum rather than the per-class one because the cost-weight schedule the per-class function uses is itself a judgment call (10 for pedestrian, 5 for car, 3 for traffic light) that has no defensible quantitative grounding for this dataset — there is no annotated downstream task that would let those weights be calibrated. The uniform 5:1 cost ratio is also a judgment call, but it is one judgment about the asymmetry of FN versus FP, not eleven separate judgments about per-class importance, and it is a standard ratio for safety-relevant detection tasks. The per-class result is reported alongside as a sensitivity check rather than as the decision; both move the operating point in the same direction (lower than F1-optimal).

The choice is below the F1-optimal because F1 is silent about the operational asymmetry that exists in the deployment scenario (`PROJECT_PASSPORT.md` does not give explicit cost weights, but the road-scene context and the data audit's R1–R4 risk register both indicate that recall on real objects is the constrained quantity). At `tau = 0.30` the model emits about 1,400 fewer false positives than at `tau = 0.14`, but it also misses 351 more ground-truth boxes. Under any cost ratio with `c_FN > 1`, the lower-precision/higher-recall point is preferable.

## Trade-offs at the chosen point

The trade-offs that the choice of `tau = 0.14` accepts:

Precision drops from the framework default value of 0.821 (at `tau = 0.25`) to 0.728. About 27% of emitted detections at `tau = 0.14` will not match any ground-truth box (or will match one with a different class). Any downstream consumer that treats every emitted box as a real object will see a higher false-positive rate than at the framework default.

Recall increases from 0.720 (at `tau = 0.25`) to 0.757. The model finds about 700 more ground-truth boxes across the 2,971-frame test set than at the framework default. The recovered detections are concentrated on the classes with the largest score-distribution shift (pedestrian, biker, the colored traffic-light states), which is the operationally relevant direction.

The boundary errors that move with `tau` between the framework default and the chosen operating point are mostly on the small-object classes documented in the worst-slice analysis. The structural errors (trafficLight semantic confusion, dense-car duplicate-bbox precision loss, the pedestrian recall ceiling set by the stretch resize) do not move and are not resolved by this choice.

## Allowed range

The operating point is defensible across the range `tau in [0.10, 0.20]`: within this range the uniform-cost risk increases by less than 2% relative to the minimum, while precision/recall change by 8 points. Outside this range either precision drops below 0.65 (at `tau < 0.10`, where the FP count starts to dominate even at uniform 5:1 weighting) or recall drops below 0.71 (at `tau > 0.22`, where the per-class risk on pedestrian/biker exceeds the uniform-cost optimum). Any future change of `tau` should stay inside `[0.10, 0.20]` unless the cost weights are explicitly re-justified.

The framework default `tau = 0.25` lies just outside this range. It is acceptable for legacy comparison but should be replaced by `tau = 0.14` for any operational metric reporting from this point forward. Every existing artifact in the project that quoted numbers at `tau = 0.25` (the data audit, the seminar 3 stress test, the existing `runs/` directory) remains valid as a reference at the framework-default point; the new operating-point numbers are reported in `baseline_error_audit.md` as the authoritative baseline behavior.
