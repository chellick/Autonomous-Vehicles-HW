# Stress Testing and Robustness Audit

This document records the functional behavior of the baseline detector (`best_yolo_auto_trasnport.pt`) under a fixed set of input perturbations. The purpose is to turn the abstract notions of invariance, robustness, and distribution shift into measurable properties of the system. After publication, this document is treated as the reference behavior of the model: any subsequent change to the model or the data is evaluated against the curves and worst-slice numbers below.

All measurements use a deterministic 100-image subsample of the group-split test set (`splits/self_driving_car/test.txt`, sample seed 42), matched against the ground-truth records from `runs/20260426_135344_groupsplit_test/gt.json`. Box matching is done at IoU ≥ 0.5; the default operating threshold is conf = 0.25.

---

## 1. Perturbation set

The space of inputs is the set of RGB images at the original 512×512 resolution. Every perturbation `T_alpha` is a deterministic function of one parameter and acts on a single image, with no interaction between images. The implementation is in `src/transforms.py`. Seven families are evaluated:

| Family | Parameter | Grid | Identity |
| --- | --- | --- | --- |
| brightness | exposure value (EV) | -0.5, -0.25, 0, +0.25, +0.5 | 0 |
| contrast | multiplicative factor | 0.5, 0.75, 1.0, 1.25, 1.5 | 1.0 |
| gamma | exponent | 0.5, 0.75, 1.0, 1.5, 2.0 | 1.0 |
| noise | Gaussian sigma on uint8 | 0, 5, 10, 20, 40 | 0 |
| blur | Gaussian radius (px) | 0, 1, 2, 4, 8 | 0 |
| jpeg | quality | 10, 20, 40, 60, 80, 100 | 100 |
| occlusion | masked area as % of frame | 0, 5, 10, 20, 30 | 0 |

Brightness within +-0.25 EV, contrast in [0.75, 1.25], gamma in [0.75, 1.5], and JPEG quality at least 80 are declared as required invariances: the system is expected to deliver essentially unchanged detections inside these ranges. Outside them, and for every value of noise, blur, and occlusion, degradation is acceptable but should be smooth and predictable rather than catastrophic. The declared invariance bands are encoded in `INVARIANT_RANGES` in the same module.

The choice to test image-level perturbations only (no temporal effects, no multi-frame artifacts) follows from the application scenario in the data audit, which treats requests as independent static frames.

---

## 2. Degradation curves

For each transform, F1 against the matched ground truth as a function of alpha:

| Transform | Nominal F1 | F1 at extreme | Drop |
| --- | ---: | ---: | ---: |
| brightness | 0.758 (alpha=0) | 0.736 (alpha=+0.5) | 3% |
| contrast | 0.758 (1.0) | 0.700 (1.5) | 8% |
| gamma | 0.758 (1.0) | 0.722 (2.0) | 5% |
| noise | 0.758 (sigma=0) | 0.254 (sigma=40) | 66% |
| blur | 0.758 (radius=0) | 0.061 (radius=8) | 92% |
| jpeg | 0.758 (q=100) | 0.571 (q=10) | 25% |
| occlusion | 0.758 (0%) | 0.352 (30%) | 54% |

The four invariance families behave as expected. Brightness and gamma curves stay within 5% of nominal across the full grid; contrast and JPEG fall inside the declared invariance band but degrade outside it (contrast 1.5 loses 8%, JPEG q=20 loses 8%, JPEG q=10 loses 25%). Whether the q=10 figure constitutes a real failure depends on whether such heavy compression appears in deployment, which is unknown for this dataset (open risk R2).

The three remaining families show structural failure modes. Blur is the most fragile: at radius 4 the model still recovers ~43% F1, but at radius 8 it collapses to 6% — the recall drops to 3.2%, meaning the model essentially stops detecting anything. This is consistent with what the data audit predicted from subdomain S9 (the stretch resize already costs sharp-edge information; further blur removes whatever pixel evidence small objects had). Noise has a similar but less severe profile: F1 stays above 0.7 up to sigma=10, then collapses through 0.59 at sigma=20 and 0.25 at sigma=40. Recall is what dies first in both cases — precision actually increases slightly under heavy noise (0.81 → 0.85), because the few detections the model still emits are the most confident ones. Occlusion at 30% area is the cleanest case: the model loses recall in proportion to the masked area until the threshold is reached, then drops sharply.

The mechanism is the same in all three failures: a threshold effect on recall, with precision either stable or rising as the model becomes more conservative. False positives do not explode under any of the perturbations tested — the failure mode is missed detections, not hallucination. This matches the baseline error analysis from the data audit (section 5), where misses already outnumber false positives 2:1 under the nominal distribution.

---

## 3. Worst-slice analysis

The 100-image subsample is decomposed into the subdomains defined in the data audit (section 2.6). Each scene can belong to multiple slices. F1 is reported at conf = 0.25.

Under the nominal distribution:

| Slice | Scenes | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| pedestrian / biker | 25 | 0.831 | 0.703 | 0.762 |
| traffic light state | 32 | 0.841 | 0.695 | 0.761 |
| truck | 19 | 0.803 | 0.702 | 0.749 |
| dense car (≥5) | 41 | 0.804 | 0.695 | 0.746 |
| rare traffic light | 4 | 0.822 | 0.661 | 0.733 |
| empty (no GT) | 13 | — | — | — |

Under the chosen stress scenario (blur radius 4, the steepest part of the degradation curve before total collapse):

| Slice | Scenes | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| dense car (≥5) | 41 | 0.733 | 0.323 | 0.448 |
| truck | 19 | 0.713 | 0.288 | 0.410 |
| rare traffic light | 4 | 0.875 | 0.250 | 0.389 |
| pedestrian / biker | 25 | 0.663 | 0.263 | 0.377 |
| traffic light state | 32 | 0.724 | 0.239 | 0.360 |

Under the nominal distribution the slices are almost flat, all between F1 0.73 and 0.76. The empty-scene slice cannot be measured by precision/recall (zero ground-truth boxes means the metric is undefined) — the meaningful number for it is the false-positive rate on background, which is reported in the cascade section.

Under blur shift the slice ordering changes. The pedestrian/biker and traffic-light-state slices, which involve the smallest objects in the frame, lose the most: F1 drops by half on traffic lights and by more than half on pedestrians/bikers. Dense car scenes degrade least in relative terms, because cars are the largest objects in the dataset and survive the resolution loss longer. The rare-TL slice shows F1 0.389 but the support is only 4 scenes, which is too small to read as an actual signal.

The conclusion is consistent across the data audit and the stress test: small-object slices (pedestrian, biker, traffic-light state) carry the most operational risk, both at nominal conditions (worst-class miss rates in section 5 of the data audit) and under input degradation (steepest F1 drop here).

---

## 4. Operating point sensitivity

Confidence threshold tau swept over {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70} with re-matching at each tau:

| tau | Nominal F1 | Shift F1 (blur=4) |
| ---: | ---: | ---: |
| 0.05 | 0.596 | 0.366 |
| 0.10 | 0.680 | 0.414 |
| 0.15 | 0.717 | 0.427 |
| 0.20 | 0.742 | 0.425 |
| 0.25 | 0.758 | 0.429 |
| 0.30 | 0.758 | 0.420 |
| 0.40 | 0.744 | 0.378 |
| 0.50 | 0.702 | 0.309 |
| 0.60 | 0.635 | 0.262 |
| 0.70 | 0.493 | 0.195 |

Nominal F1 is maximized at tau in [0.25, 0.30], with the curve essentially flat across that range. Under the blur shift the F1-optimal tau moves slightly down to 0.25 — the model emits fewer high-confidence detections under blur, so a lower acceptance threshold is needed to keep recall up. Solving the min-max explicitly gives a robust tau of 0.25 with min-F1 = 0.429. This is the same value as the per-distribution F1-optimal under shift, which means the robust point is dominated by the shift's behavior rather than by a real trade-off.

The operational reading is that the default conf = 0.25 already sits at the robust point; there is no benefit from re-tuning the threshold for the kinds of shifts measured here. What does change between distributions is the precision/recall split at the robust point: precision drops from 0.81 to 0.70 and recall drops from 0.71 to 0.31. Threshold tuning cannot recover the lost recall — that loss is structural, caused by the model not emitting boxes at all.

---

## 5. Calibration under shift

Reliability is computed in 10 equal-width confidence bins over every prediction (low-confidence predictions included). Aggregate ECE is 0.017 nominal and 0.006 under the blur shift, both very low — but the aggregate is dominated by the [0.0, 0.1] bin which contains 6,399 predictions (essentially all the noise rejections of the model's first stage), and it is well-calibrated by construction (mean conf 0.010 against accuracy 0.011).

The mid- to high-confidence bins tell a different story:

| Confidence bin | Count | Mean conf | Accuracy | Gap |
| --- | ---: | ---: | ---: | ---: |
| 0.2 – 0.3 | 90 | 0.244 | 0.267 | +0.023 |
| 0.3 – 0.4 | 69 | 0.346 | 0.493 | +0.146 |
| 0.4 – 0.5 | 80 | 0.454 | 0.650 | +0.196 |
| 0.5 – 0.6 | 66 | 0.548 | 0.864 | +0.315 |
| 0.6 – 0.7 | 110 | 0.652 | 0.891 | +0.239 |
| 0.7 – 0.8 | 130 | 0.748 | 0.969 | +0.222 |
| 0.8 – 0.9 | 87 | 0.842 | 1.000 | +0.158 |
| 0.9 – 1.0 | 6 | 0.913 | 1.000 | +0.087 |

The gap is positive in every populated bin: the model's confidence systematically under-states its accuracy by 10 to 30 points across the actionable range. A box predicted at confidence 0.55 is correct 86% of the time. This is a calibration problem in the operationally useful direction — the model is conservative rather than overconfident — but it does mean that confidence values cannot be interpreted as probabilities. Aggregate ECE hides this because most of the probability mass sits in the low-confidence reject bin.

Under the blur shift the high-confidence bins largely empty out (the model stops emitting confident detections), which is why the aggregate ECE drops further. The under-confidence pattern in the bins that survive is unchanged. This means the calibration story is essentially the same nominal and under shift: the model is consistently under-confident at its actionable operating range, and the shift does not flip the direction of the bias.

---

## 6. Local robustness and cascade

Local robustness is measured along two axes: random noise sensitivity and adversarial (gradient-aligned) sensitivity. Random noise comes from `src/adversarial.py` on a 30-image subsample (zero-mean Gaussian of varying sigma, three trials per image, sigma); adversarial sensitivity comes from `src/gradient_analysis.py` on a 50-image subsample (L_inf-bounded FGSM perturbations on the input image, single pass per epsilon). Both runs use the same matched re-detection criterion: count how many of the original (clean) detections survive in the perturbed prediction, IoU ≥ 0.5 and class-equal.

Random Gaussian noise preservation rate by sigma: 1.00 at 0, 0.98 at 1, 0.99 at 2, 0.95 at 4, 0.89 at 8, 0.67 at 16. Median sigma to flip at least one clean detection is 4.0 (IQR 2.0 – 8.0), within the [0.5, 16] grid; 70% of images flipped inside the grid.

The gradient-aligned picture is sharper. The objective for the gradient computation is the sum of the top-10 max-class confidence scores across anchors of the raw YOLO output tensor, before NMS (`confidence_objective` in `src/gradient_analysis.py`). The model is put in eval mode but with gradient flow enabled on the input tensor, and the input gradient is read after a single backward pass.

The input gradient norms are small in absolute terms but concentrated. Per-image median ‖∇x L‖_2 is 1.44 (IQR 1.00 – 4.75) and median ‖∇x L‖_∞ is 0.10, on input tensors normalized to [0, 1]. The mean per-pixel L1 gradient magnitude is 3.9e-4 — meaning a typical pixel barely matters, but there is enough concentration on the boundary-relevant pixels to change decisions with a small epsilon.

FGSM-style perturbation `x_adv = clip(x − ε · sign(∇x L), 0, 1)` confirms this. On the 50-image subsample (42 with at least one clean detection):

| Epsilon | Equivalent in 8-bit | Detection preservation rate |
| ---: | ---: | ---: |
| 1/255 | 1 pixel value | 0.746 |
| 2/255 | 2 pixel values | 0.642 |
| 4/255 | 4 pixel values | 0.536 |
| 8/255 | 8 pixel values | 0.419 |
| 16/255 | 16 pixel values | 0.185 |
| 32/255 | 32 pixel values | 0.036 |

100% of images with clean detections lost at least one detection within the [1/255, 32/255] range; the median flip threshold is 1/255 (IQR 1/255 – 2/255). At ε = 1/255, an L_inf perturbation indistinguishable to the human eye, roughly a quarter of the model's detections vanish. The same image under random Gaussian noise of comparable per-pixel magnitude (sigma = 1 on uint8 ≈ 0.004 stddev in [0, 1]) preserved 98% of detections — random noise of the same magnitude is essentially harmless, but the same-magnitude gradient-aligned perturbation removes a quarter of detections. The gap is roughly an order of magnitude in effectiveness, which is the standard adversarial-vs-random gap reported in the literature.

This means local stability claims for this model cannot be made on the basis of the noise sensitivity alone. The model is empirically robust to incidental pixel noise within sigma ≤ 4, but it is fragile to single-pixel-magnitude adversarial perturbations. For the deployment scenario described in the data audit (batch RGB analysis without an adversary), this is acceptable but should be recorded as an open assumption: the conclusion does not extend to scenarios where input pixels can be chosen by an adversary or where lossy intermediate processing happens to align with the model's gradient direction.

YOLO is single-stage but its output decomposes naturally into two conditional steps: localization (does the model produce a box with IoU ≥ 0.5 against a ground-truth box) and classification (given the box is localized, is the class label correct). Cascade decomposition on the same data:

| | Nominal | Blur radius 4 |
| --- | ---: | ---: |
| P(localized) | 0.733 | 0.318 |
| P(class correct \| localized) | 0.971 | 0.972 |
| Spurious detections per image | 0.97 | 0.82 |

The split is sharp. Under nominal conditions roughly three quarters of the ground-truth boxes are localized, and almost all the localized ones get the right class. Under blur shift the localization rate collapses to a third, but the conditional class accuracy is unchanged. In other words, blur breaks the detection step almost completely while leaving the classification step intact. This is what the degradation curve in section 2 was actually measuring under the hood — recall collapses, precision-conditional-on-detection is stable.

The operational implication is that any future improvement aimed at blur robustness needs to target the detection backbone, not the classification head. Calibration (section 5) and class-confusion fixes (data audit section 5) are orthogonal axes; they do not address the main fragility surfaced by this audit, which is the loss of the localization signal under blur, heavy noise, and large occlusion.
