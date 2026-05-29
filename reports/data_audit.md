# Data and Experimental Protocol Audit

This document fixes the data and experimental protocol of the road object detection baseline as engineering artifacts. After this document is published, any subsequent change to the data or the protocol is treated as a change to the problem statement itself.

---

## 1. Dataset origin

### 1.1. Source and acquisition

The project uses the Udacity Self Driving Car dataset, version v3 fixed-small, distributed through Roboflow Universe (roboflow-gw7yv/self-driving-car, version 3). Download is performed by [data/download.py](../data/download.py) through the Roboflow API in YOLOv8 format. The base annotations come from crowdsourcing in the open Udacity project and were later re-checked and extended by the Roboflow team: the original set had systematic missing annotations for pedestrians, bikers, cars, and traffic lights, and these were fixed manually. License is MIT; export date is January 13, 2023.

### 1.2. Capture conditions

All images come from a forward-facing dashboard camera during real driving, as documented by the upstream Udacity project. The source provides no machine-readable metadata about lighting, weather, time of day, region, or camera model; EXIF was stripped during the Roboflow export. File names contain a monotonically increasing integer identifier, which suggests sequential frames from one trip rather than independent scenes. Anything more specific about scene conditions (day/night, weather, region) cannot be claimed in this audit without a manual visual review of frames.

### 1.3. Roboflow preprocessing

Each image went through exactly two operations: auto-orientation with EXIF stripping, and a 512×512 stretch resize without aspect-ratio preservation. No augmentations were applied, which means the effect of the 1920×1200 → 512×512 stretch resize is baked into the data and applies identically to every split.

### 1.4. Selection and filtering

Out of the original Udacity dataset (about 97,942 objects across more than 22,000 frames), Roboflow assembled a subset of 15,000 images, which on disk is duplicated to 29,800 files (source_key groups of two frames). The explicit selection criteria are not publicly documented; the most plausible reading is that frames with broken EXIF or unreadable files were dropped and short trip segments were sampled. The dataset README explicitly warns about duplicate bounding boxes on the same object — particularly traffic lights — and these duplicates were not cleaned.

### 1.5. Annotation process and rules

Annotation was done as manual rectangular bounding boxes with a class chosen from a closed list of 11 categories: biker, car, pedestrian, truck, trafficLight, trafficLight-Green, trafficLight-GreenLeft, trafficLight-Red, trafficLight-RedLeft, trafficLight-Yellow, trafficLight-YellowLeft. For traffic lights, the class encodes the semantic state of the signal (color and presence of an arrow) rather than the geometry of the object. The bare `trafficLight` class is used as a fallback when the state cannot be determined, which makes it semantically overlap with the colored subclasses. Roboflow claims the annotations passed manual QA, but inter-annotator agreement is not reported, the annotator instructions are not public, and there are no clear rules for resolving edge cases such as a far-away flickering traffic light, partially occluded pedestrians, or partially occluded vehicles.

### 1.6. Systematic collection biases

This list contains only the biases that are confirmed either by dataset metadata or by file statistics. Hypotheses about lighting, weather, region, and seasonality are not included here: they require a visual review of frames, which was not done in this audit and should be recorded separately.

- The car class dominates: 65.7% of all bounding boxes and 97.9% of labeled frames; this is the base class of the dataset.
- Rare traffic light states: trafficLight-YellowLeft has 28 boxes across 26 labeled frames, and trafficLight-Yellow and trafficLight-GreenLeft do not exceed 614 boxes. Statistical power for these classes is very low.
- Twin frames: 14,800 source_key groups of size 2 produce autocorrelation between neighboring examples; see section 3.4.
- Annotation noise and duplicates: the dataset README explicitly warns of multiple overlapping boxes on a single object (especially for traffic lights), and these duplicates were left in.
- Semantic fallback in the taxonomy: the `trafficLight` class without a state suffix (3,175 frames) overlaps with the colored subclasses, which is built into the class list itself.
- Single preprocessing path: all 29,800 frames were resized to 512×512 by stretch without aspect preservation, so the original size and aspect are gone for every split.

The taxonomy excludes road signs, lane markings, motorcycles as a separate class (they fall under biker), buses (they fall under truck), animals, traffic cones, and special vehicles. This means the model is trained to ignore these objects and cannot report them in production even when they are physically in the frame.

Open hypotheses, requiring visual checks before they can be committed: the distribution by time of day, by weather, by region, and by road type (urban / suburban / highway).

---

## 2. Distribution analysis and subdomains

### 2.1. Summary facts

- Images: 29,800
- Label files: 29,800
- Empty-label frames: 3,500 (11.7%)
- Labeled frames: 26,300
- Total bounding boxes: 194,539
- Average boxes per frame: 6.53 (per labeled frame: 7.40)
- All images are 512×512
- Unique source_key groups: 15,000; groups of size 2: 14,800; max group size: 2

### 2.2. Image sizes

| Size | Count |
| --- | ---: |
| 512×512 | 29,800 |

There is no variation in size, which simplifies training but removes any natural multi-scale property; whatever is in the frame has already been through the stretch resize.

### 2.3. Splits (current protocol)

| Split | Count |
| --- | ---: |
| train | 20,860 |
| val | 5,960 |
| test | 2,980 |

Ratios are 70/20/10 of the total image count (see [configs/baseline.yaml](../configs/baseline.yaml), `train_ratio` / `val_ratio` / `test_ratio`).

### 2.4. Train/test overlap

- Exact path overlap: 0
- Source_key overlap: 2,099
- Share of unique train keys: 0.153739
- Share of unique test keys: 0.743009
- Examples of overlapping source_keys: 1478019956186247611, 1478019957180061202, 1478019962181150666, 1478019965682301515, 1478019966688931929, 1478019970188563338, 1478019973687625979, 1478019974679051391, 1478019975685727611, 1478019979179856014

There are no exact file duplicates, but 74.3% of the test frames have a twin in train — the next frame from the same trip. This is structural leakage; see section 3.4.

### 2.5. Class distribution

Top by box count:

| Class | Boxes |
| --- | ---: |
| car | 127,873 |
| pedestrian | 21,491 |
| trafficLight-Red | 13,673 |
| trafficLight-Green | 10,838 |
| truck | 7,194 |

Rarest classes:

| Class | Boxes |
| --- | ---: |
| trafficLight-YellowLeft | 28 |
| trafficLight-Yellow | 541 |
| trafficLight-GreenLeft | 614 |
| trafficLight-RedLeft | 3,482 |
| biker | 3,704 |

The car class outnumbers trafficLight-YellowLeft by roughly 3.7 orders of magnitude. Any aggregated metric (mAP across all classes) is essentially determined by detection quality on car, pedestrian, trafficLight-Red, and trafficLight-Green.

### 2.6. Subdomain hypotheses

Subdomains are recorded as engineering hypotheses, even when the semantics are not crisp; once recorded, they should be considered when analyzing errors and changing the protocol.

- S1. Twin frames from the same trip (about 14,800 groups of size 2). Inside a group the frames are visually almost identical: same scene, objects, lighting, viewing angle. Class distribution within a group is correlated. This is the main source of i.i.d. violation.
- S2. Empty scenes (about 11.7% of frames have no labeled objects). Frames with no labeled objects represent background without targets. Their share affects precision: the more there are in test, the higher the false-positive penalty. In training they act as negative-only examples.
- S3. Dense scenes dominated by cars (about 97.9% of labeled frames contain a car, average about 7.4 boxes per frame). This is the modal frame of the dataset — urban traffic with several vehicles.
- S4. Scenes with traffic lights in an active state (trafficLight-Red on 5,296 frames, trafficLight-Green on 3,939 frames). Semantically distinct from S3 and decisive for any traffic-light-related task.
- S5. Scenes with pedestrians and bikers (pedestrian on 7,030 frames, biker on 2,392). Small objects, frequent occlusion. Errors in this subdomain are critical for the target scenario.
- S6. Scenes with truck-class vehicles (truck on 5,019 frames). Mixed scales; visually overlapping with car (vans, pickups).
- S7. Rare traffic light states (Yellow, YellowLeft, GreenLeft, RedLeft). Very few frames (YellowLeft has 26 frames across the entire dataset), so per-class metrics have effectively zero statistical significance.
- S8. Semantic annotation noise (the bare `trafficLight` class without a state suffix, 3,175 frames). The class is by construction overlapping with the colored subclasses and is an irreducible source of training confusion.
- S9. The 1920×1200 → 512×512 stretch resize. Strictly speaking not a subdomain but a cross-cutting preprocessing artifact: horizontal compression distorts box proportions, especially on tall narrow objects like pedestrians and traffic-light poles.

S1, S7, S8, and S9 are consequences of the collection and annotation process, not properties of the real world; their effect will not shrink if the dataset grows within the same collection funnel.

---

## 3. Experimental protocol audit

### 3.1. Split scheme

The current split is implemented in [src/train.py](../src/train.py) (`prepare_splits`) and mirrored in [notebooks/train.ipynb](../notebooks/train.ipynb). The algorithm reads all images from `images_dir`, sorts them by name, shuffles them with `random.Random(seed=42)`, and slices the result by index — the first 70% to train, the next 20% to val, and the rest to test. The split is saved as three text files (train.txt, val.txt, test.txt) of absolute paths in [splits/self_driving_car/](../splits/self_driving_car/), plus a summary `data.yaml` for Ultralytics.

### 3.2. Split unit

The split unit is the individual image. There is no grouping by trip, by source_key, by time segment, or by scene. Twin frames from one source_key group (subdomain S1) end up in independent splits.

### 3.3. Implicit independence assumptions

The protocol implicitly assumes that frames are independent and identically distributed at the level of the individual image, that random index shuffling is a correct approximation of random sampling from the population, and that val and test represent the same population as train and are therefore suitable for an unbiased estimate of generalization.

All three assumptions are violated by the structure of the data. Frames from one source_key group are autocorrelated (S1), so i.i.d. at the frame level is wrong. A frame-level shuffle splits correlated pairs across splits, which systematically inflates metrics. And val and test are drawn from the same collection funnel as train, which means they do not cover scenarios like a new trip, a new day, a new region, or different weather.

### 3.4. What kind of generalization is actually tested

Because 74.3% of test frames have a twin in train, the current test set primarily measures generalization to the next frame of the same trip rather than generalization to a new road scenario. This is a weak form of generalization: the model has seen a frame in train that is fractions of a second away from the test frame and visually almost identical. Metrics on this kind of test are predictably inflated and do not predict quality on a genuinely new trip.

The protocol does not test generalization across trips, days, lighting, or regions in any form.

### 3.5. Information leakage

There is no leakage in the code (exact path overlap between train and test is 0), but leakage arises through the structure of the data. Twin-frame leakage (S1) is the main one: 74.3% of test scenes are already visible to the model through train. Annotation duplicates leak too — the dataset README warns about multiple overlapping boxes on a single object, especially for traffic lights, and since this noise is identical in train and test, the model learns to reproduce annotation noise rather than ignore it. There is also leakage through stable camera artifacts: a fixed sensor, fixed optics, and a fixed hood angle all become invariants the model can lean on, which leads to overestimating its real quality. Finally there is leakage through the single preprocessing path (S9) — every image went through the same stretch resize, which will not happen in production, so any artifact of compression that the model learned simply disappears at deployment.

All of these are protocol leaks, not code leaks. They cannot be fixed by a bugfix or by changing the random seed; the only fix is changing the split unit and/or the data source.

---

## 4. Protocol vs. application scenario

### 4.1. Intended application scenario

Based on [PROJECT_PASSPORT.md](../PROJECT_PASSPORT.md) and the current code, the system is intended for batch / offline analysis of static frames: input is a single RGB image, output is a list of detections. Requests are formally treated as independent — the pipeline does not model a video stream and does not use temporal continuity. The deployment domain is broad: different cameras, different regions, different times of day and weather; the project passport does not constrain the input domain beyond "road scene." Critical errors are missed cars, pedestrians, and active traffic lights, plus confusion of traffic light states. Less critical are minor box-coordinate offsets and duplicate detections in dense scenes.

### 4.2. Protocol vs. scenario

| Scenario property | Protocol property | Match |
| --- | --- | --- |
| Independent static requests | Frames are dependent (twin frames, single trip) | no |
| Any time of day, any weather | Daytime frames in good weather only | no |
| Any region, any camera | One region (Mountain View), one camera | no |
| RGB input without special preprocessing | All examples have been stretch-resized | partial |
| 11 classes including traffic light states | 11 classes plus a noisy trafficLight fallback | partial |
| Metrics should predict production quality | Test is overestimated due to twin-frame leakage (74.3%) | no |

### 4.3. Mismatches as engineering risks

Each mismatch is recorded as a risk with its preconditions, consequences, and the cost of detecting it.

R1. Baseline metrics are overestimated due to twin-frame leakage. Precondition: 74.3% of test frames have a twin in train. Consequence: when deployed to a new trip, mAP can drop by tens of points; readiness decisions made on the current test are unreliable. Detection cost: low — a group split by source_key and a metric recomputation (Task 6).

R2. The dataset's distribution over capture conditions is unknown. Precondition: there is no metadata about lighting, weather, or time of day; EXIF was stripped. Consequence: it cannot be confirmed or denied that train and test contain night, rain, snow, or low-sun frames. In the worst case, train and test cover the same narrow slice of conditions and the metrics do not predict quality outside it. Detection cost: medium — requires sampling and visually annotating frames by capture condition.

R3. The distribution over region and camera is unknown. Precondition: the upstream Udacity project mentions a forward-facing car camera, but the specific set of cameras, routes, and regions in this export is not documented. Consequence: when the data supplier changes, metrics may degrade silently and the risk surfaces only in production. Detection cost: high — requires an external dataset with a different camera and/or metadata that the current dataset does not provide.

R4. Per-class metrics on rare classes are meaningless. Precondition: trafficLight-YellowLeft has 28 boxes across 26 frames; the 70/20/10 split leaves about 3 boxes in test. Consequence: per-class mAP on these classes is noise; the model cannot be optimized against it; in production these states may be classified arbitrarily. Detection cost: low — compute the per-class support on test.

R5. Semantic annotation noise between trafficLight and trafficLight-<state>. Precondition: the bare `trafficLight` class (3,175 frames) overlaps with the colored subclasses. Consequence: the model will irreducibly confuse these classes, and the error is an annotation problem rather than a model problem. Detection cost: medium — a confusion table on test.

R6. Duplicate ground-truth boxes. Precondition: the dataset README warns about multiple boxes on a single object, especially for traffic lights. Consequence: precision is artificially depressed in places where the model produces a correct single box but the ground truth has a duplicate. Detection cost: medium — IoU check on duplicate ground-truth boxes per class.

R7. Stretch-resize artifacts. Precondition: 1920×1200 → 512×512 preprocessing without aspect preservation. Consequence: the model learns on compressed geometry; in production, if preprocessing differs, behavior changes. Detection cost: medium — eval with the original aspect preserved against the current pipeline.

R8. Closed class list. Precondition: 11 classes, no signs, lane markings, cones, or motorcycles. Consequence: the model does not report objects outside the class list, even when they are operationally critical; this is a blind spot in deployment. Detection cost: low — already documented at the problem-statement level.

### 4.4. Minimal plan to align the protocol with the scenario

Of the listed risks, what can be done within the current dataset and without new data is:

1. Switch to a group split by source_key (trip as the split unit), which closes R1 and turns test into a generalization-to-new-scenes setting. This is exactly Task 6 of the seminar.
2. Recompute baseline metrics on the same group split and record the delta (Task 6).
3. Add a per-subdomain breakdown (S1–S8) and a trafficLight* confusion table to the error report (Task 5).

R2, R3, and R8 are recorded as open and moved into the future-work plan; they cannot be closed without expanding the data source.

---

## 5. Baseline error analysis

The error analysis uses the group-split test set (2,971 frames, zero source_key overlap with train) with the ground-truth and prediction records from `runs/20260426_135344_groupsplit_test/`. Matching is done at box level: a predicted box is considered correct if it overlaps a ground-truth box with IoU ≥ 0.5 and the class matches. Confidence threshold is 0.25.

At this threshold the model produces 13,843 correct detections, 5,019 missed ground-truth boxes, and 2,648 spurious detections with no ground-truth match. An additional 354 boxes are localized correctly (IoU ≥ 0.5) but assigned the wrong class. Misses outnumber false positives by almost 2:1, which means the main failure mode is not hallucination but under-detection.

The miss rate varies significantly by class. Pedestrians and bikers are the hardest: the model misses roughly half of them (49% and 46% respectively). Both are tall, narrow objects, and the 1920×1200 → 512×512 stretch resize compresses vertical extent more than horizontal, so these classes lose the most discriminative pixels. Cars have a lower miss rate (21%) but contribute the most false positives in absolute terms (1,758), partly because the dataset README warns of duplicate ground-truth boxes on the same vehicle, which penalizes precision artificially. Trucks are frequently confused with cars (73 wrong-class matches), which is a genuine model limitation but also a real annotation ambiguity since vans and pickups sit visually between the two classes.

Traffic lights deserve separate attention. The model finds the `trafficLight` fallback class reasonably well (17% FN) but confuses it heavily with the colored state subclasses. The dominant wrong-class pattern is `trafficLight-Red → trafficLight` (32 cases) and `trafficLight → trafficLight-Green` (17 cases). Both directions are consistent with the annotation problem described in section 1.5: the bare `trafficLight` class is used when the annotator could not determine the state, so it semantically overlaps with all the state-specific classes. The model is not learning a wrong concept — it is learning an inconsistent label. The `trafficLight-Red → trafficLight-RedLeft` confusion (12 cases) is a genuine model error caused by too few RedLeft examples at this image scale. The `trafficLight-Yellow → trafficLight-Red` confusion (6 cases) would be operationally significant, but with only 63 yellow-light boxes in the entire test set any per-class conclusion is statistically unreliable.

Scenes with rare traffic-light states (Yellow, YellowLeft, GreenLeft) produced 55 wrong-class matches across 84 scenes, a higher error rate than any other subdomain. This is not a finding about the model's behavior on yellow lights — it is confirmation that the test support for these classes is too small to measure anything. Empty scenes (381 frames with no ground-truth boxes) generated only 10 false positives total, so the model is not hallucinating objects on clean backgrounds.

Separating model errors from dataset artifacts is not fully possible without re-annotating, but the breakdown is roughly this. The miss rate on pedestrians and bikers is primarily a stretch-resize artifact, not a model failure in the original resolution. The elevated false-positive count on dense car scenes is at least partly from duplicate ground-truth boxes. The trafficLight wrong-class errors are largely annotation noise. What remains — missed traffic lights with valid ground truth, truck/car confusion, and some of the car misses in cluttered scenes — is attributable to the model itself.

---

## 6. Sensitivity to the split protocol

The one change tested here is switching the split unit from individual image to source_key group. Everything else is held fixed: the same model weights (`best_yolo_auto_transport.pt`), the same evaluation hyperparameters (conf=0.001, IoU=0.6), and the same total pool of 29,800 frames.

Image-split test (2,980 frames, 74.3% source_key overlap with train): precision 0.845, recall 0.580, mAP50 0.659, mAP50-95 0.359.

Group-split test (2,971 frames, 0% source_key overlap with train): precision 0.812, recall 0.592, mAP50 0.698, mAP50-95 0.376.

The group-split gives higher mAP50 and mAP50-95, which is the opposite of what the leakage argument predicts. The explanation is that the model was trained on the group-split train set (the current `train.txt`), so the group-split test draws from trips the model never saw in training. The image-split test, by contrast, contains frames from trips that were split across train and test at the image level — some of those trips ended up underrepresented in the group-split train, making the image-split test harder for this particular model. In other words, the direction of the mAP difference depends on which split was used for training, not just which is used for testing.

Precision drops by 3.3 points on the group-split relative to image-split, while recall goes up slightly. This is consistent with what section 3.4 describes: twin frames in the image-split test are slightly easier to detect (the model has seen nearly identical scenes in train), which inflates precision without affecting recall much. The mAP gap (4 points on mAP50) is large enough to matter for any decision made on the basis of these numbers.

The broader point is that switching the split unit changes the measured result noticeably, even when the actual model and dataset are unchanged. If the training split were also changed to image-level, the numbers would shift again in a different direction. Protocol sensitivity of this magnitude means that metric comparisons across experiments are only meaningful if both experiments use the same split scheme.
