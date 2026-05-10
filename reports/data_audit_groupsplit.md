# Data Audit

## Basic Facts

- Images: 29800
- Label files: 29800
- Empty-label images: 3500
- Labeled images: 26300
- Total boxes: 194539
- Average boxes per image: 6.5282
- Average boxes per labeled image: 7.3969

## Image Sizes

| Size | Count |
| --- | ---: |
| 512x512 | 29800 |

## Split Sizes

| Split | Count |
| --- | ---: |
| train | 20878 |
| val | 5951 |
| test | 2971 |

## Train/Test Overlap

- Exact image-path overlap: 0
- Exact overlap ratio vs train: 0.0
- Exact overlap ratio vs test: 0.0
- Source-key overlap: 0
- Source-key overlap ratio vs train keys: 0.0
- Source-key overlap ratio vs test keys: 0.0
- Example overlapping source keys: none

## Top Classes By Box Count

| Class | Boxes |
| --- | ---: |
| car | 127873 |
| pedestrian | 21491 |
| trafficLight-Red | 13673 |
| trafficLight-Green | 10838 |
| truck | 7194 |

## Rarest Classes By Box Count

| Class | Boxes |
| --- | ---: |
| trafficLight-YellowLeft | 28 |
| trafficLight-Yellow | 541 |
| trafficLight-GreenLeft | 614 |
| trafficLight-RedLeft | 3482 |
| biker | 3704 |

## Simple Distribution Notes

- The dataset is uniform in image size: all images are `512x512`.
- The class distribution is highly imbalanced. `car` dominates the dataset.
- Rare traffic light state classes are strongly underrepresented, especially `trafficLight-YellowLeft`.
- The dataset contains many empty-label images, so background-only scenes are a visible part of the distribution.
- File names suggest repeated source groups: 14800 duplicated `source_key` values out of 15000 unique groups.
- Train/test exact path overlap is 0, but train/test `source_key` overlap is 0.
