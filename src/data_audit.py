import argparse
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from src.utils import ROOT, resolve_path, source_key, write_json


def load_config(config_path):
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data = config.setdefault("data", {})
    dataset_root = resolve_path(data.get("dataset_root", "data/Self-Driving-Car-3"))
    data["dataset_root"] = str(dataset_root)
    data["images_dir"] = str(resolve_path(data.get("images_dir", "export/images"), dataset_root))
    data["labels_dir"] = str(resolve_path(data.get("labels_dir", "export/labels"), dataset_root))
    data["meta_file"] = str(resolve_path(data.get("meta_file", "data.yaml"), dataset_root))
    data["split_dir"] = str(resolve_path(data.get("split_dir", "splits/self_driving_car")))
    return config


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_split_items(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split_count(path):
    items = read_split_items(path)
    if not items:
        return None
    return len(items)


def compute_train_test_overlap(split_dir):
    split_dir = Path(split_dir)
    train_items = read_split_items(split_dir / "train.txt")
    test_items = read_split_items(split_dir / "test.txt")

    if not train_items or not test_items:
        return None

    exact_overlap_paths = sorted(set(train_items) & set(test_items))
    train_source_keys = {source_key(path) for path in train_items}
    test_source_keys = {source_key(path) for path in test_items}
    source_key_overlap = sorted(train_source_keys & test_source_keys)

    return {
        "train_count": len(train_items),
        "test_count": len(test_items),
        "exact_path_overlap_count": len(exact_overlap_paths),
        "exact_path_overlap_ratio_train": round(len(exact_overlap_paths) / len(train_items), 6),
        "exact_path_overlap_ratio_test": round(len(exact_overlap_paths) / len(test_items), 6),
        "exact_path_overlap_examples": exact_overlap_paths[:10],
        "source_key_overlap_count": len(source_key_overlap),
        "source_key_overlap_ratio_train": round(len(source_key_overlap) / len(train_source_keys), 6),
        "source_key_overlap_ratio_test": round(len(source_key_overlap) / len(test_source_keys), 6),
        "source_key_overlap_examples": source_key_overlap[:10],
    }


def collect_stats(config):
    data = config["data"]
    dataset_root = Path(data["dataset_root"])
    images_dir = Path(data["images_dir"])
    labels_dir = Path(data["labels_dir"])
    meta_file = Path(data["meta_file"])
    split_dir = Path(data["split_dir"])

    with meta_file.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    class_names = meta["names"]
    image_paths = sorted([path for path in images_dir.iterdir() if path.is_file()])
    label_paths = sorted(labels_dir.glob("*.txt"))

    image_sizes = Counter()
    class_counts = Counter()
    class_image_counts = Counter()
    source_counts = Counter()
    empty_label_count = 0
    total_boxes = 0

    for image_path in image_paths:
        source_counts[source_key(image_path)] += 1
        with Image.open(image_path) as image:
            image_sizes[f"{image.width}x{image.height}"] += 1

    for label_path in label_paths:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_label_count += 1
            continue

        seen_classes = set()
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            class_name = class_names[class_id]
            class_counts[class_name] += 1
            seen_classes.add(class_name)
            total_boxes += 1

        for class_name in seen_classes:
            class_image_counts[class_name] += 1

    split_counts = {}
    for split_name in ["train", "val", "test"]:
        count = read_split_count(split_dir / f"{split_name}.txt")
        if count is not None:
            split_counts[split_name] = count

    summary = {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "empty_label_count": empty_label_count,
        "labeled_image_count": len(label_paths) - empty_label_count,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / len(image_paths), 4) if image_paths else 0.0,
        "avg_boxes_per_labeled_image": round(total_boxes / (len(label_paths) - empty_label_count), 4) if len(label_paths) - empty_label_count > 0 else 0.0,
        "image_size_counts": dict(sorted(image_sizes.items())),
        "split_counts": split_counts,
        "class_counts": {name: class_counts.get(name, 0) for name in class_names},
        "class_image_counts": {name: class_image_counts.get(name, 0) for name in class_names},
        "unique_source_keys": len(source_counts),
        "duplicated_source_keys": sum(1 for count in source_counts.values() if count > 1),
        "max_source_group_size": max(source_counts.values()) if source_counts else 0,
        "top_classes_by_boxes": [
            {"class_name": name, "count": count}
            for name, count in class_counts.most_common(5)
        ],
        "rarest_classes_by_boxes": [
            {"class_name": name, "count": count}
            for name, count in sorted(class_counts.items(), key=lambda item: (item[1], item[0]))[:5]
        ],
        "train_test_overlap": compute_train_test_overlap(split_dir),
    }

    return summary


def build_report(summary):
    overlap = summary.get("train_test_overlap")
    top_classes = "\n".join(
        [
            f"| {item['class_name']} | {item['count']} |"
            for item in summary["top_classes_by_boxes"]
        ]
    )

    rarest_classes = "\n".join(
        [
            f"| {item['class_name']} | {item['count']} |"
            for item in summary["rarest_classes_by_boxes"]
        ]
    )

    image_sizes = "\n".join(
        [
            f"| {size} | {count} |"
            for size, count in summary["image_size_counts"].items()
        ]
    )

    split_rows = "\n".join(
        [
            f"| {name} | {count} |"
            for name, count in summary["split_counts"].items()
        ]
    )

    if overlap:
        overlap_section = f"""## Train/Test Overlap

- Exact image-path overlap: {overlap["exact_path_overlap_count"]}
- Exact overlap ratio vs train: {overlap["exact_path_overlap_ratio_train"]}
- Exact overlap ratio vs test: {overlap["exact_path_overlap_ratio_test"]}
- Source-key overlap: {overlap["source_key_overlap_count"]}
- Source-key overlap ratio vs train keys: {overlap["source_key_overlap_ratio_train"]}
- Source-key overlap ratio vs test keys: {overlap["source_key_overlap_ratio_test"]}
- Example overlapping source keys: {", ".join(overlap["source_key_overlap_examples"]) if overlap["source_key_overlap_examples"] else "none"}

"""
    else:
        overlap_section = """## Train/Test Overlap

- Train/test overlap could not be computed because one of the split files is missing or empty.

"""

    overlap_note = (
        f'- Train/test exact path overlap is {overlap["exact_path_overlap_count"]}, '
        f'but train/test `source_key` overlap is {overlap["source_key_overlap_count"]}.'
        if overlap
        else "- Train/test overlap is not available."
    )

    return f"""# Data Audit

## Basic Facts

- Images: {summary["image_count"]}
- Label files: {summary["label_count"]}
- Empty-label images: {summary["empty_label_count"]}
- Labeled images: {summary["labeled_image_count"]}
- Total boxes: {summary["total_boxes"]}
- Average boxes per image: {summary["avg_boxes_per_image"]}
- Average boxes per labeled image: {summary["avg_boxes_per_labeled_image"]}

## Image Sizes

| Size | Count |
| --- | ---: |
{image_sizes}

## Split Sizes

| Split | Count |
| --- | ---: |
{split_rows}

{overlap_section}## Top Classes By Box Count

| Class | Boxes |
| --- | ---: |
{top_classes}

## Rarest Classes By Box Count

| Class | Boxes |
| --- | ---: |
{rarest_classes}

## Simple Distribution Notes

- The dataset is uniform in image size: all images are `512x512`.
- The class distribution is highly imbalanced. `car` dominates the dataset.
- Rare traffic light state classes are strongly underrepresented, especially `trafficLight-YellowLeft`.
- The dataset contains many empty-label images, so background-only scenes are a visible part of the distribution.
- File names suggest repeated source groups: {summary["duplicated_source_keys"]} duplicated `source_key` values out of {summary["unique_source_keys"]} unique groups.
{overlap_note}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-json", default="reports/data_audit.json")
    parser.add_argument("--output-md", default="reports/data_audit.md")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = collect_stats(config)
    write_json(args.output_json, summary)
    write_text(args.output_md, build_report(summary))
    print(Path(args.output_md).resolve())
