import csv
import json
from collections import defaultdict
from pathlib import Path

import click

from src.box_metrics import match_image, precision_recall_f1


TL_STATES = {"trafficLight-Green", "trafficLight-GreenLeft", "trafficLight-Red",
             "trafficLight-RedLeft", "trafficLight-Yellow", "trafficLight-YellowLeft"}
RARE_TL = {"trafficLight-YellowLeft", "trafficLight-Yellow", "trafficLight-GreenLeft"}


def box_area(b):
    x1, y1, x2, y2 = b["xyxy"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def scene_slices(gt_boxes, image_w=512, image_h=512):
    classes = {b["class_name"] for b in gt_boxes}
    labels = set()

    if not gt_boxes:
        labels.add("empty")
        return labels

    if classes & TL_STATES:
        labels.add("tl_state")
    if classes & RARE_TL:
        labels.add("rare_tl")
    if "car" in classes and sum(1 for b in gt_boxes if b["class_name"] == "car") >= 5:
        labels.add("dense_car")
    if classes & {"pedestrian", "biker"}:
        labels.add("ped_biker")
    if "truck" in classes:
        labels.add("truck")

    frame_area = image_w * image_h
    avg_size = sum(box_area(b) for b in gt_boxes) / len(gt_boxes) / frame_area
    if avg_size < 0.005:
        labels.add("size_small")
    elif avg_size < 0.05:
        labels.add("size_medium")
    else:
        labels.add("size_large")

    if len(gt_boxes) <= 2:
        labels.add("count_sparse")
    elif len(gt_boxes) <= 8:
        labels.add("count_medium")
    else:
        labels.add("count_dense")

    return labels


def per_class_stats(pairs_per_image):
    classes = set()
    fn = defaultdict(int)
    fp = defaultdict(int)
    tp = defaultdict(int)
    wc = defaultdict(int)
    for entry in pairs_per_image:
        for p in entry["pairs"]:
            if p["kind"] == "tp":
                tp[p["gt_class"]] += 1
                classes.add(p["gt_class"])
            elif p["kind"] == "fn":
                fn[p["gt_class"]] += 1
                classes.add(p["gt_class"])
            elif p["kind"] == "wrong_class":
                wc[p["gt_class"]] += 1
                classes.add(p["gt_class"])
            elif p["kind"] == "fp":
                fp[p["pred_class"]] += 1
                classes.add(p["pred_class"])
    rows = []
    for cls in sorted(classes):
        t = tp[cls]
        f_p = fp[cls]
        f_n = fn[cls]
        w_c = wc[cls]
        prec = t / max(1, t + f_p + w_c)
        rec = t / max(1, t + f_n + w_c)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({
            "slice": f"class:{cls}",
            "support": t + f_n + w_c,
            "tp": t, "fp": f_p, "fn": f_n, "wrong_class": w_c,
            "precision": prec, "recall": rec, "f1": f1,
        })
    return rows


def aggregate_slice(pairs_subset):
    counts = {"tp": 0, "fp": 0, "fn": 0, "wrong_class": 0}
    for pairs in pairs_subset:
        for p in pairs:
            counts[p["kind"]] += 1
    m = precision_recall_f1(counts)
    return {**counts, **m}


def compute_slices(gt_by_path, pred_by_path, tau=0.25):
    pairs_per_image = []
    for path, gt_boxes in gt_by_path.items():
        pred_boxes = pred_by_path.get(path, [])
        pairs = match_image(gt_boxes, pred_boxes, conf_thresh=tau)
        pairs_per_image.append({"image_path": path, "pairs": pairs})

    by_slice = defaultdict(list)
    for entry in pairs_per_image:
        gt_boxes = gt_by_path.get(entry["image_path"], [])
        for label in scene_slices(gt_boxes):
            by_slice[label].append(entry["pairs"])

    rows = []
    for label, pairs_list in by_slice.items():
        agg = aggregate_slice(pairs_list)
        rows.append({
            "slice": f"scene:{label}",
            "support": len(pairs_list),
            **agg,
        })

    rows.extend(per_class_stats(pairs_per_image))
    return rows


@click.command()
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--pred", "pred_path", default="runs/20260426_135344_groupsplit_test/pred.json", type=click.Path(path_type=Path))
@click.option("--tau", default=0.25, type=float)
@click.option("--out", default="reports/slice_metrics.csv", type=click.Path(path_type=Path))
def main(gt_path, pred_path, tau, out):
    gt_data = json.load(open(gt_path))
    pred_data = json.load(open(pred_path))
    gt_by_path = {r["image_path"]: r["boxes"] for r in gt_data["images"]}
    pred_by_path = {r["image_path"]: r["boxes"] for r in pred_data["images"]}

    rows = compute_slices(gt_by_path, pred_by_path, tau=tau)
    rows.sort(key=lambda r: r["f1"])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slice", "support", "tp", "fp", "fn", "wrong_class", "precision", "recall", "f1"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    click.echo(str(out))


if __name__ == "__main__":
    main()
