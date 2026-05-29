import csv
import json
from pathlib import Path

import click

from src.box_metrics import iou_xyxy, match_image, precision_recall_f1
from src.utils import write_json


PER_CLASS_FN_COSTS = {
    "pedestrian": 10,
    "biker": 10,
    "car": 5,
    "truck": 5,
    "trafficLight-Red": 3,
    "trafficLight-Green": 3,
    "trafficLight-Yellow": 3,
    "trafficLight-RedLeft": 3,
    "trafficLight-GreenLeft": 3,
    "trafficLight-YellowLeft": 3,
    "trafficLight": 2,
}
PER_CLASS_FP_COSTS = {k: 1 for k in PER_CLASS_FN_COSTS}


def load_records(gt_path, pred_path):
    gt_data = json.load(open(gt_path))
    pred_data = json.load(open(pred_path))
    gt_by_path = {r["image_path"]: r["boxes"] for r in gt_data["images"]}
    pred_by_path = {r["image_path"]: r["boxes"] for r in pred_data["images"]}
    return gt_by_path, pred_by_path


def sweep_one_threshold(gt_by_path, pred_by_path, tau, iou_thresh=0.5):
    counts = {"tp": 0, "fp": 0, "fn": 0, "wrong_class": 0}
    fn_by_class = {}
    fp_by_class = {}
    tn_scenes = 0
    fp_scenes = 0
    empty_scenes = 0

    for path, gt_boxes in gt_by_path.items():
        pred_boxes = pred_by_path.get(path, [])
        pairs = match_image(gt_boxes, pred_boxes, iou_thresh=iou_thresh, conf_thresh=tau)
        scene_has_emission = any(p["confidence"] >= tau for p in pred_boxes if p.get("confidence") is not None)

        if not gt_boxes:
            empty_scenes += 1
            if scene_has_emission:
                fp_scenes += 1
            else:
                tn_scenes += 1

        for p in pairs:
            counts[p["kind"]] += 1
            if p["kind"] == "fn":
                fn_by_class[p["gt_class"]] = fn_by_class.get(p["gt_class"], 0) + 1
            elif p["kind"] == "fp":
                fp_by_class[p["pred_class"]] = fp_by_class.get(p["pred_class"], 0) + 1
            elif p["kind"] == "wrong_class":
                fn_by_class[p["gt_class"]] = fn_by_class.get(p["gt_class"], 0) + 1
                fp_by_class[p["pred_class"]] = fp_by_class.get(p["pred_class"], 0) + 1

    metrics = precision_recall_f1(counts)
    tpr = metrics["recall"]
    fp_total = counts["fp"] + counts["wrong_class"]
    fpr = fp_total / max(1, fp_total + tn_scenes)

    accuracy = (counts["tp"] + tn_scenes) / max(1, counts["tp"] + counts["fp"] + counts["fn"] + counts["wrong_class"] + tn_scenes)

    return {
        "threshold": tau,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": tn_scenes,
        "fn": counts["fn"],
        "wrong_class": counts["wrong_class"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fpr": fpr,
        "tpr": tpr,
        "f1": metrics["f1"],
        "accuracy": accuracy,
        "fn_by_class": fn_by_class,
        "fp_by_class": fp_by_class,
        "empty_scenes": empty_scenes,
    }


def uniform_risk(row, c_fp, c_fn):
    return c_fp * (row["fp"] + row["wrong_class"]) + c_fn * (row["fn"] + row["wrong_class"])


def per_class_risk(row, fn_costs=PER_CLASS_FN_COSTS, fp_costs=PER_CLASS_FP_COSTS):
    fn_cost = sum(row["fn_by_class"].get(cls, 0) * cost for cls, cost in fn_costs.items())
    fp_cost = sum(row["fp_by_class"].get(cls, 0) * cost for cls, cost in fp_costs.items())
    return fn_cost + fp_cost


def write_csv(rows, path, c_fp, c_fn):
    cols = ["threshold", "tp", "fp", "tn", "fn", "wrong_class",
            "precision", "recall", "fpr", "tpr", "f1", "accuracy",
            "risk_uniform", "risk_per_class"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            r = {k: row[k] for k in cols if k in row}
            r["risk_uniform"] = uniform_risk(row, c_fp, c_fn)
            r["risk_per_class"] = per_class_risk(row)
            w.writerow(r)


@click.command()
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--pred", "pred_path", default="runs/20260426_135344_groupsplit_test/pred.json", type=click.Path(path_type=Path))
@click.option("--out-csv", default="reports/threshold_sweep.csv", type=click.Path(path_type=Path))
@click.option("--out-json", default="reports/threshold_sweep.json", type=click.Path(path_type=Path))
@click.option("--c-fp", default=1.0, type=float)
@click.option("--c-fn", default=5.0, type=float)
@click.option("--n-thresholds", default=51, type=int)
def main(gt_path, pred_path, out_csv, out_json, c_fp, c_fn, n_thresholds):
    gt_by_path, pred_by_path = load_records(gt_path, pred_path)

    thresholds = [round(i / (n_thresholds - 1), 4) for i in range(n_thresholds)]
    rows = []
    for tau in thresholds:
        click.echo(f"tau={tau:.4f}")
        rows.append(sweep_one_threshold(gt_by_path, pred_by_path, tau))

    out_csv = Path(out_csv)
    out_json = Path(out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv, c_fp, c_fn)

    enriched = []
    for row in rows:
        r = dict(row)
        r["risk_uniform"] = uniform_risk(row, c_fp, c_fn)
        r["risk_per_class"] = per_class_risk(row)
        enriched.append(r)
    write_json(out_json, {
        "c_fp": c_fp,
        "c_fn": c_fn,
        "per_class_fn_costs": PER_CLASS_FN_COSTS,
        "per_class_fp_costs": PER_CLASS_FP_COSTS,
        "rows": enriched,
    })

    click.echo(str(out_csv))
    click.echo(str(out_json))


if __name__ == "__main__":
    main()
