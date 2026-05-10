import json
from collections import defaultdict
from pathlib import Path

import click

from src.box_metrics import aggregate, match_image, precision_recall_f1
from src.utils import write_json


TL_STATES = {"trafficLight-Green", "trafficLight-GreenLeft", "trafficLight-Red",
             "trafficLight-RedLeft", "trafficLight-Yellow", "trafficLight-YellowLeft"}
RARE_TL = {"trafficLight-YellowLeft", "trafficLight-Yellow", "trafficLight-GreenLeft"}


def load_pairs_for(run_dir, transform, alpha):
    fname = run_dir / f"{transform}_a{alpha}.json"
    return json.load(open(fname))


def degradation_curves(run_dir):
    summary = json.load(open(run_dir / "summary.json"))
    by_transform = defaultdict(list)
    for r in summary["results"]:
        by_transform[r["transform"]].append({
            "alpha": r["alpha"],
            "unit": r["unit"],
            "counts": r["counts"],
            "metrics": r["metrics"],
        })
    for t in by_transform:
        by_transform[t].sort(key=lambda r: r["alpha"])
    return dict(by_transform)


def calibration(pairs_all, n_bins=10):
    bins = [{"low": i / n_bins, "high": (i + 1) / n_bins, "n": 0, "correct": 0, "conf_sum": 0.0}
            for i in range(n_bins)]
    for entry in pairs_all:
        for p in entry["pairs"]:
            if p["kind"] == "fn" or p.get("confidence") is None:
                continue
            c = p["confidence"]
            idx = min(int(c * n_bins), n_bins - 1)
            bins[idx]["n"] += 1
            bins[idx]["conf_sum"] += c
            if p["kind"] == "tp":
                bins[idx]["correct"] += 1

    total = sum(b["n"] for b in bins) or 1
    ece = 0.0
    out = []
    for b in bins:
        if b["n"] == 0:
            out.append({"low": b["low"], "high": b["high"], "count": 0, "acc": None, "conf": None})
            continue
        acc = b["correct"] / b["n"]
        conf = b["conf_sum"] / b["n"]
        ece += (b["n"] / total) * abs(acc - conf)
        out.append({"low": b["low"], "high": b["high"], "count": b["n"], "acc": acc, "conf": conf})
    return {"ece": ece, "bins": out}


def rematch_at_tau(predictions, gt_by_path, tau, iou_thresh=0.5):
    pairs_all = []
    for pred in predictions:
        gt_boxes = gt_by_path.get(pred["image_path"], [])
        pairs = match_image(gt_boxes, pred["boxes"], iou_thresh=iou_thresh, conf_thresh=tau)
        pairs_all.append({"image_path": pred["image_path"], "pairs": pairs})
    return pairs_all


def operating_point_sweep(predictions, gt_by_path, taus, iou_thresh=0.5):
    rows = []
    for tau in taus:
        pairs_all = rematch_at_tau(predictions, gt_by_path, tau, iou_thresh=iou_thresh)
        c = aggregate(pairs_all)
        m = precision_recall_f1(c)
        rows.append({"tau": tau, "counts": c, **m})
    return rows


def subdomain_label(gt_boxes):
    labels = set()
    classes = {b["class_name"] for b in gt_boxes}
    if not gt_boxes:
        labels.add("empty")
        return labels
    if classes & TL_STATES:
        labels.add("tl_state")
    if classes & RARE_TL:
        labels.add("rare_tl")
    car_n = sum(1 for b in gt_boxes if b["class_name"] == "car")
    if car_n >= 5:
        labels.add("dense_car")
    if classes & {"pedestrian", "biker"}:
        labels.add("ped_biker")
    if "truck" in classes:
        labels.add("truck")
    return labels


def worst_slice(pairs_all, gt_by_path):
    by_slice = defaultdict(list)
    for entry in pairs_all:
        path = entry["image_path"]
        gt_boxes = gt_by_path.get(path, [])
        for label in subdomain_label(gt_boxes):
            by_slice[label].append(entry["pairs"])

    out = {}
    for label, all_pairs in by_slice.items():
        c = {"tp": 0, "fp": 0, "fn": 0, "wrong_class": 0}
        for pairs in all_pairs:
            for p in pairs:
                c[p["kind"]] += 1
        out[label] = {
            "n_scenes": len(all_pairs),
            "counts": c,
            **precision_recall_f1(c),
        }
    return out


def cascade_decomposition(pairs_all):
    detected = 0
    detected_correct_class = 0
    detected_wrong_class = 0
    missed = 0
    spurious = 0
    for entry in pairs_all:
        for p in entry["pairs"]:
            if p["kind"] == "tp":
                detected += 1
                detected_correct_class += 1
            elif p["kind"] == "wrong_class":
                detected += 1
                detected_wrong_class += 1
            elif p["kind"] == "fn":
                missed += 1
            elif p["kind"] == "fp":
                spurious += 1
    total_gt = detected + missed
    return {
        "p_localized": detected / max(1, total_gt),
        "p_correct_class_given_localized": detected_correct_class / max(1, detected),
        "p_missed": missed / max(1, total_gt),
        "spurious_per_image": spurious / max(1, len(pairs_all)),
        "counts": {
            "localized": detected,
            "localized_correct": detected_correct_class,
            "localized_wrong_class": detected_wrong_class,
            "missed": missed,
            "spurious": spurious,
        },
    }


@click.command()
@click.option("--run-dir", required=True, type=click.Path(path_type=Path))
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--shift-transform", default="blur")
@click.option("--shift-alpha", default=4, type=float)
def main(run_dir, gt_path, shift_transform, shift_alpha):
    run_dir = Path(run_dir)
    out = run_dir / "analysis.json"

    summary = json.load(open(run_dir / "summary.json"))
    gt_data = json.load(open(gt_path))
    gt_by_path = {r["image_path"]: r["boxes"] for r in gt_data["images"]}

    curves = degradation_curves(run_dir)

    nominal_choices = [
        ("blur", 0), ("noise", 0), ("brightness", 0), ("contrast", 1.0),
        ("gamma", 1.0), ("jpeg", 100), ("occlusion", 0),
    ]
    nominal_loaded = None
    for tname, alpha in nominal_choices:
        f = run_dir / f"{tname}_a{alpha}.json"
        if f.exists():
            nominal_loaded = json.load(open(f))
            break
    if nominal_loaded is None:
        raise FileNotFoundError("no nominal (identity) run found")

    nominal_preds = nominal_loaded["predictions"]
    nominal_pairs_all = rematch_at_tau(nominal_preds, gt_by_path, tau=0.0)
    nominal_pairs_default = rematch_at_tau(nominal_preds, gt_by_path, tau=0.25)
    nominal_calib = calibration(nominal_pairs_all)
    nominal_op = operating_point_sweep(nominal_preds, gt_by_path,
                                       [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    nominal_slices = worst_slice(nominal_pairs_default, gt_by_path)
    nominal_cascade = cascade_decomposition(nominal_pairs_default)

    shift_file = run_dir / f"{shift_transform}_a{shift_alpha}.json"
    if not shift_file.exists():
        for ext in (int(shift_alpha), float(shift_alpha)):
            cand = run_dir / f"{shift_transform}_a{ext}.json"
            if cand.exists():
                shift_file = cand
                break
    shift_loaded = json.load(open(shift_file))
    shift_preds = shift_loaded["predictions"]
    shift_pairs_all = rematch_at_tau(shift_preds, gt_by_path, tau=0.0)
    shift_pairs_default = rematch_at_tau(shift_preds, gt_by_path, tau=0.25)
    shift_calib = calibration(shift_pairs_all)
    shift_op = operating_point_sweep(shift_preds, gt_by_path,
                                     [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    shift_slices = worst_slice(shift_pairs_default, gt_by_path)
    shift_cascade = cascade_decomposition(shift_pairs_default)

    nom_best = max(nominal_op, key=lambda r: r["f1"])
    sh_best = max(shift_op, key=lambda r: r["f1"])
    robust_taus = sorted(set(r["tau"] for r in nominal_op))
    robust = []
    for tau in robust_taus:
        nom_f1 = next(r["f1"] for r in nominal_op if r["tau"] == tau)
        sh_f1 = next(r["f1"] for r in shift_op if r["tau"] == tau)
        robust.append({"tau": tau, "min_f1": min(nom_f1, sh_f1)})
    robust_best = max(robust, key=lambda r: r["min_f1"])

    out_payload = {
        "run_dir": str(run_dir),
        "n_images": summary["n_images"],
        "shift": {"transform": shift_transform, "alpha": shift_alpha, "file": shift_file.name},
        "degradation_curves": curves,
        "calibration": {"nominal": nominal_calib, "shift": shift_calib},
        "operating_point": {
            "nominal": nominal_op,
            "shift": shift_op,
            "nominal_best_f1_tau": nom_best["tau"],
            "shift_best_f1_tau": sh_best["tau"],
            "robust_tau": robust_best["tau"],
            "robust_min_f1": robust_best["min_f1"],
        },
        "worst_slice": {"nominal": nominal_slices, "shift": shift_slices},
        "cascade": {"nominal": nominal_cascade, "shift": shift_cascade},
    }
    write_json(out, out_payload)
    click.echo(str(out))


if __name__ == "__main__":
    main()
