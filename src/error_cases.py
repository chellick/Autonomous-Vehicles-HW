import json
import random
from collections import defaultdict
from pathlib import Path

import click
from PIL import Image, ImageDraw, ImageFont

from src.box_metrics import match_image


GT_COLOR = "#20a050"
PRED_COLOR = "#d04040"


def load_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_label(draw, x, y, text, color, font):
    bbox = draw.textbbox((x, y), text, font=font) if font else draw.textbbox((x, y), text)
    pad = (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1)
    draw.rectangle(pad, fill=color)
    draw.text((x, y), text, fill="white", font=font)


def render_overlay(image_path, gt_boxes, pred_boxes, tau, target_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()
    for gb in gt_boxes:
        x1, y1, x2, y2 = gb["xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=GT_COLOR, width=2)
        draw_label(draw, x1, max(0, y1 - 12), f"GT {gb['class_name']}", GT_COLOR, font)
    for pb in pred_boxes:
        if pb.get("confidence", 1.0) < tau:
            continue
        x1, y1, x2, y2 = pb["xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=PRED_COLOR, width=2)
        label = f"PRED {pb['class_name']} {pb['confidence']:.2f}"
        draw_label(draw, x1, y2 + 1, label, PRED_COLOR, font)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path)


def categorize(gt_boxes, pred_boxes, tau, iou_thresh=0.5):
    pairs = match_image(gt_boxes, pred_boxes, iou_thresh=iou_thresh, conf_thresh=tau)
    has_tp = any(p["kind"] == "tp" for p in pairs)
    has_fp = any(p["kind"] == "fp" or p["kind"] == "wrong_class" for p in pairs)
    has_fn = any(p["kind"] == "fn" for p in pairs)
    label = []
    if not gt_boxes and not has_fp:
        label.append("true_negative")
    if has_tp:
        label.append("true_positive")
    if has_fp:
        label.append("false_positive")
    if has_fn:
        label.append("false_negative")

    max_fp_conf = 0.0
    for p in pairs:
        if p["kind"] in ("fp", "wrong_class") and p.get("confidence") is not None:
            max_fp_conf = max(max_fp_conf, p["confidence"])
    max_tp_conf = 0.0
    for p in pairs:
        if p["kind"] == "tp" and p.get("confidence") is not None:
            max_tp_conf = max(max_tp_conf, p["confidence"])
    return label, {"max_fp_conf": max_fp_conf, "max_tp_conf": max_tp_conf, "n_fn": sum(1 for p in pairs if p["kind"] == "fn")}


def select_examples(candidates, by, k_top, k_random, rng):
    sorted_high = sorted(candidates, key=lambda c: c["info"][by], reverse=True)
    chosen = sorted_high[:k_top]
    chosen_paths = {c["image_path"] for c in chosen}
    pool = [c for c in candidates if c["image_path"] not in chosen_paths]
    if pool and k_random > 0:
        chosen.extend(rng.sample(pool, min(k_random, len(pool))))
    return chosen


@click.command()
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--pred", "pred_path", default="runs/20260426_135344_groupsplit_test/pred.json", type=click.Path(path_type=Path))
@click.option("--out-dir", default="reports/error_cases", type=click.Path(path_type=Path))
@click.option("--tau", default=0.25, type=float)
@click.option("--per-cat", default=12, type=int)
def main(gt_path, pred_path, out_dir, tau, per_cat):
    gt_data = json.load(open(gt_path))
    pred_data = json.load(open(pred_path))
    gt_by_path = {r["image_path"]: r["boxes"] for r in gt_data["images"]}
    pred_by_path = {r["image_path"]: r["boxes"] for r in pred_data["images"]}

    cat_candidates = defaultdict(list)
    for path, gt_boxes in gt_by_path.items():
        pred_boxes = pred_by_path.get(path, [])
        labels, info = categorize(gt_boxes, pred_boxes, tau)
        for label in labels:
            cat_candidates[label].append({
                "image_path": path,
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                "info": info,
            })

    rng = random.Random(42)
    out_dir = Path(out_dir)
    selection_summary = {}

    sort_keys = {
        "false_positive": "max_fp_conf",
        "false_negative": "n_fn",
        "true_positive": "max_tp_conf",
        "true_negative": "max_fp_conf",
    }
    for cat in ("false_positive", "false_negative", "true_positive", "true_negative"):
        candidates = cat_candidates.get(cat, [])
        if not candidates:
            selection_summary[cat] = {"available": 0, "selected": []}
            continue
        chosen = select_examples(candidates, sort_keys[cat], k_top=per_cat // 2, k_random=per_cat - per_cat // 2, rng=rng)
        cat_dir = out_dir / cat
        if cat_dir.exists():
            for f in cat_dir.glob("*.jpg"):
                f.unlink()
            for f in cat_dir.glob("*.png"):
                f.unlink()
        cat_dir.mkdir(parents=True, exist_ok=True)
        for c in chosen:
            target = cat_dir / Path(c["image_path"]).name
            render_overlay(c["image_path"], c["gt_boxes"], c["pred_boxes"], tau, target)
        selection_summary[cat] = {
            "available": len(candidates),
            "selected": [Path(c["image_path"]).name for c in chosen],
        }

    summary_path = out_dir / "selection.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"tau": tau, "per_category": per_cat, "categories": selection_summary}, f, indent=2)
    click.echo(str(out_dir))


if __name__ == "__main__":
    main()
