import json
import random
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from PIL import Image

from src.box_metrics import iou_xyxy
from src.model import get_model
from src.utils import ROOT, write_json


def predict_boxes(model, image, conf=0.25, imgsz=640):
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    names = result.names
    xyxy = result.boxes.xyxy.cpu().tolist()
    conf_l = result.boxes.conf.cpu().tolist()
    cls_l = result.boxes.cls.cpu().tolist()
    for x, c, k in zip(xyxy, conf_l, cls_l):
        k = int(k)
        cn = names[k] if isinstance(names, (list, tuple)) else names.get(k, str(k))
        boxes.append({"class_id": k, "class_name": cn, "confidence": float(c), "xyxy": x})
    return boxes


def preserved_count(clean, perturbed, iou_thresh=0.5):
    used = set()
    matched = 0
    for cb in clean:
        for j, pb in enumerate(perturbed):
            if j in used or pb["class_name"] != cb["class_name"]:
                continue
            if iou_xyxy(cb["xyxy"], pb["xyxy"]) >= iou_thresh:
                matched += 1
                used.add(j)
                break
    return matched


def perturb(image, sigma, rng):
    if sigma == 0:
        return image
    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def find_min_flip_sigma(model, image, clean_boxes, sigmas, n_trials, rng):
    if not clean_boxes:
        return None
    for sigma in sigmas:
        for _ in range(n_trials):
            perturbed = predict_boxes(model, perturb(image, sigma, rng))
            kept = preserved_count(clean_boxes, perturbed)
            if kept < len(clean_boxes):
                return float(sigma)
    return None


@click.command()
@click.option("--model", "model_path", default="models/best_yolo_auto_trasnport.pt", type=click.Path(path_type=Path))
@click.option("--subset", default="splits/self_driving_car/test.txt", type=click.Path(path_type=Path))
@click.option("--n", default=50, type=int)
@click.option("--trials", default=3, type=int)
@click.option("--name", default=None)
def main(model_path, subset, n, trials, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"adversarial_{stamp}" + (f"_{name}" if name else "")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    paths_all = [line.strip() for line in Path(subset).read_text().splitlines() if line.strip()]
    rng_sample = random.Random(42)
    paths = sorted(rng_sample.sample(paths_all, min(n, len(paths_all))))

    model = get_model(model_path)
    rng = np.random.default_rng(42)

    fine_sigmas = [0.5, 1, 2, 4, 8, 16]
    sigmas = [0, 1, 2, 4, 8, 16]

    per_image = []
    flip_sigmas = []
    preservation_curve = {sigma: [] for sigma in sigmas}

    for path in paths:
        image = Image.open(path).convert("RGB")
        clean = predict_boxes(model, image)
        n_clean = len(clean)
        record = {"image_path": str(path), "n_clean_boxes": n_clean, "per_sigma": []}

        for sigma in sigmas:
            preserved_runs = []
            for _ in range(trials):
                perturbed = predict_boxes(model, perturb(image, sigma, rng))
                preserved_runs.append(preserved_count(clean, perturbed))
            mean_preserved = float(np.mean(preserved_runs)) if preserved_runs else 0.0
            rate = mean_preserved / n_clean if n_clean > 0 else 1.0
            record["per_sigma"].append({"sigma": sigma, "preserved_mean": mean_preserved, "preservation_rate": rate})
            preservation_curve[sigma].append(rate)

        min_flip = find_min_flip_sigma(model, image, clean, fine_sigmas, trials, rng)
        record["min_flip_sigma"] = min_flip
        if min_flip is not None:
            flip_sigmas.append(min_flip)
        per_image.append(record)

    aggregate = {
        "n_images": len(paths),
        "trials_per_sigma": trials,
        "preservation_rate_by_sigma": {
            sigma: float(np.mean(vals)) if vals else None for sigma, vals in preservation_curve.items()
        },
        "min_flip_sigma": {
            "fraction_flipped_within_grid": len(flip_sigmas) / max(1, len(per_image)),
            "median": float(np.median(flip_sigmas)) if flip_sigmas else None,
            "p25": float(np.percentile(flip_sigmas, 25)) if flip_sigmas else None,
            "p75": float(np.percentile(flip_sigmas, 75)) if flip_sigmas else None,
        },
    }

    write_json(run_dir / "summary.json", aggregate)
    write_json(run_dir / "per_image.json", per_image)
    click.echo(str(run_dir))


if __name__ == "__main__":
    main()
