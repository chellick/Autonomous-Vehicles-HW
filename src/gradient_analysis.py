import random
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image

from src.box_metrics import iou_xyxy
from src.model import get_model
from src.utils import ROOT, write_json


IMGSZ = 640
DEVICE = "cpu"


def preprocess(image_path, imgsz=IMGSZ, device=DEVICE):
    img = Image.open(image_path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(x):
    arr = (x.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def raw_predictions(model, x):
    out = model.model(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def confidence_objective(raw, top_k=10):
    nc = raw.shape[1] - 4
    cls = raw[:, 4:4 + nc, :]
    if cls.max() > 1.0 + 1e-3:
        cls = cls.sigmoid()
    max_per_anchor = cls.max(dim=1)[0].flatten()
    k = min(top_k, max_per_anchor.numel())
    top_vals, _ = max_per_anchor.topk(k)
    return top_vals.sum()


def gradient_stats(model, image_path, top_k=10):
    x = preprocess(image_path)
    x.requires_grad_(True)
    model.model.eval()
    with torch.enable_grad():
        raw = raw_predictions(model, x)
        objective = confidence_objective(raw, top_k=top_k)
        objective.backward()
    grad = x.grad.detach()
    return {
        "image_path": str(image_path),
        "top_k": top_k,
        "objective": float(objective.item()),
        "grad_l2": float(grad.flatten().norm(p=2).item()),
        "grad_linf": float(grad.abs().max().item()),
        "grad_l1_per_pixel": float(grad.abs().mean().item()),
    }


def predict_pil(model, image, conf=0.25):
    result = model.predict(image, imgsz=IMGSZ, conf=conf, verbose=False)[0]
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


def fgsm_perturb(model, image_path, epsilon):
    x = preprocess(image_path)
    x.requires_grad_(True)
    model.model.eval()
    with torch.enable_grad():
        raw = raw_predictions(model, x)
        objective = confidence_objective(raw)
        objective.backward()
    sign = x.grad.sign()
    adv = (x.detach() - epsilon * sign).clamp(0, 1)
    return tensor_to_pil(adv)


def fgsm_min_flip(model, image_path, epsilons):
    base_image = Image.open(image_path).convert("RGB").resize((IMGSZ, IMGSZ), Image.BILINEAR)
    clean = predict_pil(model, base_image)
    n_clean = len(clean)
    if n_clean == 0:
        return {"n_clean": 0, "min_flip_epsilon": None, "preservation": []}
    record = []
    min_flip = None
    for eps in epsilons:
        adv = fgsm_perturb(model, image_path, eps)
        preds = predict_pil(model, adv)
        kept = preserved_count(clean, preds)
        record.append({"epsilon": float(eps), "preserved": kept, "n_clean": n_clean})
        if kept < n_clean and min_flip is None:
            min_flip = float(eps)
    return {"n_clean": n_clean, "min_flip_epsilon": min_flip, "preservation": record}


@click.command()
@click.option("--model", "model_path", default="models/best_yolo_auto_trasnport.pt", type=click.Path(path_type=Path))
@click.option("--subset", default="splits/self_driving_car/test.txt", type=click.Path(path_type=Path))
@click.option("--n", default=50, type=int)
@click.option("--top-k", default=10, type=int)
@click.option("--name", default=None)
def main(model_path, subset, n, top_k, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gradient_{stamp}" + (f"_{name}" if name else "")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    paths_all = [line.strip() for line in Path(subset).read_text().splitlines() if line.strip()]
    rng = random.Random(42)
    paths = sorted(rng.sample(paths_all, min(n, len(paths_all))))

    model = get_model(model_path)

    epsilons = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255]

    gradient_records = []
    fgsm_records = []
    for i, path in enumerate(paths, 1):
        click.echo(f"[{i}/{len(paths)}] {Path(path).name}")
        gradient_records.append(gradient_stats(model, path, top_k=top_k))
        fgsm_records.append({"image_path": str(path), **fgsm_min_flip(model, path, epsilons)})

    grad_l2 = [r["grad_l2"] for r in gradient_records]
    grad_linf = [r["grad_linf"] for r in gradient_records]
    grad_per_pixel = [r["grad_l1_per_pixel"] for r in gradient_records]
    flips = [r["min_flip_epsilon"] for r in fgsm_records if r["min_flip_epsilon"] is not None]
    n_with_clean = sum(1 for r in fgsm_records if r["n_clean"] > 0)

    preservation_by_eps = {f"{eps:.5f}": [] for eps in epsilons}
    for r in fgsm_records:
        if r["n_clean"] == 0:
            continue
        for entry in r["preservation"]:
            rate = entry["preserved"] / entry["n_clean"]
            preservation_by_eps[f"{entry['epsilon']:.5f}"].append(rate)

    summary = {
        "n_images": len(paths),
        "top_k": top_k,
        "epsilons": epsilons,
        "gradient_norm": {
            "l2_median": float(np.median(grad_l2)),
            "l2_p25": float(np.percentile(grad_l2, 25)),
            "l2_p75": float(np.percentile(grad_l2, 75)),
            "linf_median": float(np.median(grad_linf)),
            "per_pixel_median": float(np.median(grad_per_pixel)),
        },
        "fgsm": {
            "n_with_clean_detections": n_with_clean,
            "fraction_flipped_within_grid": len(flips) / max(1, n_with_clean),
            "min_flip_epsilon_median": float(np.median(flips)) if flips else None,
            "min_flip_epsilon_p25": float(np.percentile(flips, 25)) if flips else None,
            "min_flip_epsilon_p75": float(np.percentile(flips, 75)) if flips else None,
            "preservation_rate_by_epsilon": {
                eps: float(np.mean(rates)) if rates else None for eps, rates in preservation_by_eps.items()
            },
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "gradient_records.json", gradient_records)
    write_json(run_dir / "fgsm_records.json", fgsm_records)
    click.echo(str(run_dir))


if __name__ == "__main__":
    main()
