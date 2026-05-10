import json
import random
from datetime import datetime
from pathlib import Path

import click
from PIL import Image

from src.box_metrics import aggregate, match_image, precision_recall_f1
from src.model import get_model
from src.transforms import TRANSFORMS
from src.utils import ROOT, write_json


def load_subset(subset_file, n=None, seed=42):
    paths = [line.strip() for line in Path(subset_file).read_text().splitlines() if line.strip()]
    if n is None or n >= len(paths):
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n))


def load_gt(gt_path, image_paths):
    data = json.load(open(gt_path))
    by_path = {r["image_path"]: r for r in data["images"]}
    return {p: by_path[p]["boxes"] for p in image_paths if p in by_path}


def predict_transformed(model, image_paths, transform_fn, alpha, conf=0.001, imgsz=640):
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(transform_fn(img, alpha))
    results = model.predict(images, imgsz=imgsz, conf=conf, verbose=False)
    out = []
    for path, result in zip(image_paths, results):
        boxes = []
        names = result.names
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().tolist()
            conf_l = result.boxes.conf.cpu().tolist()
            cls_l = result.boxes.cls.cpu().tolist()
            for x, c, k in zip(xyxy, conf_l, cls_l):
                k = int(k)
                cn = names[k] if isinstance(names, (list, tuple)) else names.get(k, str(k))
                boxes.append({
                    "class_id": k,
                    "class_name": cn,
                    "confidence": round(float(c), 6),
                    "xyxy": [round(float(v), 3) for v in x],
                })
        out.append({"image_path": str(path), "boxes": boxes})
    return out


@click.command()
@click.option("--model", "model_path", default="models/best_yolo_auto_trasnport.pt", type=click.Path(path_type=Path))
@click.option("--subset", default="splits/self_driving_car/test.txt", type=click.Path(path_type=Path))
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--n", default=200, type=int)
@click.option("--transform", "transform_filter", default=None)
@click.option("--name", default=None)
def main(model_path, subset, gt_path, n, transform_filter, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"stress_{stamp}" + (f"_{name}" if name else "")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = load_subset(subset, n=n)
    gt_by_path = load_gt(gt_path, paths)
    paths = [p for p in paths if p in gt_by_path]

    model = get_model(model_path)

    transforms_to_run = TRANSFORMS if transform_filter is None else {transform_filter: TRANSFORMS[transform_filter]}

    summary = {"run_dir": str(run_dir), "n_images": len(paths), "subset": str(subset), "results": []}

    for tname, (tfn, alphas, unit) in transforms_to_run.items():
        for alpha in alphas:
            click.echo(f"{tname} alpha={alpha}")
            preds = predict_transformed(model, paths, tfn, alpha)

            pairs_all = []
            for pred in preds:
                gt_boxes = gt_by_path[pred["image_path"]]
                pairs = match_image(gt_boxes, pred["boxes"], conf_thresh=0.25)
                pairs_all.append({"image_path": pred["image_path"], "pairs": pairs})

            counts = aggregate(pairs_all)
            metrics = precision_recall_f1(counts)

            tag = f"{tname}_a{alpha}"
            write_json(run_dir / f"{tag}.json", {
                "transform": tname,
                "alpha": alpha,
                "unit": unit,
                "n_images": len(paths),
                "counts": counts,
                "metrics": metrics,
                "predictions": preds,
                "pairs": pairs_all,
            })

            summary["results"].append({
                "transform": tname,
                "alpha": alpha,
                "unit": unit,
                "counts": counts,
                "metrics": metrics,
                "file": f"{tag}.json",
            })

    write_json(run_dir / "summary.json", summary)
    click.echo(str(run_dir))


if __name__ == "__main__":
    main()
