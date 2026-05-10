import click
import tempfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from src.model import get_model
from src.mlflow_logger import log_metrics, log_params, mlflow_run
from src.utils import (
    ROOT,
    extract_metrics,
    make_run_name,
    pick,
    resolve_path,
    write_json,
    write_yaml,
)

DEFAULT_LOCAL_MODEL = ROOT / "models" / "best_yolo_auto_trasnport.pt"
DEFAULT_MODEL = DEFAULT_LOCAL_MODEL
DEFAULT_DATA = ROOT / "splits" / "self_driving_car" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path):
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    project = config.setdefault("project", {})
    project["runs_dir"] = str(resolve_path(project.get("runs_dir", "runs")))

    model = config.setdefault("model", {})
    model["default_model"] = str(resolve_path(model.get("default_model", DEFAULT_LOCAL_MODEL)))

    data = config.setdefault("data", {})
    data["split_dir"] = str(resolve_path(data.get("split_dir", "splits/self_driving_car")))

    eval_config = config.setdefault("eval", {})
    eval_config["data"] = str(resolve_path(eval_config.get("data", Path(data["split_dir"]) / "data.yaml")))
    eval_config["split"] = eval_config.get("split", "test")
    eval_config["imgsz"] = eval_config.get("imgsz", 640)
    eval_config["batch"] = eval_config.get("batch", 16)
    eval_config["conf"] = eval_config.get("conf", 0.001)
    eval_config["iou"] = eval_config.get("iou", 0.6)
    eval_config["device"] = eval_config.get("device")
    eval_config["workers"] = eval_config.get("workers", 2)
    eval_config["exist_ok"] = eval_config.get("exist_ok", True)
    eval_config["save_json"] = eval_config.get("save_json", False)
    eval_config["plots"] = eval_config.get("plots", False)
    eval_config["verbose"] = eval_config.get("verbose", True)
    eval_config["limit"] = eval_config.get("limit")
    eval_config["offset"] = eval_config.get("offset", 0)
    eval_config["save_gt_pred"] = eval_config.get("save_gt_pred", False)
    eval_config["save_images"] = eval_config.get("save_images", False)
    eval_config["export_conf"] = eval_config.get("export_conf", 0.25)
    eval_config["export_iou"] = eval_config.get("export_iou", eval_config["iou"])

    config.setdefault("name", None)
    return config


def load_data_config(data_path):
    data_path = Path(data_path).resolve()
    with data_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_root(data_config, data_path):
    data_path = Path(data_path).resolve()
    dataset_root = data_config.get("path")
    if dataset_root is None:
        return data_path.parent.resolve()
    return resolve_path(dataset_root, data_path.parent)


def resolve_data_entry(data_config, data_path, value):
    return resolve_path(value, resolve_dataset_root(data_config, data_path))


def load_split_images(data_path, split):
    data_config = load_data_config(data_path)

    if split not in data_config:
        raise KeyError(f"Split '{split}' is not defined in {data_path}")

    split_source = resolve_data_entry(data_config, data_path, data_config[split])

    if split_source.is_file():
        image_paths = [Path(line).resolve() for line in split_source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return image_paths, data_config

    if split_source.is_dir():
        image_paths = sorted(
            path.resolve()
            for path in split_source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        return image_paths, data_config

    raise FileNotFoundError(f"Split source does not exist: {split_source}")


def select_subset(image_paths, limit=None, offset=0):
    if offset < 0:
        raise ValueError("Offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("Limit must be positive")

    end = None if limit is None else offset + limit
    subset = image_paths[offset:end]

    if not subset:
        raise ValueError("Selected evaluation subset is empty")

    return subset


def materialize_subset(data_path, data_config, split, image_paths, output_dir):
    output_dir = Path(output_dir)
    subset_txt_path = output_dir / f"{split}.txt"
    subset_txt_path.write_text("".join(f"{path}\n" for path in image_paths), encoding="utf-8")

    subset_data_config = dict(data_config)
    subset_data_config["path"] = str(resolve_dataset_root(data_config, data_path))

    for key in ("train", "val", "test"):
        if key in data_config and key != split:
            subset_data_config[key] = str(resolve_data_entry(data_config, data_path, data_config[key]))

    subset_data_config[split] = str(subset_txt_path)

    subset_data_path = output_dir / "data.yaml"
    write_yaml(subset_data_path, subset_data_config)
    return subset_data_path, subset_txt_path, subset_data_config


def normalize_class_names(names):
    if isinstance(names, dict):
        payload = {}
        for key, value in names.items():
            try:
                payload[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return payload

    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}

    return {}


def get_class_name(class_names, class_id):
    return class_names.get(class_id, str(class_id))


def round_box(values, digits=3):
    return [round(float(value), digits) for value in values]


def round_normalized(values, digits=6):
    return [round(float(value), digits) for value in values]


def yolo_xywhn_to_xyxy(x_center, y_center, box_width, box_height, image_width, image_height):
    half_width = box_width / 2
    half_height = box_height / 2
    x1 = (x_center - half_width) * image_width
    y1 = (y_center - half_height) * image_height
    x2 = (x_center + half_width) * image_width
    y2 = (y_center + half_height) * image_height
    return [x1, y1, x2, y2]


def infer_label_path(image_path):
    image_path = Path(image_path).resolve()

    try:
        images_idx = image_path.parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")

    label_path = Path(*image_path.parts[:images_idx], "labels", *image_path.parts[images_idx + 1 :])
    return label_path.with_suffix(".txt")


def load_ground_truth(image_paths, class_names):
    records = []

    for image_path in image_paths:
        image_path = Path(image_path).resolve()
        label_path = infer_label_path(image_path)

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                x_center, y_center, box_width, box_height = map(float, parts[1:5])
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": get_class_name(class_names, class_id),
                        "xywhn": round_normalized([x_center, y_center, box_width, box_height]),
                        "xyxy": round_box(
                            yolo_xywhn_to_xyxy(
                                x_center,
                                y_center,
                                box_width,
                                box_height,
                                image_width,
                                image_height,
                            )
                        ),
                    }
                )

        records.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_width": image_width,
                "image_height": image_height,
                "boxes": boxes,
            }
        )

    return records


def to_prediction_boxes(result, class_names):
    result_class_names = normalize_class_names(getattr(result, "names", None)) or class_names
    raw_boxes = result.boxes

    if raw_boxes is None or len(raw_boxes) == 0:
        return []

    xyxy_list = raw_boxes.xyxy.cpu().tolist()
    conf_list = raw_boxes.conf.cpu().tolist()
    cls_list = raw_boxes.cls.cpu().tolist()

    boxes = []
    for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
        class_id = int(cls_id)
        boxes.append(
            {
                "class_id": class_id,
                "class_name": get_class_name(result_class_names, class_id),
                "confidence": round(float(conf), 6),
                "xyxy": round_box(xyxy),
            }
        )

    return boxes


def run_subset_prediction(model, subset_source, imgsz, conf, iou, device, class_names):
    predict_args = {
        "source": str(subset_source),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "save": False,
        "save_txt": False,
        "save_conf": False,
        "verbose": False,
    }

    if device is not None:
        predict_args["device"] = device

    results = model.predict(**predict_args)
    return [
        {
            "image_path": str(Path(result.path).resolve()),
            "boxes": to_prediction_boxes(result, class_names),
        }
        for result in results
    ]


def load_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_label(draw, x1, y1, text, color, font):
    if not text:
        return

    left = max(0, int(round(x1)))
    top = max(0, int(round(y1)) - 14)

    if font is not None:
        bbox = draw.textbbox((left, top), text, font=font)
    else:
        bbox = draw.textbbox((left, top), text)

    padded_bbox = (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1)
    draw.rectangle(padded_bbox, fill=color)
    draw.text((left, top), text, fill="white", font=font)


def annotate_image(image_path, gt_boxes=None, pred_boxes=None):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()

    for box in gt_boxes or []:
        x1, y1, x2, y2 = box["xyxy"]
        label = box["class_name"]
        draw.rectangle((x1, y1, x2, y2), outline="#20a050", width=3)
        draw_label(draw, x1, y1, f"GT {label}", "#20a050", font)

    for box in pred_boxes or []:
        x1, y1, x2, y2 = box["xyxy"]
        confidence = box.get("confidence")
        if confidence is None:
            label = box["class_name"]
        else:
            label = f'{box["class_name"]} {confidence:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline="#d04040", width=3)
        draw_label(draw, x1, y2 + 2, f"PRED {label}", "#d04040", font)

    return image


def save_annotated_images(run_dir, gt_records, pred_records):
    run_dir = Path(run_dir)
    images_root = run_dir / "images"
    gt_dir = images_root / "gt"
    pred_dir = images_root / "pred"
    overlay_dir = images_root / "overlay"

    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pred_by_image = {record["image_path"]: record["boxes"] for record in pred_records}

    for gt_record in gt_records:
        image_path = Path(gt_record["image_path"])
        filename = image_path.name
        gt_boxes = gt_record["boxes"]
        pred_boxes = pred_by_image.get(gt_record["image_path"], [])

        annotate_image(image_path, gt_boxes=gt_boxes).save(gt_dir / filename)
        annotate_image(image_path, pred_boxes=pred_boxes).save(pred_dir / filename)
        annotate_image(image_path, gt_boxes=gt_boxes, pred_boxes=pred_boxes).save(overlay_dir / filename)

    return {
        "images_dir": str(images_root.resolve()),
        "gt_images_dir": str(gt_dir.resolve()),
        "pred_images_dir": str(pred_dir.resolve()),
        "overlay_images_dir": str(overlay_dir.resolve()),
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=click.Path(path_type=Path), default=None)
@click.option("--data", type=click.Path(path_type=Path), default=None)
@click.option("--split", default=None)
@click.option("--project", type=click.Path(path_type=Path), default=None)
@click.option("--name", default=None)
@click.option("--imgsz", type=int, default=None)
@click.option("--batch", type=int, default=None)
@click.option("--conf", type=float, default=None)
@click.option("--iou", type=float, default=None)
@click.option("--device", default=None)
@click.option("--workers", type=int, default=None)
@click.option("--exist-ok/--no-exist-ok", default=None)
@click.option("--save-json/--no-save-json", default=None)
@click.option("--plots/--no-plots", default=None)
@click.option("--verbose/--no-verbose", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--offset", type=int, default=None)
@click.option("--save-gt-pred/--no-save-gt-pred", default=None)
@click.option("--save-images/--no-save-images", default=None)
@click.option("--export-conf", type=float, default=None)
@click.option("--export-iou", type=float, default=None)
def main(
    config,
    model,
    data,
    split,
    project,
    name,
    imgsz,
    batch,
    conf,
    iou,
    device,
    workers,
    exist_ok,
    save_json,
    plots,
    verbose,
    limit,
    offset,
    save_gt_pred,
    save_images,
    export_conf,
    export_iou,
):
    config = load_config(config) if config else None
    eval_config = config["eval"] if config else {}
    model_config = config["model"] if config else {}
    project_config = config["project"] if config else {}

    model_value = pick(model, model_config.get("default_model") if config else None, str(DEFAULT_MODEL))
    data_value = pick(data, eval_config.get("data") if config else None, str(DEFAULT_DATA))
    split_value = pick(split, eval_config.get("split") if config else None, "test")
    project_value = pick(project, project_config.get("runs_dir") if config else None, str(DEFAULT_PROJECT))
    imgsz_value = pick(imgsz, eval_config.get("imgsz") if config else None, 640)
    batch_value = pick(batch, eval_config.get("batch") if config else None, 16)
    conf_value = pick(conf, eval_config.get("conf") if config else None, 0.001)
    iou_value = pick(iou, eval_config.get("iou") if config else None, 0.6)
    device_value = pick(device, eval_config.get("device") if config else None, None)
    workers_value = pick(workers, eval_config.get("workers") if config else None, 2)
    limit_value = pick(limit, eval_config.get("limit") if config else None, None)
    offset_value = pick(offset, eval_config.get("offset") if config else None, 0)
    export_conf_value = pick(export_conf, eval_config.get("export_conf") if config else None, 0.25)
    export_iou_value = pick(export_iou, eval_config.get("export_iou") if config else None, iou_value)

    if config:
        name_value = make_run_name(name if name is not None else config.get("name"))
    else:
        name_value = name or "eval"

    exist_ok_value = pick(exist_ok, eval_config.get("exist_ok") if config else None, False)
    save_json_value = pick(save_json, eval_config.get("save_json") if config else None, False)
    plots_value = pick(plots, eval_config.get("plots") if config else None, False)
    verbose_value = pick(verbose, eval_config.get("verbose") if config else None, False)
    save_gt_pred_value = pick(save_gt_pred, eval_config.get("save_gt_pred") if config else None, False)
    save_images_value = pick(save_images, eval_config.get("save_images") if config else None, False)

    model_path = Path(model_value).resolve()
    data_path = Path(data_value).resolve()
    project_path = Path(project_value).resolve()

    image_paths, data_config = load_split_images(data_path, split_value)
    selected_images = select_subset(image_paths, limit=limit_value, offset=offset_value)

    model = get_model(model_path)
    class_names = normalize_class_names(data_config.get("names"))

    mlflow_tags = {"task": "eval", "split": split_value, "model": str(model_path)}
    with mlflow_run(experiment="eval", run_name=name_value, tags=mlflow_tags):
        log_params({
            "model": str(model_path),
            "data": str(data_path),
            "split": split_value,
            "imgsz": imgsz_value,
            "batch": batch_value,
            "conf": conf_value,
            "iou": iou_value,
            "limit": limit_value,
            "offset": offset_value,
            "num_images": len(selected_images),
        })

        with tempfile.TemporaryDirectory(prefix="eval_subset_") as tmp_dir:
            subset_data_path, subset_txt_path, subset_data_config = materialize_subset(
                data_path,
                data_config,
                split_value,
                selected_images,
                tmp_dir,
            )

            val_args = {
                "data": str(subset_data_path),
                "split": split_value,
                "imgsz": imgsz_value,
                "batch": batch_value,
                "project": str(project_path),
                "name": name_value,
                "exist_ok": exist_ok_value,
                "conf": conf_value,
                "iou": iou_value,
                "workers": workers_value,
                "save_json": save_json_value,
                "plots": plots_value,
                "verbose": verbose_value,
            }

            if device_value is not None:
                val_args["device"] = device_value

            metrics = model.val(**val_args)
            run_dir = Path(getattr(metrics, "save_dir", project_path / name_value))
            run_dir.mkdir(parents=True, exist_ok=True)

            subset_manifest_path = run_dir / f"{split_value}_subset.txt"
            subset_manifest_path.write_text(subset_txt_path.read_text(encoding="utf-8"), encoding="utf-8")

            gt_path = None
            pred_path = None
            image_artifacts = None
            gt_records = None
            pred_records = None

            if save_gt_pred_value or save_images_value:
                gt_records = load_ground_truth(selected_images, class_names)
                pred_records = run_subset_prediction(
                    model,
                    subset_txt_path,
                    imgsz_value,
                    export_conf_value,
                    export_iou_value,
                    device_value,
                    normalize_class_names(subset_data_config.get("names")) or class_names,
                )

            if save_gt_pred_value:
                gt_path = run_dir / "gt.json"
                pred_path = run_dir / "pred.json"

                gt_payload = {
                    "model_path": str(model_path),
                    "data": str(data_path),
                    "split": split_value,
                    "subset": {
                        "offset": offset_value,
                        "limit": limit_value,
                        "num_images": len(selected_images),
                        "manifest_path": str(subset_manifest_path.resolve()),
                    },
                    "images": gt_records,
                }

                pred_payload = {
                    "model_path": str(model_path),
                    "data": str(data_path),
                    "split": split_value,
                    "subset": {
                        "offset": offset_value,
                        "limit": limit_value,
                        "num_images": len(selected_images),
                        "manifest_path": str(subset_manifest_path.resolve()),
                    },
                    "images": pred_records,
                }

                write_json(gt_path, gt_payload)
                write_json(pred_path, pred_payload)

            if save_images_value:
                image_artifacts = save_annotated_images(run_dir, gt_records, pred_records)

            extracted = extract_metrics(metrics)
            if extracted.get("results_dict"):
                log_metrics(extracted["results_dict"])

            result = {
                "model_path": str(model_path),
                "data": str(data_path),
                "split": split_value,
                "run_dir": str(run_dir),
                "subset": {
                    "offset": offset_value,
                    "limit": limit_value,
                    "num_images": len(selected_images),
                    "manifest_path": str(subset_manifest_path.resolve()),
                },
                "export_settings": {
                    "conf": export_conf_value,
                    "iou": export_iou_value,
                },
                "gt_path": str(gt_path.resolve()) if gt_path is not None else None,
                "pred_path": str(pred_path.resolve()) if pred_path is not None else None,
                "image_artifacts": image_artifacts,
                "metrics": extracted,
            }

            result_path = run_dir / "result.json"
            write_json(result_path, result)

    click.echo(result_path)


if __name__ == "__main__":
    main()
