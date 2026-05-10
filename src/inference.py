import click
from pathlib import Path

import yaml

from src.model import get_model
from src.mlflow_logger import log_metrics, log_params, mlflow_run
from src.utils import (
    ROOT,
    make_run_name,
    pick,
    resolve_path,
    write_json,
)

DEFAULT_LOCAL_MODEL = ROOT / "models" / "best_yolo_auto_trasnport.pt"
DEFAULT_MODEL = DEFAULT_LOCAL_MODEL
DEFAULT_SOURCE = ROOT / "splits" / "self_driving_car" / "test.txt"
DEFAULT_FALLBACK_SOURCE = ROOT / "data" / "Self-Driving-Car-3" / "export" / "images"
DEFAULT_PROJECT = ROOT / "runs"


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

    inference = config.setdefault("inference", {})
    inference["model"] = str(resolve_path(inference.get("model", model["default_model"])))
    inference["source"] = str(resolve_path(inference.get("source", Path(data["split_dir"]) / "test.txt")))
    inference["imgsz"] = inference.get("imgsz", 640)
    inference["conf"] = inference.get("conf", 0.25)
    inference["iou"] = inference.get("iou", 0.7)
    inference["device"] = inference.get("device")
    inference["exist_ok"] = inference.get("exist_ok", True)
    inference["save_txt"] = inference.get("save_txt", False)
    inference["save_conf"] = inference.get("save_conf", False)
    inference["verbose"] = inference.get("verbose", False)

    config.setdefault("name", None)
    return config


def get_source(source):
    if source is not None:
        return Path(source).resolve()
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE.resolve()
    return DEFAULT_FALLBACK_SOURCE.resolve()


def to_boxes(result):
    boxes = []
    names = result.names
    raw_boxes = result.boxes

    if raw_boxes is None or len(raw_boxes) == 0:
        return boxes

    xyxy_list = raw_boxes.xyxy.cpu().tolist()
    conf_list = raw_boxes.conf.cpu().tolist()
    cls_list = raw_boxes.cls.cpu().tolist()

    for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
        cls_id = int(cls_id)
        boxes.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id)),
                "confidence": round(float(conf), 6),
                "xyxy": [round(float(value), 3) for value in xyxy],
            }
        )

    return boxes


def build_summary(results, model_path, source_path, run_dir):
    predictions = []

    for result in results:
        predictions.append(
            {
                "image_path": str(Path(result.path).resolve()),
                "boxes": to_boxes(result),
            }
        )

    return {
        "model_path": str(model_path),
        "source": str(source_path),
        "run_dir": str(run_dir),
        "num_images": len(predictions),
        "predictions": predictions,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=click.Path(path_type=Path), default=None)
@click.option("--source", type=click.Path(path_type=Path), default=None)
@click.option("--project", type=click.Path(path_type=Path), default=None)
@click.option("--name", default=None)
@click.option("--imgsz", type=int, default=None)
@click.option("--conf", type=float, default=None)
@click.option("--iou", type=float, default=None)
@click.option("--device", default=None)
@click.option("--exist-ok/--no-exist-ok", default=None)
@click.option("--save-txt/--no-save-txt", default=None)
@click.option("--save-conf/--no-save-conf", default=None)
@click.option("--verbose/--no-verbose", default=None)
def main(config, model, source, project, name, imgsz, conf, iou, device, exist_ok, save_txt, save_conf, verbose):
    config = load_config(config) if config else None
    inference_config = config["inference"] if config else {}
    project_config = config["project"] if config else {}

    model_value = pick(model, inference_config.get("model") if config else None, str(DEFAULT_MODEL))
    source_value = pick(source, inference_config.get("source") if config else None, None)
    project_value = pick(project, project_config.get("runs_dir") if config else None, str(DEFAULT_PROJECT))
    imgsz_value = pick(imgsz, inference_config.get("imgsz") if config else None, 640)
    conf_value = pick(conf, inference_config.get("conf") if config else None, 0.25)
    iou_value = pick(iou, inference_config.get("iou") if config else None, 0.7)
    device_value = pick(device, inference_config.get("device") if config else None, None)

    if config:
        name_value = make_run_name(name if name is not None else config.get("name"))
    else:
        name_value = name or "inference"

    exist_ok_value = pick(exist_ok, inference_config.get("exist_ok") if config else None, False)
    save_txt_value = pick(save_txt, inference_config.get("save_txt") if config else None, False)
    save_conf_value = pick(save_conf, inference_config.get("save_conf") if config else None, False)
    verbose_value = pick(verbose, inference_config.get("verbose") if config else None, False)

    model_path = Path(model_value).resolve()
    source_path = get_source(source_value)
    project_path = Path(project_value).resolve()

    model = get_model(model_path)

    predict_args = {
        "source": str(source_path),
        "imgsz": imgsz_value,
        "conf": conf_value,
        "iou": iou_value,
        "project": str(project_path),
        "name": name_value,
        "exist_ok": exist_ok_value,
        "save": True,
        "save_txt": save_txt_value,
        "save_conf": save_conf_value,
        "verbose": verbose_value,
    }

    if device_value is not None:
        predict_args["device"] = device_value

    mlflow_tags = {"task": "inference", "model": str(model_path)}
    with mlflow_run(experiment="inference", run_name=name_value, tags=mlflow_tags):
        log_params({
            "model": str(model_path),
            "source": str(source_path),
            "imgsz": imgsz_value,
            "conf": conf_value,
            "iou": iou_value,
        })

        results = model.predict(**predict_args)

        if results:
            run_dir = Path(results[0].save_dir)
        else:
            run_dir = project_path / name_value
            run_dir.mkdir(parents=True, exist_ok=True)

        summary = build_summary(results, model_path, source_path, run_dir)
        result_path = run_dir / "result.json"
        write_json(result_path, summary)

        num_images = summary["num_images"]
        total_boxes = sum(len(p["boxes"]) for p in summary["predictions"])
        log_metrics({
            "num_images": float(num_images),
            "total_detections": float(total_boxes),
            "avg_detections_per_image": float(total_boxes / num_images) if num_images else 0.0,
        })

    click.echo(result_path)


if __name__ == "__main__":
    main()
