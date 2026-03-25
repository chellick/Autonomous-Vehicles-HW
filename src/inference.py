import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.model import get_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MODEL = ROOT / "models" / "best_yolo_auto_trasnport.pt"
DEFAULT_MODEL = DEFAULT_LOCAL_MODEL
DEFAULT_SOURCE = ROOT / "splits" / "self_driving_car" / "test.txt"
DEFAULT_FALLBACK_SOURCE = ROOT / "data" / "Self-Driving-Car-3" / "export" / "images"
DEFAULT_PROJECT = ROOT / "runs"


def resolve_path(value, base=ROOT):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base) / path).resolve()


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


def make_run_name(name=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (name or "").strip().replace(" ", "_")
    if not name:
        return stamp
    return f"{stamp}_{name}"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def has_arg(flag):
    return flag in sys.argv or any(value.startswith(flag + "=") for value in sys.argv[1:])


def pick(flag, cli_value, config_value, default=None):
    if has_arg(flag):
        return cli_value
    if config_value is not None:
        return config_value
    return default


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    inference_config = config["inference"] if config else {}
    project_config = config["project"] if config else {}

    model_value = pick("--model", args.model, inference_config.get("model") if config else None, str(DEFAULT_MODEL))
    source_value = pick("--source", args.source, inference_config.get("source") if config else None, None)
    project_value = pick("--project", args.project, project_config.get("runs_dir") if config else None, str(DEFAULT_PROJECT))
    imgsz_value = pick("--imgsz", args.imgsz, inference_config.get("imgsz") if config else None, 640)
    conf_value = pick("--conf", args.conf, inference_config.get("conf") if config else None, 0.25)
    iou_value = pick("--iou", args.iou, inference_config.get("iou") if config else None, 0.7)
    device_value = pick("--device", args.device, inference_config.get("device") if config else None, None)

    if config:
        name_value = make_run_name(args.name if args.name is not None else config.get("name"))
    else:
        name_value = args.name or "inference"

    exist_ok_value = args.exist_ok or (inference_config.get("exist_ok", False) if config else False)
    save_txt_value = args.save_txt or (inference_config.get("save_txt", False) if config else False)
    save_conf_value = args.save_conf or (inference_config.get("save_conf", False) if config else False)
    verbose_value = args.verbose or (inference_config.get("verbose", False) if config else False)

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

    results = model.predict(**predict_args)

    if results:
        run_dir = Path(results[0].save_dir)
    else:
        run_dir = project_path / name_value
        run_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(results, model_path, source_path, run_dir)
    result_path = run_dir / "result.json"
    write_json(result_path, summary)

    print(result_path)


if __name__ == "__main__":
    main()
