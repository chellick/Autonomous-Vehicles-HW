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
DEFAULT_DATA = ROOT / "splits" / "self_driving_car" / "data.yaml"
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


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def extract_metrics(metrics):
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return to_jsonable(metrics)

    payload = {}

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        payload["results_dict"] = to_jsonable(results_dict)

    speed = getattr(metrics, "speed", None)
    if speed is not None:
        payload["speed"] = to_jsonable(speed)

    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None:
        mean_results = getattr(box_metrics, "mean_results", None)
        if callable(mean_results):
            payload["box_mean_results"] = to_jsonable(mean_results())

        fitness = getattr(box_metrics, "fitness", None)
        if callable(fitness):
            payload["box_fitness"] = to_jsonable(fitness())

    return payload


def has_arg(flag):
    return flag in sys.argv or any(value.startswith(flag + "=") for value in sys.argv[1:])


def pick(flag, cli_value, config_value, default=None):
    if has_arg(flag):
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    eval_config = config["eval"] if config else {}
    model_config = config["model"] if config else {}
    project_config = config["project"] if config else {}

    model_value = pick("--model", args.model, model_config.get("default_model") if config else None, str(DEFAULT_MODEL))
    data_value = pick("--data", args.data, eval_config.get("data") if config else None, str(DEFAULT_DATA))
    split_value = pick("--split", args.split, eval_config.get("split") if config else None, "test")
    project_value = pick("--project", args.project, project_config.get("runs_dir") if config else None, str(DEFAULT_PROJECT))
    imgsz_value = pick("--imgsz", args.imgsz, eval_config.get("imgsz") if config else None, 640)
    batch_value = pick("--batch", args.batch, eval_config.get("batch") if config else None, 16)
    conf_value = pick("--conf", args.conf, eval_config.get("conf") if config else None, 0.001)
    iou_value = pick("--iou", args.iou, eval_config.get("iou") if config else None, 0.6)
    device_value = pick("--device", args.device, eval_config.get("device") if config else None, None)
    workers_value = pick("--workers", args.workers, eval_config.get("workers") if config else None, 2)

    if config:
        name_value = make_run_name(args.name if args.name is not None else config.get("name"))
    else:
        name_value = args.name or "eval"

    exist_ok_value = args.exist_ok or (eval_config.get("exist_ok", False) if config else False)
    save_json_value = args.save_json or (eval_config.get("save_json", False) if config else False)
    plots_value = args.plots or (eval_config.get("plots", False) if config else False)
    verbose_value = args.verbose or (eval_config.get("verbose", False) if config else False)

    model_path = Path(model_value).resolve()
    data_path = Path(data_value).resolve()
    project_path = Path(project_value).resolve()

    model = get_model(model_path)

    val_args = {
        "data": str(data_path),
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

    result = {
        "model_path": str(model_path),
        "data": str(data_path),
        "split": split_value,
        "run_dir": str(run_dir),
        "metrics": extract_metrics(metrics),
    }

    result_path = run_dir / "result.json"
    write_json(result_path, result)

    print(result_path)


if __name__ == "__main__":
    main()
