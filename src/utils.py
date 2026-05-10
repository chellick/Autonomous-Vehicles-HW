import json
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value, base=ROOT):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base) / path).resolve()


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


def write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
        except (ValueError, TypeError):
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


def source_key(path):
    return Path(path).name.split("_jpg.rf.")[0]


def pick(cli_value, config_value, default=None):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default
