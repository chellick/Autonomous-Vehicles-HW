import click
import copy
import random
from pathlib import Path

import yaml

from src.model import get_model
from src.mlflow_logger import log_metrics, log_params, mlflow_run
from src.utils import (
    ROOT,
    extract_metrics,
    make_run_name,
    resolve_path,
    source_key,
    write_json,
    write_yaml,
)


def resolve_model_weights(value):
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    if len(path.parts) > 1 or str(value).startswith("."):
        return str(resolve_path(value))
    return str(value)


def load_config(config_path):
    config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = copy.deepcopy(config)
    config["config_path"] = str(config_path)
    config.setdefault("name", None)
    config.setdefault("seed", 42)

    project = config.setdefault("project", {})
    project["runs_dir"] = str(resolve_path(project.get("runs_dir", "runs")))
    project["models_dir"] = str(resolve_path(project.get("models_dir", "models")))

    data = config.setdefault("data", {})
    dataset_root = resolve_path(data.get("dataset_root", "data/Self-Driving-Car-3"))
    data["dataset_root"] = str(dataset_root)
    data["images_dir"] = str(resolve_path(data.get("images_dir", "export/images"), dataset_root))
    data["labels_dir"] = str(resolve_path(data.get("labels_dir", "export/labels"), dataset_root))
    data["split_dir"] = str(resolve_path(data.get("split_dir", "splits/self_driving_car")))
    data["meta_file"] = str(resolve_path(data.get("meta_file", "data.yaml"), dataset_root))
    data["split_seed"] = data.get("split_seed", config["seed"])
    data["train_ratio"] = data.get("train_ratio", 0.7)
    data["val_ratio"] = data.get("val_ratio", 0.2)
    data["test_ratio"] = data.get("test_ratio", 0.1)
    data["split_unit"] = data.get("split_unit", "image")

    model = config.setdefault("model", {})
    model["weights"] = resolve_model_weights(model.get("weights", "yolov8n.pt"))
    model["weights_dir"] = str(resolve_path(model.get("weights_dir", project["models_dir"])))

    train = config.setdefault("train", {})
    train["epochs"] = train.get("epochs", 30)
    train["imgsz"] = train.get("imgsz", 640)
    train["batch"] = train.get("batch", 64)
    train["device"] = train.get("device")
    train["workers"] = train.get("workers", 4)
    train["exist_ok"] = train.get("exist_ok", True)
    train["deterministic"] = train.get("deterministic", True)
    train["optimizer"] = train.get("optimizer", "AdamW")
    train["lr0"] = train.get("lr0", 1e-3)
    train["lrf"] = train.get("lrf", 1e-2)
    train["weight_decay"] = train.get("weight_decay", 5e-4)
    train["warmup_epochs"] = train.get("warmup_epochs", 3)
    train["close_mosaic"] = train.get("close_mosaic", 10)
    train["cos_lr"] = train.get("cos_lr", True)
    train["patience"] = train.get("patience", 15)
    train["box"] = train.get("box", 7.5)
    train["cls"] = train.get("cls", 0.5)
    train["dfl"] = train.get("dfl", 1.5)
    train["hsv_h"] = train.get("hsv_h", 0.015)
    train["hsv_s"] = train.get("hsv_s", 0.7)
    train["hsv_v"] = train.get("hsv_v", 0.4)
    train["translate"] = train.get("translate", 0.1)
    train["scale"] = train.get("scale", 0.5)
    train["fliplr"] = train.get("fliplr", 0.5)
    train["mosaic"] = train.get("mosaic", 1.0)
    train["mixup"] = train.get("mixup", 0.0)
    train["amp"] = train.get("amp", True)
    train["plots"] = train.get("plots", True)
    train["save"] = train.get("save", True)
    train["save_period"] = train.get("save_period", 5)
    train["verbose"] = train.get("verbose", True)

    eval_config = config.setdefault("eval", {})
    eval_config["imgsz"] = eval_config.get("imgsz", 640)
    eval_config["batch"] = eval_config.get("batch", 16)
    eval_config["conf"] = eval_config.get("conf", 0.001)
    eval_config["iou"] = eval_config.get("iou", 0.6)
    eval_config["device"] = eval_config.get("device")
    eval_config["workers"] = eval_config.get("workers", 2)
    eval_config["save_json"] = eval_config.get("save_json", False)
    eval_config["plots"] = eval_config.get("plots", False)
    eval_config["verbose"] = eval_config.get("verbose", True)

    return config


def get_data_yaml_path(config):
    return Path(config["data"]["split_dir"]) / "data.yaml"


def get_split_summary_path(config):
    return Path(config["data"]["split_dir"]) / "split_summary.json"


def prepare_splits(config):
    data = config["data"]
    dataset_root = Path(data["dataset_root"])
    images_dir = Path(data["images_dir"])
    labels_dir = Path(data["labels_dir"])
    split_dir = Path(data["split_dir"])
    meta_file = Path(data["meta_file"])
    split_unit = data.get("split_unit", "image")

    if split_unit not in ("image", "source_key"):
        raise ValueError(f"Unknown split_unit: {split_unit!r}; expected 'image' or 'source_key'")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Dataset metadata file does not exist: {meta_file}")

    total_ratio = data["train_ratio"] + data["val_ratio"] + data["test_ratio"]
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    with meta_file.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    image_paths = sorted([path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if split_unit == "image":
        items = list(image_paths)
        random.Random(data["split_seed"]).shuffle(items)
        total_count = len(items)
        train_count = int(total_count * data["train_ratio"])
        val_count = int(total_count * data["val_ratio"])
        splits = {
            "train": items[:train_count],
            "val": items[train_count:train_count + val_count],
            "test": items[train_count + val_count:],
        }
    else:
        groups = {}
        for path in image_paths:
            groups.setdefault(source_key(path), []).append(path)
        group_keys = sorted(groups.keys())
        random.Random(data["split_seed"]).shuffle(group_keys)
        total_groups = len(group_keys)
        train_groups = int(total_groups * data["train_ratio"])
        val_groups = int(total_groups * data["val_ratio"])
        split_keys = {
            "train": group_keys[:train_groups],
            "val": group_keys[train_groups:train_groups + val_groups],
            "test": group_keys[train_groups + val_groups:],
        }
        splits = {
            name: sorted(path for key in keys for path in groups[key])
            for name, keys in split_keys.items()
        }

    split_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_images in splits.items():
        split_path = split_dir / f"{split_name}.txt"
        content = "\n".join(str(path.resolve()) for path in split_images)
        if content:
            content += "\n"
        split_path.write_text(content, encoding="utf-8")

    data_yaml_path = get_data_yaml_path(config)
    data_config = {
        "path": str(dataset_root.resolve()),
        "train": str((split_dir / "train.txt").resolve()),
        "val": str((split_dir / "val.txt").resolve()),
        "test": str((split_dir / "test.txt").resolve()),
        "nc": meta["nc"],
        "names": meta["names"],
    }

    data_yaml_path.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "split_dir": str(split_dir.resolve()),
        "split_unit": split_unit,
        "split_seed": data["split_seed"],
        "counts": {name: len(values) for name, values in splits.items()},
    }

    write_json(get_split_summary_path(config), summary)


def build_eval_args(config, run_dir, split):
    eval_config = config["eval"]
    args = {
        "data": str(get_data_yaml_path(config)),
        "split": split,
        "imgsz": eval_config["imgsz"],
        "batch": eval_config["batch"],
        "project": str(run_dir),
        "name": split,
        "exist_ok": True,
        "conf": eval_config["conf"],
        "iou": eval_config["iou"],
        "workers": eval_config["workers"],
        "save_json": eval_config["save_json"],
        "plots": eval_config["plots"],
        "verbose": eval_config["verbose"],
    }

    if eval_config["device"] not in (None, "", []):
        args["device"] = eval_config["device"]

    return args


def run_training(config, config_path=None, name_override=None):
    project_dir = Path(config["project"]["runs_dir"])
    run_name = make_run_name(name_override if name_override is not None else config.get("name"))

    prepare_splits(config)

    model_config = config["model"]
    train_config = config["train"]
    data_yaml_path = get_data_yaml_path(config)

    model = get_model(
        model_config["weights"],
        output_dir=model_config["weights_dir"],
    )

    train_args = {
        "data": str(data_yaml_path),
        "epochs": train_config["epochs"],
        "imgsz": train_config["imgsz"],
        "batch": train_config["batch"],
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": train_config["exist_ok"],
        "seed": config["seed"],
        "deterministic": train_config["deterministic"],
        "optimizer": train_config["optimizer"],
        "lr0": train_config["lr0"],
        "lrf": train_config["lrf"],
        "weight_decay": train_config["weight_decay"],
        "warmup_epochs": train_config["warmup_epochs"],
        "close_mosaic": train_config["close_mosaic"],
        "cos_lr": train_config["cos_lr"],
        "patience": train_config["patience"],
        "workers": train_config["workers"],
        "box": train_config["box"],
        "cls": train_config["cls"],
        "dfl": train_config["dfl"],
        "hsv_h": train_config["hsv_h"],
        "hsv_s": train_config["hsv_s"],
        "hsv_v": train_config["hsv_v"],
        "translate": train_config["translate"],
        "scale": train_config["scale"],
        "fliplr": train_config["fliplr"],
        "mosaic": train_config["mosaic"],
        "mixup": train_config["mixup"],
        "amp": train_config["amp"],
        "plots": train_config["plots"],
        "save": train_config["save"],
        "save_period": train_config["save_period"],
        "verbose": train_config["verbose"],
    }

    if train_config["device"] not in (None, "", []):
        train_args["device"] = train_config["device"]

    mlflow_tags = {"task": "train", "model_weights": str(model_config["weights"])}
    with mlflow_run(experiment="train", run_name=run_name, tags=mlflow_tags):
        log_params({
            "seed": config["seed"],
            "model": model_config["weights"],
            "train": train_config,
            "eval": config["eval"],
        })

        train_results = model.train(**train_args)
        run_dir = Path(getattr(train_results, "save_dir", project_dir / run_name))
        best_weights = run_dir / "weights" / "best.pt"
        last_weights = run_dir / "weights" / "last.pt"
        checkpoint_path = best_weights if best_weights.exists() else last_weights

        if checkpoint_path.exists():
            best_model = get_model(str(checkpoint_path))
        else:
            best_model = model

        val_results = best_model.val(**build_eval_args(config, run_dir, "val"))
        test_results = best_model.val(**build_eval_args(config, run_dir, "test"))

        train_metrics = extract_metrics(train_results)
        val_metrics = extract_metrics(val_results)
        test_metrics = extract_metrics(test_results)

        if train_metrics.get("results_dict"):
            log_metrics({"train." + k: v for k, v in train_metrics["results_dict"].items()})
        if val_metrics.get("results_dict"):
            log_metrics({"val." + k: v for k, v in val_metrics["results_dict"].items()})
        if test_metrics.get("results_dict"):
            log_metrics({"test." + k: v for k, v in test_metrics["results_dict"].items()})

        snapshot = copy.deepcopy(config)
        snapshot["run_name"] = run_name
        config_snapshot_path = run_dir / "config.yaml"
        write_yaml(config_snapshot_path, snapshot)

        result = {
            "config_path": str(Path(config_path).resolve()) if config_path else None,
            "config_snapshot": str(config_snapshot_path.resolve()),
            "dataset_root": config["data"]["dataset_root"],
            "data_yaml": str(data_yaml_path.resolve()),
            "split_summary": str(get_split_summary_path(config).resolve()),
            "run_name": run_name,
            "run_dir": str(run_dir.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path.exists() else None,
            "best_weights": str(best_weights.resolve()),
            "last_weights": str(last_weights.resolve()),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        result_path = run_dir / "result.json"
        write_json(result_path, result)

    return result_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=Path("configs/baseline.yaml"), show_default=True)
@click.option("--name", default=None)
def main(config, name):
    config = Path(config)
    loaded_config = load_config(config)
    result_path = run_training(loaded_config, config, name)
    click.echo(result_path)


if __name__ == "__main__":
    main()
