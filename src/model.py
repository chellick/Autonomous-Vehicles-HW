from pathlib import Path

from ultralytics import YOLO, settings


def get_model(
    weights="yolov8n.pt",
    output_dir=None,
):
    if output_dir is not None:
        settings.update({"weights_dir": str(Path(output_dir).resolve())})

    weights_path = Path(weights)

    if weights_path.exists():
        model_source = weights_path.resolve()
    else:
        model_source = str(weights)

    return YOLO(str(model_source))
