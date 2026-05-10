import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_port = os.environ.get("MLFLOW_PORT", "6777")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"http://localhost:{_port}")


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        return None


@contextmanager
def mlflow_run(experiment: str, run_name: str, tags: dict[str, str] | None = None):
    mlflow = _get_mlflow()
    if mlflow is None or os.environ.get("MLFLOW_DISABLE") == "1":
        yield None
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        run_ctx = mlflow.start_run(run_name=run_name, tags=tags or {})
    except Exception as exc:
        print(f"[mlflow_logger] tracking server unavailable ({exc}); continuing without mlflow")
        yield None
        return

    with run_ctx as run:
        yield run


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full_key, sep))
        else:
            out[full_key] = value
    return out


def log_params(params: dict):
    mlflow = _get_mlflow()
    if mlflow is None or not mlflow.active_run():
        return
    flat = _flatten(params)
    mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})


def _sanitize_key(key: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_\-\. :/]", "_", key)


def log_metrics(metrics: dict, step: int | None = None):
    mlflow = _get_mlflow()
    if mlflow is None or not mlflow.active_run():
        return
    flat = _flatten(metrics)
    numeric = {}
    for k, v in flat.items():
        try:
            numeric[_sanitize_key(k)] = float(v)
        except (TypeError, ValueError):
            pass
    if numeric:
        mlflow.log_metrics(numeric, step=step)


