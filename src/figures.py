import csv
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        return [
            {k: (float(v) if k != "threshold" or "." in v else float(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def precision_recall(rows, out_path):
    rec = [r["recall"] for r in rows]
    prec = [r["precision"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, marker=".", linewidth=1.2)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision vs recall (threshold sweep)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def roc(rows, out_path):
    fpr = [r["fpr"] for r in rows]
    tpr = [r["tpr"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, marker=".", linewidth=1.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("false positive rate (scene-level)")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC (threshold sweep)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def metrics_vs_threshold(rows, out_path):
    tau = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau, [r["precision"] for r in rows], label="precision", linewidth=1.4)
    ax.plot(tau, [r["recall"] for r in rows], label="recall", linewidth=1.4)
    ax.plot(tau, [r["f1"] for r in rows], label="F1", linewidth=1.4)
    ax.plot(tau, [r["accuracy"] for r in rows], label="accuracy", linewidth=1.0, linestyle="--")
    ax.set_xlabel("threshold tau")
    ax.set_ylabel("metric value")
    ax.set_title("Metrics vs threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def risk_vs_threshold(rows, out_path):
    tau = [r["threshold"] for r in rows]
    risk_u = [r["risk_uniform"] for r in rows]
    risk_pc = [r["risk_per_class"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(tau, risk_u, marker=".", linewidth=1.2, color="C0")
    axes[0].set_xlabel("threshold tau")
    axes[0].set_ylabel("risk")
    axes[0].set_title("Uniform cost (c_fn=5, c_fp=1)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    tau_min_u = tau[risk_u.index(min(risk_u))]
    axes[0].axvline(tau_min_u, color="C1", linestyle="--", linewidth=0.8)
    axes[0].text(tau_min_u, max(risk_u) * 0.95, f"  argmin = {tau_min_u}", color="C1")

    axes[1].plot(tau, risk_pc, marker=".", linewidth=1.2, color="C2")
    axes[1].set_xlabel("threshold tau")
    axes[1].set_ylabel("risk")
    axes[1].set_title("Per-class cost (pedestrian/biker=10, car/truck=5, TL=2-3)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    tau_min_pc = tau[risk_pc.index(min(risk_pc))]
    axes[1].axvline(tau_min_pc, color="C1", linestyle="--", linewidth=0.8)
    axes[1].text(tau_min_pc, max(risk_pc) * 0.95, f"  argmin = {tau_min_pc}", color="C1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reliability_diagram(calib_json_path, out_path):
    data = json.load(open(calib_json_path))
    bins = [b for b in data["bins"] if b["count"] > 0]
    if not bins:
        raise ValueError("no populated bins in calibration")
    centers = [(b["low"] + b["high"]) / 2 for b in bins]
    accs = [b["acc"] for b in bins]
    confs = [b["conf"] for b in bins]
    counts = [b["count"] for b in bins]
    width = (bins[0]["high"] - bins[0]["low"]) * 0.9

    fig, (ax_acc, ax_count) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax_acc.bar(centers, accs, width=width, alpha=0.7, edgecolor="black", label="accuracy")
    ax_acc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="perfect calibration")
    ax_acc.scatter(centers, confs, color="red", marker="x", s=50, zorder=3, label="mean confidence")
    ax_acc.set_ylabel("accuracy / mean confidence")
    ax_acc.set_xlim(0, 1)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"Reliability diagram (ECE = {data['ece']:.3f})")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="upper left")

    ax_count.bar(centers, counts, width=width, alpha=0.7, color="C2", edgecolor="black")
    ax_count.set_xlabel("predicted confidence")
    ax_count.set_ylabel("count")
    ax_count.set_xlim(0, 1)
    ax_count.set_yscale("log")
    ax_count.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--csv", "csv_path", default="reports/threshold_sweep.csv", type=click.Path(path_type=Path))
@click.option("--calibration", "calib_path", default="reports/calibration.json", type=click.Path(path_type=Path))
@click.option("--out-dir", default="reports/figures", type=click.Path(path_type=Path))
def main(csv_path, calib_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    precision_recall(rows, out_dir / "precision_recall_curve.png")
    roc(rows, out_dir / "roc_curve.png")
    metrics_vs_threshold(rows, out_dir / "metrics_vs_threshold.png")
    risk_vs_threshold(rows, out_dir / "risk_vs_threshold.png")

    if Path(calib_path).exists():
        reliability_diagram(calib_path, out_dir / "reliability_diagram.png")
        click.echo(f"reliability_diagram from {calib_path}")
    else:
        click.echo(f"skipped reliability_diagram (no {calib_path})")

    click.echo(str(out_dir))


if __name__ == "__main__":
    main()
