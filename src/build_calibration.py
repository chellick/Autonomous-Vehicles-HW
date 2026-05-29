import json
from pathlib import Path

import click

from src.analyze_stress import calibration, rematch_at_tau
from src.utils import write_json


@click.command()
@click.option("--gt", "gt_path", default="runs/20260426_135344_groupsplit_test/gt.json", type=click.Path(path_type=Path))
@click.option("--pred", "pred_path", default="runs/20260426_135344_groupsplit_test/pred.json", type=click.Path(path_type=Path))
@click.option("--out", default="reports/calibration.json", type=click.Path(path_type=Path))
@click.option("--n-bins", default=10, type=int)
@click.option("--tau", default=0.0, type=float, help="match cutoff (use 0 to include all preds)")
def main(gt_path, pred_path, out, n_bins, tau):
    gt_data = json.load(open(gt_path))
    pred_data = json.load(open(pred_path))
    gt_by_path = {r["image_path"]: r["boxes"] for r in gt_data["images"]}
    predictions = pred_data["images"]

    pairs_all = rematch_at_tau(predictions, gt_by_path, tau=tau)
    calib = calibration(pairs_all, n_bins=n_bins)

    confs = []
    correct = []
    n_positive_decisions_at_default = 0
    for entry in pairs_all:
        for p in entry["pairs"]:
            if p["kind"] == "fn" or p.get("confidence") is None:
                continue
            confs.append(p["confidence"])
            correct.append(1 if p["kind"] == "tp" else 0)
            if p["confidence"] >= 0.25:
                n_positive_decisions_at_default += 1

    mean_conf = sum(confs) / max(1, len(confs))
    n_positive = sum(correct)
    fraction_positive = n_positive / max(1, len(correct))

    payload = {
        "n_bins": n_bins,
        "ece": calib["ece"],
        "mean_confidence": mean_conf,
        "fraction_positive_decisions": fraction_positive,
        "n_predictions": len(confs),
        "n_positive_decisions_at_default_tau_0_25": n_positive_decisions_at_default,
        "tau_for_matching": tau,
        "bins": calib["bins"],
    }
    write_json(out, payload)
    click.echo(str(out))


if __name__ == "__main__":
    main()
