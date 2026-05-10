# Download Dataset

Install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create `data/.env`:

```text
ROBOFLOW_API_KEY=your_key
```

Download the dataset:

```bash
python data/download.py
```

Expected dataset path after download:

```text
data/Self-Driving-Car-3/export/images
data/Self-Driving-Car-3/export/labels
```

# Run Baseline Training

Train the baseline from `configs/baseline.yaml`:

```bash
python -m src.train --config configs/baseline.yaml
```

Train with a custom run name:

```bash
python -m src.train --config configs/baseline.yaml --name baseline
```

Training creates a run directory in `runs/`:

```text
runs/YYYYMMDD_HHMMSS/
runs/YYYYMMDD_HHMMSS_name/
```

After training, the run directory should contain baseline artifacts such as:

```text
config.yaml
result.json
weights/best.pt
weights/last.pt
```

An example saved training run is available in `runs.example/yolov8n_baseline` with `args.yaml`, `results.csv`, and `weights/`.

# Run Evaluation

`src.train` already runs validation and test evaluation after training and stores the metrics in `result.json`.

You can also run evaluation separately for a saved checkpoint:

```bash
python -m src.eval \
  --config configs/baseline.yaml \
  --model runs/<run_name>/weights/best.pt
```

Run evaluation on only part of the test split and save ground truth plus predictions:

```bash
python -m src.eval \
  --config configs/baseline.yaml \
  --model runs/<run_name>/weights/best.pt \
  --split test \
  --limit 64 \
  --save-gt-pred
```

This writes `result.json`, `<split>_subset.txt`, `gt.json`, and `pred.json` into the run directory.

# Run Inference

With config:

```bash
python -m src.inference --config configs/baseline.yaml
```

Direct run with local model:

```bash
python -m src.inference \
  --model models/best_yolo_auto_trasnport.pt \
  --source data/Self-Driving-Car-3/export/images
```

Results are written to `runs/`.
