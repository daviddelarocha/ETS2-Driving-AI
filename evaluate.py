from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from driving_dataset import DrivingDataset, get_transform
from model import DrivingModel


TARGET_NAMES = ["steering", "throttle", "brake"]
DATASET_PATH = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación multimodal ETS2")
    # parser.add_argument("--csv", type=str, default=f"{DATASET_PATH}/samples.csv")
    # parser.add_argument("--images", type=str, default=f"{DATASET_PATH}/images")
    # parser.add_argument("--model", type=str, default="artifacts/best_model.pt")
    # parser.add_argument("--split", type=str, default="artifacts/data_split.json")
    # parser.add_argument("--output-dir", type=str, default="artifacts/eval")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    print("[Metrics] Computing regression metrics...")
    metrics: dict[str, dict[str, float]] = {}

    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 1e-8 and np.std(yp) > 1e-8 else float("nan")

        metrics[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "corr": corr,
        }

    abs_error = np.abs(y_true - y_pred)
    metrics["overall"] = {
        "mean_abs_error_all_outputs": float(abs_error.mean()),
        "median_abs_error_all_outputs": float(np.median(abs_error)),
    }

    return metrics


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    print("[Plots] Generating scatter plots...")
    for i, name in enumerate(TARGET_NAMES):
        print(f"[Plots] Creating scatter plot for: {name}")
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.3)
        min_v = min(y_true[:, i].min(), y_pred[:, i].min())
        max_v = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_v, max_v], [min_v, max_v])
        plt.xlabel(f"{name} real")
        plt.ylabel(f"{name} predicho")
        plt.title(f"Real vs Predicho: {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{name}.png", dpi=140)
        plt.close()


def save_error_summary(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    print("[Metrics] Saving error percentiles...")
    abs_error = np.abs(y_true - y_pred)
    summary = {}

    for i, name in enumerate(TARGET_NAMES):
        e = abs_error[:, i]
        summary[name] = {
            "p50_abs_error": float(np.percentile(e, 50)),
            "p90_abs_error": float(np.percentile(e, 90)),
            "p95_abs_error": float(np.percentile(e, 95)),
            "max_abs_error": float(np.max(e)),
        }

    (output_dir / "error_percentiles.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("[Start] Evaluation script started")
    args = parse_args()
    print(f"[Args] {vars(args)}")

    dataset_path = Path(args.dataset)
    artifacts_path = Path(args.artifacts)

    output_dir = artifacts_path / "eval"
    print("[Setup] Creating evaluation output directory...")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Artifacts] Loading model checkpoint...")
    ckpt = torch.load(artifacts_path / "best_model.pt", map_location="cpu")
    img_size = int(ckpt["img_size"])
    pretrained = bool(ckpt["pretrained"])

    print("[Artifacts] Loading data split...")
    split_info = json.loads((artifacts_path / "data_split.json").read_text(encoding="utf-8"))
    test_idx = split_info["test_indices"]

    print("[Setup] Building evaluation transforms...")
    transform = get_transform(img_size)

    print("[Data] Loading dataset...")
    dataset = DrivingDataset(
        csv_path=dataset_path / "samples.csv",
        images_root=dataset_path / "images",
        transform=transform,
        verify_images=True,
    )

    print("[Data] Building test subset...")
    test_ds = Subset(dataset, test_idx)
    print(f"[Data] Test samples: {len(test_ds)}")

    print("[Data] Building dataloader...")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("[Model] Initializing model...")
    model = DrivingModel(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[Inference] Running model on test set...")
    preds = []
    trues = []

    with torch.no_grad():
        for batch_idx, (images, features, targets) in enumerate(loader, start=1):
            outputs = model(images, features).cpu().numpy()
            preds.append(outputs)
            trues.append(targets.numpy())

            if batch_idx % 100 == 0:
                print(f"[Inference] Processed batch {batch_idx}")

    print("[Post] Concatenating predictions and targets...")
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    metrics = compute_metrics(y_true, y_pred)

    print("[Artifacts] Saving metrics.json...")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_scatter(y_true, y_pred, output_dir)
    save_error_summary(y_true, y_pred, output_dir)

    print("[Done] Evaluation completed")
    print(json.dumps(metrics, indent=2))
    print(f"[Done] Metrics saved to: {output_dir / 'metrics.json'}")
    print(f"[Done] Evaluation artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
    