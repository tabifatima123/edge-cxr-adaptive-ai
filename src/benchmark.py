"""
benchmark.py
------------
Main pipeline that runs every model on the dataset and saves a single
results.csv with all metrics for downstream analysis (Pareto, plots,
adaptive selector, report).

Run:
    python -m src.benchmark
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import ChestXrayDataset, get_default_transform
from .models import load_all_models
from .profiler import profile_model
from .energy import estimate_energy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"


def evaluate_accuracy(model, dataset, max_samples: int = 1000) -> dict:
    """
    Quick top-1 accuracy probe using available labeled samples.
    Note: lightweight encoders are pretrained on ImageNet, NOT chest X-rays.
    So this number reflects *embedding usefulness* on out-of-domain data,
    not real clinical accuracy. We report it transparently.

    Returns dict with: accuracy, num_evaluated, embedding_norm_mean.
    """
    model = model.eval()
    n = min(len(dataset), max_samples)
    if n == 0:
        return {"accuracy": float("nan"), "num_evaluated": 0, "embedding_norm_mean": 0.0}

    correct = 0
    norms = []
    with torch.no_grad():
        for i in range(n):
            x, y = dataset[i]
            x = x.unsqueeze(0)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            # Map predicted ImageNet-class index modulo num_classes (placeholder)
            mapped_pred = pred % len(dataset.classes)
            correct += int(mapped_pred == y)
            norms.append(float(torch.linalg.norm(logits)))

    return {
        "accuracy": round(correct / n, 4),
        "num_evaluated": n,
        "embedding_norm_mean": round(float(np.mean(norms)), 3),
    }


def run_full_benchmark(device_profile: str = "laptop_cpu") -> pd.DataFrame:
    """Profile every model, record everything to results.csv."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EDGE CXR BENCHMARK")
    print("=" * 60)

    dataset = ChestXrayDataset(transform=get_default_transform())
    models = load_all_models()
    rows = []

    for m in models:
        print(f"\n--- Profiling {m.name} ---")
        prof = profile_model(m)
        eng = estimate_energy(m, device_profile=device_profile)
        acc = evaluate_accuracy(m, dataset)

        row = {
            **prof.to_dict(),
            "energy_mj": eng.energy_per_inference_mj,
            "avg_power_w": eng.avg_power_w,
            "cpu_util_pct": eng.cpu_util_pct,
            "device_profile": device_profile,
            **acc,
        }
        rows.append(row)
        print(f"  latency={prof.latency_ms}ms  mem={prof.memory_mb}MB  "
              f"energy={eng.energy_per_inference_mj}mJ  acc={acc['accuracy']}")

    df = pd.DataFrame(rows)
    # Stable column order
    cols = [
        "model", "num_params_m", "model_size_mb",
        "latency_ms", "p95_latency_ms", "throughput_ips",
        "memory_mb", "energy_mj", "avg_power_w", "cpu_util_pct",
        "accuracy", "num_evaluated", "embedding_norm_mean",
        "weights_pretrained", "device_profile",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[benchmark] Saved results to {RESULTS_CSV}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_full_benchmark()
