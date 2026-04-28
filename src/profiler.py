"""
profiler.py
-----------
Latency + memory + throughput profiling for vision encoders on edge CPU.

Measures:
    - latency_ms        : avg forward-pass time per image
    - p95_latency_ms    : 95th percentile (matters for clinical SLO)
    - throughput_ips    : images/second under steady-state
    - memory_mb         : peak RSS delta during inference
    - model_size_mb     : on-disk size approximation (params * 4 bytes)

Uses psutil for cross-platform CPU memory measurement (works on Windows,
Linux, macOS, Raspberry Pi, Jetson). For GPU you would swap in
torch.cuda.max_memory_allocated.
"""

import gc
import time
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import psutil
import torch

from .models import EncoderWrapper


@dataclass
class ProfileResult:
    model: str
    latency_ms: float
    p95_latency_ms: float
    throughput_ips: float
    memory_mb: float
    model_size_mb: float
    num_params_m: float
    weights_pretrained: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def _get_process_rss_mb() -> float:
    """Current process resident set size in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def profile_model(
    model: EncoderWrapper,
    num_warmup: int = 5,
    num_iters: int = 30,
    batch_size: int = 1,
    device: str = "cpu",
) -> ProfileResult:
    """
    Profile a single model. Uses random inputs of the model's expected size.

    For chest X-ray edge inference, batch_size=1 is the realistic case (one
    patient image at a time on a Jetson / Pi).
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, model.img_size, model.img_size, device=device)

    # ---- Warmup (allocates buffers, JIT, etc.) ----
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # ---- Measure memory baseline ----
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline_mem = _get_process_rss_mb()

    # ---- Timed loop ----
    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    peak_mem = _get_process_rss_mb()
    mem_delta = max(0.0, peak_mem - baseline_mem)

    latencies = np.array(latencies)
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / avg_lat * batch_size

    num_params = model.num_params()
    model_size_mb = num_params * 4 / (1024 * 1024)  # fp32 estimate

    return ProfileResult(
        model=model.name,
        latency_ms=round(avg_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        throughput_ips=round(throughput, 2),
        memory_mb=round(mem_delta, 2),
        model_size_mb=round(model_size_mb, 2),
        num_params_m=round(num_params / 1e6, 2),
        weights_pretrained=model.weights_loaded,
    )


if __name__ == "__main__":
    from .models import load_all_models
    for m in load_all_models():
        r = profile_model(m)
        print(r)
