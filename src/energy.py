"""
energy.py
---------
Analytical energy-per-inference estimator for edge devices.

Why analytical?
    Most laptops / Raspberry Pi do not expose a per-process power meter.
    We therefore estimate energy as:

        E_per_inference (J)  =  P_avg (W)  ×  latency (s)

    where P_avg is approximated as:

        P_avg = TDP × utilization_fraction

    `utilization_fraction` is taken from psutil.cpu_percent during the
    benchmark window. TDP defaults are configured per device profile.

    On Jetson Nano you should replace this with `tegrastats` ground truth
    (mentioned in README).
"""

import time
from dataclasses import dataclass
from typing import Dict

import psutil
import torch

from .models import EncoderWrapper


# Per-device TDP (Thermal Design Power) lookup, in watts.
# Conservative published numbers for typical edge / laptop CPUs.
DEVICE_TDP_W: Dict[str, float] = {
    "laptop_cpu":     15.0,   # typical mid-range laptop CPU package
    "raspberry_pi_4":  6.4,   # Pi 4 under full load
    "raspberry_pi_5":  8.0,
    "jetson_nano":    10.0,   # MAXN mode
    "jetson_orin":    15.0,
    "default":        15.0,
}


@dataclass
class EnergyResult:
    model: str
    device_profile: str
    avg_power_w: float
    energy_per_inference_mj: float    # millijoules
    latency_ms: float
    cpu_util_pct: float


def estimate_energy(
    model: EncoderWrapper,
    device_profile: str = "laptop_cpu",
    num_iters: int = 30,
    batch_size: int = 1,
) -> EnergyResult:
    """
    Run the model for `num_iters` and estimate energy/inference using
    TDP × CPU utilization × latency.
    """
    tdp = DEVICE_TDP_W.get(device_profile, DEVICE_TDP_W["default"])

    model = model.eval()
    x = torch.randn(batch_size, 3, model.img_size, model.img_size)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    # Prime cpu_percent (first call returns 0.0 with interval=None)
    psutil.cpu_percent(interval=None)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)
    t1 = time.perf_counter()

    cpu_util = psutil.cpu_percent(interval=None)  # % since last call
    cpu_util = max(cpu_util, 5.0)  # clamp — never assume 0% during inference

    total_time_s = t1 - t0
    latency_s = total_time_s / num_iters
    avg_power = tdp * (cpu_util / 100.0)
    energy_per_inference_j = avg_power * latency_s

    return EnergyResult(
        model=model.name,
        device_profile=device_profile,
        avg_power_w=round(avg_power, 3),
        energy_per_inference_mj=round(energy_per_inference_j * 1000.0, 3),
        latency_ms=round(latency_s * 1000.0, 2),
        cpu_util_pct=round(cpu_util, 1),
    )


if __name__ == "__main__":
    from .models import load_all_models
    for m in load_all_models():
        r = estimate_energy(m)
        print(r)
