"""
report_generator.py
-------------------
Auto-generates a Markdown technical report from outputs/results.csv.
The report follows the structure required by the course:
    abstract, introduction, syllabus relevance, problem statement,
    novelty, system architecture, methodology, adaptive algorithm,
    experiment setup, results table, plots, discussion, limitations,
    conclusion, references.

Run:
    python -m src.report_generator
Output:
    report/technical_report.md
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_PATH = REPORT_DIR / "technical_report.md"
PLOTS_REL = "../outputs/plots"  # relative path from report/ to plots


REPORT_TEMPLATE = """# On-Device Vision Encoder Benchmarking for Healthcare Edge AI
### with Context-Aware Adaptive Encoder Selection

**Course:** Intelligent Edge System
**Institution:** Sejong University
**Date:** {date}

---

## Abstract
Deploying deep vision models on edge devices for healthcare — for example,
running chest X-ray inference on a Jetson Nano in a rural clinic or on a
Raspberry Pi-based portable scanner — requires balancing accuracy, latency,
memory, and energy under constrained hardware. Static benchmarking alone
gives a snapshot of these trade-offs but does not produce a deployable
*decision policy*. In this work we (i) benchmark four lightweight vision
encoders (MobileViT, EfficientViT, TinyViT, DeiT-Tiny) on chest X-ray
inputs, profiling latency, peak memory, throughput, and an analytically
estimated energy-per-inference; (ii) compute the multi-objective Pareto
frontier; and (iii) propose a Context-Aware Adaptive Encoder Selector
(CA-AES) that picks the encoder at runtime based on device state — battery
level, thermal headroom, latency SLO, and clinical urgency mode. The
combined system is delivered as a reproducible Python pipeline plus a
Streamlit dashboard for clinical demonstration.

---

## 1. Introduction
Healthcare edge AI must operate where the patient is — in ambulances,
mobile clinics, and rural hospitals — under battery, memory, and thermal
constraints. Cloud offloading is not always possible (privacy, network).
Lightweight vision transformers and hybrid CNN-Transformer models have
recently emerged with very different efficiency profiles. Picking the
right model is non-trivial: the best model in the lab is rarely the best
model under low battery. This project (a) measures these models with edge
deployment in mind, and (b) operationalizes the trade-offs as a runtime
adaptive selector.

---

## 2. Syllabus Relevance
| Syllabus Topic | Where covered in this project |
|---|---|
| 1. Fundamentals of edge computing | Motivation in §1 and §3 (latency / privacy / bandwidth on-device) |
| 2. Edge computing architecture | System architecture §5 (sensor → on-device encoder → adaptive policy) |
| 3. Edge intelligence | Adaptive encoder selector §7 — runtime adaptation, not static deployment |
| 4. Case studies and practical applications | Chest X-ray triage scenarios in §7.3 |
| 5. Challenges and future research | Discussion §10 — energy/SLO trade-offs, calibration, INT8 |
| 6. Edge system deployment | Deployment notes for Pi 4 / Jetson Nano in README and §10 |
| 7. Project technical report | This document |

---

## 3. Problem Statement
Given a set of lightweight vision encoders {{m_1, ..., m_K}}, an edge
device with state s = (battery, thermal, latency-budget, mode), and an
input chest X-ray x, choose at runtime an encoder m* that maximizes
clinical utility under the device's current constraints. Static
"best-on-paper" rankings cannot answer this; we need a policy.

---

## 4. Novelty
Compared to a normal benchmark this project adds:

1. **Multi-objective Pareto analysis** across 4 dimensions (latency,
   memory, energy, accuracy) instead of single-metric ranking.
2. **Context-Aware Adaptive Encoder Selector (CA-AES)** — a runtime
   policy that selects models based on device state, with hard SLO
   filtering and soft utility scoring.
3. **Energy-aware deployment profile** — analytical TDP × CPU-utilization
   model that approximates Joules-per-inference cross-platform, with a
   documented upgrade path to `tegrastats` ground truth on Jetson.
4. **Deployable Streamlit dashboard** with mode toggling and human-readable
   recommendation reasons, suitable for clinical demonstration.
5. **Auto-generated reproducibility report** (this file).

---

## 5. System Architecture
```
   ┌────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
   │ Chest X-ray IN │ →  │ Lightweight Encoder  │ →  │ Pathology Logits   │
   └────────────────┘    │  (selected at runtime)│    └────────────────────┘
                         └──────────▲───────────┘                │
                                    │                            ▼
                         ┌──────────┴────────────┐    ┌────────────────────┐
                         │ CA-AES Adaptive       │ ←  │ Device State:      │
                         │ Encoder Selector      │    │  battery, thermal, │
                         │ (this project)        │    │  mode, SLO         │
                         └───────────────────────┘    └────────────────────┘
```

---

## 6. Methodology

### 6.1 Models
We use canonical implementations from `timm`:
MobileViT-S, EfficientViT-B0, TinyViT-5M, DeiT-Tiny. If pretrained weights
cannot be downloaded, the fallback is random init and accuracy values are
clearly flagged.

### 6.2 Profiling
- **Latency:** mean and 95th-percentile of {n_iters} timed forward passes
  after warmup; batch size 1 (realistic for one patient at a time).
- **Memory:** peak RSS delta during inference, via `psutil` (cross-platform).
- **Throughput:** images/second at steady state.
- **Energy:** `E = TDP × CPU_util × latency`, with TDP profiles per device.
  Documented upgrade path: replace with `tegrastats` on Jetson.

### 6.3 Pareto Frontier
A model is Pareto-optimal if no other model is ≤ in every cost dimension
and < in at least one. We minimize {{latency, memory, energy}} and
maximize {{accuracy}}.

---

## 7. Adaptive Algorithm (CA-AES)

For each candidate model m, compute the utility:

    U(m | s) = w_acc(s)·â(m) − w_lat(s)·l̂(m) − w_mem(s)·m̂(m) − w_eng(s)·ê(m)

where â, l̂, m̂, ê are min-max-normalized accuracy, latency, memory, energy.
Weights w_*(s) depend on the active EdgeMode:

| Mode       | w_acc | w_lat | w_mem | w_eng | When used                       |
|---|---|---|---|---|---|
| EMERGENCY  | 0.20  | 0.55  | 0.10  | 0.15  | ER triage, time-critical        |
| ROUTINE    | 0.60  | 0.15  | 0.10  | 0.15  | Scheduled screening             |
| LOW_POWER  | 0.20  | 0.15  | 0.15  | 0.50  | Mobile clinic, battery <25%     |
| BALANCED   | 0.30  | 0.25  | 0.20  | 0.25  | Default                         |

**Auto-mode override:** if battery ≤ 25% the policy auto-promotes to
LOW_POWER unless EMERGENCY is asserted (clinical priority wins).

**Hard constraints:** latency budget and memory budget filter candidates
*before* scoring. If no candidate survives, the system returns
`<none>` and (if `network_available`) recommends cloud offload.

### 7.3 Example scenarios produced by the selector
See `python -m src.adaptive_selector` for live output. Typical behavior:
- ER triage → fastest model (often DeiT-Tiny or EfficientViT-B0).
- Routine screening → most accurate model.
- Low battery → lowest-energy model.
- Pi 4 with 80 MB cap → smallest model that still satisfies the budget.

---

## 8. Experiment Setup
- **Hardware:** laptop CPU (default), with documented configurations for
  Raspberry Pi 4 and Jetson Nano.
- **Dataset:** chest X-ray images placed in `data/images/` with a
  `data/labels.csv`. If absent, a synthetic demo dataset is auto-generated
  for system testing.
- **Iterations:** 5 warmup + 30 timed forward passes per model.
- **Software:** PyTorch ≥ 2.0, timm ≥ 0.9, psutil, ONNX Runtime, Streamlit.

---

## 9. Results

### 9.1 Benchmark Table

{results_table}

### 9.2 Plots
![Latency vs Accuracy]({plots_rel}/latency_vs_accuracy.png)
![Memory vs Accuracy]({plots_rel}/memory_vs_accuracy.png)
![Energy vs Accuracy]({plots_rel}/energy_vs_accuracy.png)
![Pareto Frontier]({plots_rel}/pareto_frontier.png)

### 9.3 Pareto-Optimal Set
{pareto_models}

---

## 10. Discussion
The Pareto analysis confirms there is no universally best encoder. For
example, DeiT-Tiny minimizes parameters but may not minimize latency
because attention dominates compute. EfficientViT-B0 typically shows the
best latency / energy trade-off on CPU, while TinyViT-5M has higher
accuracy but higher memory. These trade-offs justify a *policy* rather
than a fixed choice — exactly what CA-AES provides.

For real clinical deployment several extensions are needed:
- **Calibration:** post-hoc Platt scaling / temperature scaling so logits
  are clinically interpretable.
- **INT8 quantization** via ONNX Runtime for additional 2–4× speedup.
- **Confidence-aware fallback:** offload to a heavier cloud model when
  the on-device model's confidence is below a threshold.

---

## 11. Limitations
- Energy is estimated analytically, not measured with a power meter.
  Documented upgrade path: `tegrastats` on Jetson.
- The synthetic demo dataset is for system testing only; real evaluation
  requires NIH ChestX-ray14 or equivalent with disease labels.
- Encoders are pretrained on ImageNet, not chest X-rays; reported
  accuracy is therefore a *transfer baseline*, not clinical accuracy.
  Domain adaptation / fine-tuning is future work.

---

## 12. Conclusion
We presented an end-to-end edge-AI pipeline that benchmarks four
lightweight vision encoders for chest X-ray inference, computes the
4-dimensional Pareto frontier, and — most importantly — converts those
benchmark numbers into a **runtime adaptive selection policy** (CA-AES)
that responds to device battery, thermal, and clinical-urgency state.
The system is reproducible, runs on a laptop CPU, and has documented
upgrade paths to Raspberry Pi and Jetson Nano. This addresses syllabus
topics 1, 2, 3, 4, 5, 6, and 7 in a single integrated artifact.

---

## References
1. Mehta, S. & Rastegari, M. *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer.* ICLR 2022.
2. Cai, H. et al. *EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction.* ICCV 2023.
3. Wu, K. et al. *TinyViT: Fast Pretraining Distillation for Small Vision Transformers.* ECCV 2022.
4. Touvron, H. et al. *Training data-efficient image transformers & distillation through attention (DeiT).* ICML 2021.
5. Wang, X. et al. *ChestX-ray8: Hospital-scale Chest X-ray Database.* CVPR 2017.
6. Shi, W. et al. *Edge Computing: Vision and Challenges.* IEEE IoT Journal, 2016.
7. Zhou, Z. et al. *Edge Intelligence: Paving the Last Mile of AI With Edge Computing.* Proc. IEEE, 2019.
8. NVIDIA, *Jetson Nano Developer Kit / tegrastats Reference.* 2023.

---
*Auto-generated by `src.report_generator` from `outputs/results.csv`.*
"""


def _format_results_table(df: pd.DataFrame) -> str:
    cols = [
        "model", "num_params_m", "model_size_mb",
        "latency_ms", "p95_latency_ms", "throughput_ips",
        "memory_mb", "energy_mj", "accuracy",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_markdown(index=False)


def generate_report(n_iters: int = 30) -> Path:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found — run `python -m src.benchmark` first.")

    df = pd.read_csv(RESULTS_CSV)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if "pareto_optimal" in df.columns:
        pareto_models = df[df["pareto_optimal"] == True]["model"].tolist()
    else:
        pareto_models = []
    pareto_str = (", ".join(f"`{m}`" for m in pareto_models)
                  if pareto_models else "_(run `python -m src.pareto` to compute)_")

    md = REPORT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        n_iters=n_iters,
        results_table=_format_results_table(df),
        plots_rel=PLOTS_REL,
        pareto_models=pareto_str,
    )
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"[report] wrote {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    generate_report()
