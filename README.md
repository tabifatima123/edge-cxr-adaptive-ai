# On-Device Vision Encoder Benchmarking for Healthcare Edge AI
### with Context-Aware Adaptive Encoder Selection

**Course:** Intelligent Edge System — Sejong University
**Project Type:** Midterm Project
**Author:** [Your Name]

---

## 🎯 Project Summary

This project benchmarks lightweight vision encoders (MobileViT, EfficientViT, TinyViT, DeiT-Tiny) on chest X-ray classification for healthcare edge devices, and introduces a **Context-Aware Adaptive Encoder Selector (CA-AES)** that dynamically picks the optimal model at runtime based on device state (battery, thermal load, clinical urgency).

### Novelty over a normal benchmark
A typical benchmark project just measures latency/accuracy and ranks models. This project goes further:

1. **Pareto-frontier analysis** across 4 dimensions (latency, memory, energy, accuracy) instead of single-metric ranking.
2. **Adaptive runtime selector** — the deployed system *changes which model it uses* based on simulated device context (low battery → smaller model; emergency mode → fastest model; routine screening → most accurate model).
3. **Energy-aware deployment profile** — combines CPU profiling + analytical power model to estimate Joules per inference, which is the actual currency on Jetson Nano / Raspberry Pi.
4. **Auto-generated technical report** for reproducibility.

---

## 📁 Folder Structure

```
edge_cxr_bench/
├── data/
│   ├── images/               # chest X-ray images (.png/.jpg)
│   └── labels.csv            # filename,label columns
├── src/
│   ├── __init__.py
│   ├── dataset.py            # dataset loader + demo fallback
│   ├── models.py             # lightweight encoder loader
│   ├── profiler.py           # latency + memory + throughput profiling
│   ├── energy.py             # analytical energy estimator
│   ├── benchmark.py          # main benchmarking pipeline
│   ├── pareto.py             # Pareto frontier computation + plots
│   ├── adaptive_selector.py  # context-aware adaptive encoder selector
│   └── report_generator.py   # auto-generates Markdown technical report
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── outputs/
│   ├── results.csv
│   └── plots/
│       ├── latency_vs_accuracy.png
│       ├── memory_vs_accuracy.png
│       ├── pareto_frontier.png
│       └── energy_vs_accuracy.png
├── report/
│   └── technical_report.md
├── requirements.txt
└── README.md
```

---

## 🛠 Installation (Windows)

```powershell
# 1. Clone or unzip the project
cd edge_cxr_bench

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Step-by-Step Run Commands (Windows PowerShell)

```powershell
# Step 1: Activate venv
venv\Scripts\activate

# Step 2: (Optional) Place real chest X-ray images in data/images/
#         and a labels.csv in data/.
#         If you skip this, the project auto-generates demo synthetic images.

# Step 3: Run the full benchmarking pipeline
python -m src.benchmark

# Step 4: Run the adaptive selector demo
python -m src.adaptive_selector

# Step 5: Generate plots and Pareto frontier
python -m src.pareto

# Step 6: Auto-generate the technical report
python -m src.report_generator

# Step 7: Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

---

## 🔬 What Each Script Does

| Script | Purpose |
|---|---|
| `src/dataset.py` | Loads chest X-rays from `data/images/`. If empty, generates synthetic demo images. |
| `src/models.py` | Loads the 4 lightweight vision encoders via `timm`, falls back to torchvision if unavailable. |
| `src/profiler.py` | Measures latency (ms), throughput (img/s), memory (MB) using `psutil`. |
| `src/energy.py` | Estimates energy per inference (mJ) using analytical TDP model. |
| `src/benchmark.py` | Runs all models on the dataset, saves `outputs/results.csv`. |
| `src/pareto.py` | Computes Pareto-optimal models, generates 4 plots. |
| `src/adaptive_selector.py` | **Novel contribution** — picks best model at runtime per device context. |
| `src/report_generator.py` | Auto-generates `report/technical_report.md`. |
| `dashboard/app.py` | Streamlit UI with image upload, mode selector, recommendation reason. |

---

## 🖥 Running on Real Edge Hardware

The default code runs on a laptop CPU. To deploy on real edge hardware:

### Raspberry Pi 4 / 5
```bash
# Install ARM-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m src.benchmark
```

### NVIDIA Jetson Nano / Orin Nano
```bash
# Use the JetPack-provided PyTorch wheel (CUDA-enabled)
# Then in src/profiler.py, set device = 'cuda'
# Energy can be measured on-device via tegrastats:
sudo tegrastats --interval 100 --logfile power.log
```

The Pareto/adaptive selector logic is hardware-agnostic and works on any edge device.

---

## 📊 Outputs

After running the full pipeline, you will get:
- `outputs/results.csv` — all benchmark numbers
- `outputs/plots/*.png` — 4 plots
- `report/technical_report.md` — full technical report
- Streamlit dashboard live at `http://localhost:8501`

---

## ⚠️ Notes

- If `timm` cannot download model weights (no internet), the code falls back to **randomly initialized** weights and prints a warning. Latency/memory measurements remain valid; accuracy numbers will not.
- The energy estimate uses an analytical TDP model. For ground-truth values on Jetson, use `tegrastats`.
- The synthetic dataset is for **system testing only** — not clinical use.

---

## 📚 References
See `report/technical_report.md` for full bibliography.
