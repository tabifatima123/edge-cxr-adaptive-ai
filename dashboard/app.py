"""
dashboard/app.py
----------------
Streamlit UI for the Context-Aware Adaptive Encoder Selector.

Features (per spec):
    - Upload a chest X-ray
    - Select an edge mode (EMERGENCY / ROUTINE / LOW_POWER / BALANCED)
    - Show the chosen model, latency, memory, energy
    - Show predicted output placeholder
    - Show recommendation reason

Run:
    streamlit run dashboard/app.py
"""
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
import streamlit as st

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


# ------------------------------------------------------------
# Cloud-safe fallback results
# ------------------------------------------------------------
def ensure_results_csv():
    """
    Streamlit Cloud does not keep local generated files.
    If outputs/results.csv is missing, create a lightweight fallback table
    using the benchmark results measured locally.
    """
    if RESULTS_CSV.exists():
        return

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {
            "model": "MobileViT-S",
            "num_params_m": 4.94,
            "model_size_mb": 18.86,
            "latency_ms": 97.91,
            "p95_latency_ms": 115.25,
            "throughput_ips": 10.21,
            "memory_mb": 0.20,
            "energy_mj": 770.22,
            "accuracy": 0.124,
            "num_evaluated": 1000,
            "weights_pretrained": True,
            "device_profile": "laptop_cpu",
        },
        {
            "model": "EfficientViT-B0",
            "num_params_m": 2.14,
            "model_size_mb": 8.16,
            "latency_ms": 31.37,
            "p95_latency_ms": 44.56,
            "throughput_ips": 31.88,
            "memory_mb": 0.02,
            "energy_mj": 493.724,
            "accuracy": 0.110,
            "num_evaluated": 1000,
            "weights_pretrained": True,
            "device_profile": "laptop_cpu",
        },
        {
            "model": "TinyViT-5M",
            "num_params_m": 5.07,
            "model_size_mb": 19.36,
            "latency_ms": 80.77,
            "p95_latency_ms": 88.97,
            "throughput_ips": 12.38,
            "memory_mb": 0.01,
            "energy_mj": 682.317,
            "accuracy": 0.128,
            "num_evaluated": 1000,
            "weights_pretrained": True,
            "device_profile": "laptop_cpu",
        },
        {
            "model": "DeiT-Tiny",
            "num_params_m": 5.53,
            "model_size_mb": 21.08,
            "latency_ms": 39.99,
            "p95_latency_ms": 51.46,
            "throughput_ips": 25.01,
            "memory_mb": 0.00,
            "energy_mj": 284.841,
            "accuracy": 0.119,
            "num_evaluated": 1000,
            "weights_pretrained": True,
            "device_profile": "laptop_cpu",
        },
    ])

    df.to_csv(RESULTS_CSV, index=False)


ensure_results_csv()

from src.adaptive_selector import AdaptiveSelector, DeviceContext, EdgeMode  # noqa: E402


# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Edge CXR — Adaptive Encoder Selector",
    layout="wide",
    page_icon="🩻",
)

st.title("🩻 Edge CXR — Adaptive Vision Encoder Selector")
st.caption(
    "Context-aware runtime model selection for chest X-ray edge AI · "
    "Sejong University, *Intelligent Edge System* midterm project"
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("⚙️ Device Context")

mode_label = st.sidebar.selectbox(
    "Edge Mode",
    options=["BALANCED", "EMERGENCY", "ROUTINE", "LOW_POWER"],
    index=0,
    help="Controls the selector weights for accuracy, latency, memory, and energy.",
)
mode = EdgeMode[mode_label]

battery = st.sidebar.slider("Battery (%)", 0, 100, 80)
thermal = st.sidebar.slider("Thermal headroom (°C below throttle)", 0, 50, 30)

lat_budget = st.sidebar.number_input(
    "Latency budget (ms, 0 = no limit)",
    min_value=0,
    value=0,
    step=10,
)

mem_budget = st.sidebar.number_input(
    "Memory budget (MB, 0 = no limit)",
    min_value=0,
    value=0,
    step=10,
)

network = st.sidebar.checkbox(
    "Network available (cloud offload allowed)",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Cloud-safe mode: if benchmark results are missing, the app loads saved fallback results."
)


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Chest X-ray")

    upload = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])

    if upload is not None:
        img = Image.open(upload).convert("RGB")
        st.image(img, caption="Uploaded image", width="stretch")
    else:
        st.info("Upload a chest X-ray image to simulate edge inference.")

with col2:
    st.subheader("🤖 Adaptive Selection")

    selector = AdaptiveSelector(df)

    ctx = DeviceContext(
        battery_pct=float(battery),
        thermal_headroom_c=float(thermal),
        mode=mode,
        latency_budget_ms=float(lat_budget) if lat_budget > 0 else None,
        memory_budget_mb=float(mem_budget) if mem_budget > 0 else None,
        network_available=network,
    )

    decision = selector.select(ctx)

    if decision.chosen_model == "<none>":
        st.error("❌ No model satisfies the hard constraints.")
        st.write(decision.reason)
        if decision.rejected:
            st.write("**Rejected models:**")
            st.json(decision.rejected)
        st.stop()

    chosen_row = df[df["model"] == decision.chosen_model].iloc[0]

    st.success(f"✅ Selected model: **{decision.chosen_model}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{chosen_row['latency_ms']:.1f} ms")
    m2.metric("Memory", f"{chosen_row['memory_mb']:.2f} MB")
    m3.metric("Energy", f"{chosen_row['energy_mj']:.1f} mJ")
    m4.metric("Accuracy", f"{chosen_row['accuracy']:.3f}")

    st.markdown(
        "**Predicted output (placeholder):** "
        "Pneumonia probability *0.27* | No Finding *0.62* | …"
    )

    st.caption(
        "This is a deployment prototype. The current models are ImageNet-pretrained, "
        "so prediction is a placeholder; the main focus is edge efficiency and adaptive selection."
    )

    st.markdown("### 🧠 Why this model?")
    st.write(decision.reason)

    with st.expander("All candidate scores"):
        st.json(decision.candidate_scores)

    if decision.rejected:
        with st.expander("Rejected by hard constraints"):
            st.json(decision.rejected)


# ------------------------------------------------------------
# Benchmark table
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Full Benchmark Results")

cols_to_show = [
    "model",
    "num_params_m",
    "model_size_mb",
    "latency_ms",
    "p95_latency_ms",
    "throughput_ips",
    "memory_mb",
    "energy_mj",
    "accuracy",
    "num_evaluated",
]
cols_to_show = [c for c in cols_to_show if c in df.columns]

st.dataframe(df[cols_to_show], width="stretch")


# ------------------------------------------------------------
# Plot display
# ------------------------------------------------------------
if PLOTS_DIR.exists():
    plots = sorted(PLOTS_DIR.glob("*.png"))
    if plots:
        st.subheader("📈 Benchmark Plots")
        cols = st.columns(2)
        for i, plot_path in enumerate(plots):
            with cols[i % 2]:
                st.image(str(plot_path), caption=plot_path.stem, width="stretch")

st.caption("© Edge CXR Benchmark · Intelligent Edge System · Sejong University")