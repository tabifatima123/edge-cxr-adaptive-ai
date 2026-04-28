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

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_selector import (   # noqa: E402
    AdaptiveSelector, DeviceContext, EdgeMode,
)

RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"

# ------------------------------------------------------------
st.set_page_config(
    page_title="Edge CXR — Adaptive Encoder Selector",
    layout="wide",
    page_icon="🩻",
)

st.title("🩻 Edge CXR — Adaptive Vision Encoder Selector")
st.caption("Context-aware runtime model selection for chest X-ray edge AI · "
           "Sejong University, *Intelligent Edge System* midterm project")


# ------------------------------------------------------------
# Sidebar — device context controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Device Context")

mode_label = st.sidebar.selectbox(
    "Edge Mode",
    options=["BALANCED", "EMERGENCY", "ROUTINE", "LOW_POWER"],
    index=0,
    help="Drives the utility weights used by the selector.",
)
mode = EdgeMode[mode_label]

battery = st.sidebar.slider("Battery (%)", 0, 100, 80)
thermal = st.sidebar.slider("Thermal headroom (°C below throttle)", 0, 50, 30)
lat_budget = st.sidebar.number_input(
    "Latency budget (ms, 0 = no limit)", min_value=0, value=0, step=10)
mem_budget = st.sidebar.number_input(
    "Memory budget (MB, 0 = no limit)", min_value=0, value=0, step=10)
network = st.sidebar.checkbox("Network available (cloud offload allowed)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Run `python -m src.benchmark` first to populate results.csv.")


# ------------------------------------------------------------
# Main content
# ------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Chest X-ray")
    upload = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
    if upload is not None:
        img = Image.open(upload).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
    else:
        # Try to show a demo image if dataset has been generated
        demo = PROJECT_ROOT / "data" / "images" / "demo_000.png"
        if demo.exists():
            st.image(str(demo), caption="Demo X-ray (synthetic)",
                     use_column_width=True)
        else:
            st.info("Upload a chest X-ray, or run `python -m src.benchmark` "
                    "to generate demo images.")

with col2:
    st.subheader("🤖 Adaptive Selection")

    if not RESULTS_CSV.exists():
        st.error(f"`{RESULTS_CSV}` not found.\n\n"
                 "Run **`python -m src.benchmark`** first.")
        st.stop()

    df = pd.read_csv(RESULTS_CSV)
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
    m2.metric("Memory",  f"{chosen_row['memory_mb']:.1f} MB")
    m3.metric("Energy",  f"{chosen_row['energy_mj']:.1f} mJ")
    m4.metric("Accuracy",
              f"{chosen_row['accuracy']:.3f}" if pd.notna(chosen_row["accuracy"]) else "—")

    st.markdown("**Predicted output (placeholder):** "
                "Pneumonia probability *0.27* | No Finding *0.62* | …")
    st.caption("This is a placeholder — wire your fine-tuned classification "
               "head here for real predictions.")

    st.markdown("### 🧠 Why this model?")
    st.write(decision.reason)

    with st.expander("All candidate scores (current context)"):
        st.json(decision.candidate_scores)
    if decision.rejected:
        with st.expander("Rejected by hard constraints"):
            st.json(decision.rejected)


# ------------------------------------------------------------
# Bottom panel — full benchmark + Pareto
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Full Benchmark")

cols_to_show = [
    "model", "num_params_m", "latency_ms", "p95_latency_ms",
    "memory_mb", "energy_mj", "accuracy",
]
cols_to_show = [c for c in cols_to_show if c in df.columns]
st.dataframe(df[cols_to_show], use_container_width=True)

plots_dir = PROJECT_ROOT / "outputs" / "plots"
if plots_dir.exists():
    plots = sorted(plots_dir.glob("*.png"))
    if plots:
        st.subheader("📈 Plots")
        cols = st.columns(2)
        for i, p in enumerate(plots):
            with cols[i % 2]:
                st.image(str(p), caption=p.stem, use_column_width=True)

st.caption("© Edge CXR Benchmark · Intelligent Edge System (Sejong University)")
