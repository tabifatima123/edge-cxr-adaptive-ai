"""
adaptive_selector.py
--------------------
**Core novel contribution of this project.**

Context-Aware Adaptive Encoder Selector (CA-AES).

A normal benchmarking project just ranks models. This module turns the
benchmark into a *runtime decision policy*: given the current device state
(battery %, thermal headroom, clinical urgency, network availability), it
selects the best lightweight encoder by:

    1. Reading benchmark results from outputs/results.csv.
    2. Filtering models that violate hard constraints
       (e.g. latency budget, memory cap).
    3. Scoring remaining candidates with a context-weighted utility:

           U(m) = w_acc · norm(accuracy)
                - w_lat · norm(latency)
                - w_mem · norm(memory)
                - w_eng · norm(energy)

       The weights w_* depend on the current `EdgeMode`:

           EMERGENCY    : maximize speed (w_lat dominant)
           ROUTINE      : maximize accuracy (w_acc dominant)
           LOW_POWER    : minimize energy   (w_eng dominant)
           BALANCED     : even split

This implements syllabus topics:
    - Edge intelligence (3): runtime adaptation, not static deployment
    - Challenges and future research (5): power/SLO trade-offs
    - Edge system deployment (6): policy you actually deploy

Usage:
    from src.adaptive_selector import AdaptiveSelector, EdgeMode, DeviceContext
    sel = AdaptiveSelector.from_csv("outputs/results.csv")
    decision = sel.select(DeviceContext(battery_pct=20, mode=EdgeMode.LOW_POWER))
    print(decision.chosen_model, decision.reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "outputs" / "results.csv"


# ---------------------------------------------------------------
# Edge modes — what the device prioritizes right now
# ---------------------------------------------------------------
class EdgeMode(str, Enum):
    EMERGENCY = "emergency"      # ER triage — speed at all costs
    ROUTINE = "routine"          # scheduled screening — favor accuracy
    LOW_POWER = "low_power"      # mobile clinic, battery low — favor energy
    BALANCED = "balanced"        # default


# Per-mode utility weights. Must each sum so accuracy is always positive
# and the cost terms are negative; the math handles signs internally.
MODE_WEIGHTS: Dict[EdgeMode, Dict[str, float]] = {
    EdgeMode.EMERGENCY:  {"acc": 0.20, "lat": 0.55, "mem": 0.10, "eng": 0.15},
    EdgeMode.ROUTINE:    {"acc": 0.60, "lat": 0.15, "mem": 0.10, "eng": 0.15},
    EdgeMode.LOW_POWER:  {"acc": 0.20, "lat": 0.15, "mem": 0.15, "eng": 0.50},
    EdgeMode.BALANCED:   {"acc": 0.30, "lat": 0.25, "mem": 0.20, "eng": 0.25},
}


@dataclass
class DeviceContext:
    """Snapshot of device state used to drive selection."""
    battery_pct: float = 100.0       # 0-100
    thermal_headroom_c: float = 30.0  # degrees C below throttle (higher = cooler)
    mode: EdgeMode = EdgeMode.BALANCED
    latency_budget_ms: Optional[float] = None   # hard SLO (skip slower models)
    memory_budget_mb: Optional[float] = None    # hard cap
    network_available: bool = False              # True → could offload to cloud


@dataclass
class SelectionDecision:
    chosen_model: str
    score: float
    reason: str
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------
# Selector
# ---------------------------------------------------------------
class AdaptiveSelector:
    REQUIRED_COLS = ["model", "latency_ms", "memory_mb", "energy_mj", "accuracy"]

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"results dataframe missing columns: {missing}")
        self.df = df.copy().reset_index(drop=True)
        self.df["accuracy"] = self.df["accuracy"].fillna(0.0)
        self._normalized = self._normalize_metrics(self.df)

    @classmethod
    def from_csv(cls, path: Path = RESULTS_CSV) -> "AdaptiveSelector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python -m src.benchmark` first.")
        return cls(pd.read_csv(path))

    @staticmethod
    def _normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize each metric to [0, 1]."""
        out = pd.DataFrame({"model": df["model"]})
        for col in ["latency_ms", "memory_mb", "energy_mj", "accuracy"]:
            v = df[col].astype(float).to_numpy()
            lo, hi = float(np.min(v)), float(np.max(v))
            if hi - lo < 1e-9:
                out[col] = 0.0
            else:
                out[col] = (v - lo) / (hi - lo)
        return out

    # -------- battery-aware mode override --------
    @staticmethod
    def auto_mode(ctx: DeviceContext) -> EdgeMode:
        """If battery is low, override into LOW_POWER unless EMERGENCY."""
        if ctx.mode == EdgeMode.EMERGENCY:
            return EdgeMode.EMERGENCY
        if ctx.battery_pct <= 25.0:
            return EdgeMode.LOW_POWER
        return ctx.mode

    # -------- main API --------
    def select(self, ctx: DeviceContext) -> SelectionDecision:
        active_mode = self.auto_mode(ctx)
        weights = MODE_WEIGHTS[active_mode]

        rejected: Dict[str, str] = {}
        candidate_scores: Dict[str, float] = {}

        for i, row in self.df.iterrows():
            name = row["model"]

            # Hard constraints
            if ctx.latency_budget_ms is not None and row["latency_ms"] > ctx.latency_budget_ms:
                rejected[name] = (f"latency {row['latency_ms']}ms > "
                                  f"budget {ctx.latency_budget_ms}ms")
                continue
            if ctx.memory_budget_mb is not None and row["memory_mb"] > ctx.memory_budget_mb:
                rejected[name] = (f"memory {row['memory_mb']}MB > "
                                  f"budget {ctx.memory_budget_mb}MB")
                continue

            # Soft scoring on normalized metrics
            n = self._normalized.iloc[i]
            score = (
                + weights["acc"] * n["accuracy"]
                - weights["lat"] * n["latency_ms"]
                - weights["mem"] * n["memory_mb"]
                - weights["eng"] * n["energy_mj"]
            )
            candidate_scores[name] = round(float(score), 4)

        if not candidate_scores:
            return SelectionDecision(
                chosen_model="<none>",
                score=float("-inf"),
                reason="No model satisfies the hard constraints. "
                       "Consider relaxing budgets or offloading to cloud "
                       "(network_available=" + str(ctx.network_available) + ").",
                rejected=rejected,
            )

        chosen = max(candidate_scores, key=candidate_scores.get)
        chosen_row = self.df[self.df["model"] == chosen].iloc[0]

        reason = self._explain(chosen_row, active_mode, ctx, candidate_scores)
        return SelectionDecision(
            chosen_model=chosen,
            score=candidate_scores[chosen],
            reason=reason,
            candidate_scores=candidate_scores,
            rejected=rejected,
        )

    @staticmethod
    def _explain(row, mode: EdgeMode, ctx: DeviceContext, scores: Dict[str, float]) -> str:
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])
        runner_up = sorted_scores[1][0] if len(sorted_scores) > 1 else "—"
        lines = [
            f"Mode active: {mode.value.upper()} "
            f"(battery={ctx.battery_pct:.0f}%, "
            f"thermal_headroom={ctx.thermal_headroom_c:.1f}°C).",
            f"Chosen: {row['model']} — latency={row['latency_ms']}ms, "
            f"memory={row['memory_mb']}MB, energy={row['energy_mj']}mJ, "
            f"acc={row['accuracy']}.",
            f"Best score under current weights "
            f"(acc={MODE_WEIGHTS[mode]['acc']}, lat={MODE_WEIGHTS[mode]['lat']}, "
            f"mem={MODE_WEIGHTS[mode]['mem']}, eng={MODE_WEIGHTS[mode]['eng']}). "
            f"Runner-up: {runner_up}.",
        ]
        return " ".join(lines)


# ---------------------------------------------------------------
# CLI demo: shows the selector adapting across 4 scenarios
# ---------------------------------------------------------------
def _demo():
    sel = AdaptiveSelector.from_csv()
    scenarios = [
        ("ER triage, plugged in",
         DeviceContext(battery_pct=100, mode=EdgeMode.EMERGENCY,
                       latency_budget_ms=200)),
        ("Routine screening, plugged in",
         DeviceContext(battery_pct=100, mode=EdgeMode.ROUTINE)),
        ("Mobile clinic, low battery",
         DeviceContext(battery_pct=18, mode=EdgeMode.BALANCED)),
        ("Tight RAM budget on Pi 4",
         DeviceContext(battery_pct=80, mode=EdgeMode.BALANCED,
                       memory_budget_mb=80)),
    ]
    print("\n" + "=" * 60)
    print("ADAPTIVE SELECTOR — SCENARIO DEMO")
    print("=" * 60)
    for label, ctx in scenarios:
        d = sel.select(ctx)
        print(f"\n[{label}]")
        print(f"  → {d.chosen_model}  (score={d.score})")
        print(f"  reason: {d.reason}")
        if d.rejected:
            print(f"  rejected: {d.rejected}")


if __name__ == "__main__":
    _demo()
