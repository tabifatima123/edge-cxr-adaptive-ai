"""
models.py
---------
Loads lightweight vision encoders for chest X-ray edge benchmarking.

Encoders compared:
    - MobileViT       (Apple, 2022) — hybrid CNN+Transformer
    - EfficientViT    (MIT-Han Lab) — efficient attention
    - TinyViT         (Microsoft)   — distilled small ViT
    - DeiT-Tiny       (Facebook)    — data-efficient image transformer
    - (optional) ViT-Small as a heavier baseline

We use `timm` for canonical implementations. If `timm` weights cannot be
downloaded (no internet / firewall), the model is constructed with random
weights so latency/memory profiling still works. Accuracy numbers in that
case are not meaningful and the code prints a clear warning.

Each model is wrapped to expose:
    - .forward(x)       → embedding or logits
    - .num_params()
    - .name
"""

from typing import Dict, Callable, List
import warnings

import torch
import torch.nn as nn

# --- Model registry: (display_name, timm_model_name, image_size) ---
MODEL_REGISTRY: Dict[str, Dict] = {
    "MobileViT-S":     {"timm_name": "mobilevit_s",         "img_size": 256},
    "EfficientViT-B0": {"timm_name": "efficientvit_b0.r224_in1k", "img_size": 224},
    "TinyViT-5M":      {"timm_name": "tiny_vit_5m_224.dist_in22k_ft_in1k", "img_size": 224},
    "DeiT-Tiny":       {"timm_name": "deit_tiny_patch16_224", "img_size": 224},
    # Optional heavier baseline (commented out — uncomment to include)
    # "ViT-Small":      {"timm_name": "vit_small_patch16_224", "img_size": 224},
}

NUM_CLASSES = 8  # matches DEMO_CLASSES in dataset.py


class EncoderWrapper(nn.Module):
    """
    Wraps a timm model so its forward() returns class logits.
    Stores metadata used by the profiler/report.
    """

    def __init__(self, name: str, model: nn.Module, img_size: int, weights_loaded: bool):
        super().__init__()
        self.name = name
        self.model = model
        self.img_size = img_size
        self.weights_loaded = weights_loaded
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _build_with_timm(timm_name: str, num_classes: int) -> (nn.Module, bool):
    """Try to build a timm model with pretrained weights, fall back to random init."""
    try:
        import timm
    except ImportError:
        raise RuntimeError("`timm` not installed. Run: pip install timm")

    # Try with pretrained weights first
    try:
        model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        return model, True
    except Exception as e:
        warnings.warn(
            f"[models] Could not load pretrained weights for '{timm_name}' ({e}). "
            "Falling back to random init — latency/memory still valid, accuracy is NOT."
        )
        try:
            model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
            return model, False
        except Exception as e2:
            raise RuntimeError(f"Failed to build {timm_name}: {e2}")


def load_model(display_name: str) -> EncoderWrapper:
    """Public API: load one model by friendly name."""
    if display_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{display_name}'. Choices: {list(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[display_name]
    model, weights_loaded = _build_with_timm(cfg["timm_name"], NUM_CLASSES)
    return EncoderWrapper(display_name, model, cfg["img_size"], weights_loaded)


def load_all_models() -> List[EncoderWrapper]:
    """Load every model in the registry. Skips ones that fail entirely."""
    out = []
    for name in MODEL_REGISTRY:
        try:
            m = load_model(name)
            print(f"[models] Loaded {name} ({m.num_params()/1e6:.2f}M params, "
                  f"weights={'pretrained' if m.weights_loaded else 'random'})")
            out.append(m)
        except Exception as e:
            print(f"[models] Skipping {name}: {e}")
    return out


if __name__ == "__main__":
    models = load_all_models()
    for m in models:
        x = torch.randn(1, 3, m.img_size, m.img_size)
        y = m(x)
        print(f"{m.name}: input {tuple(x.shape)} → output {tuple(y.shape)}")
