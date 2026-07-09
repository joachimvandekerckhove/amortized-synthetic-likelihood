"""Neural network architectures for multivariate emulator training."""

import os
import sys

import torch
import torch.nn as nn

from asl.config import load_config


class DeepWideMLP(nn.Module):
    """Multi-layer perceptron with BatchNorm and GELU activations."""

    def __init__(self, width: int, depth: int, in_dim: int, out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d, width), nn.BatchNorm1d(width), nn.GELU()])
            d = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters in this network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _select_device() -> torch.device:
    """Select the best available compute device."""
    if load_config().get("device", "unset_cuda_visible_devices", False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if not torch.cuda.is_available():
        print("[mlp] CUDA not available; training on CPU.")
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[mlp] Using CUDA device: {name}")
        return torch.device("cuda")
    except Exception as exc:
        print(
            "[mlp] CUDA is visible but unusable; training on CPU. "
            f"({exc}). For GTX 1080 Ti / sm_61 GPUs install torch 2.6+cu126: "
            "pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126",
            file=sys.stderr,
        )
        return torch.device("cpu")


DEVICE = _select_device()


def resolve_training_settings() -> dict:
    """Resolve training hyperparameters from TOML configuration."""
    config = load_config()
    return {
        "subsample": config.get("training", "max_training_rows"),
        "n_epochs": int(config.get("training", "training_epochs")),
        "batch_size": int(config.get("training", "batch_size")),
        "lr": float(config.get("training", "learning_rate")),
    }


def build_architecture(name: str, in_dim: int, out_dim: int):
    """Return a builder for a supported architecture name."""
    catalogue = {
        "DeepWide_24x4": lambda: DeepWideMLP(24, 4, in_dim, out_dim),
        "DeepWide_32x6": lambda: DeepWideMLP(32, 6, in_dim, out_dim),
    }
    if name not in catalogue:
        raise ValueError(
            f"Unknown architecture '{name}'. Supported: {sorted(catalogue)}"
        )
    return catalogue[name]
