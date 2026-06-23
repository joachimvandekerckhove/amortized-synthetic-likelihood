"""Neural network architectures for multivariate emulator training."""

import os

import torch
import torch.nn as nn


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


def _select_device() -> torch.device:
    """Select the best available compute device."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


DEVICE = _select_device()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_training_settings() -> dict:
    """Resolve training hyperparameters from environment."""
    is_smoke = os.environ.get("ESL_SMOKE", "0") == "1"
    n_epochs = 300 if is_smoke else 10000
    if "ESL_N_EPOCHS" in os.environ:
        n_epochs = int(os.environ["ESL_N_EPOCHS"])
    return {
        "subsample": 5000 if is_smoke else None,
        "n_epochs": n_epochs,
        "batch_size": 512 if is_smoke else 4096,
        "lr": 1e-3,
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
