"""
asl.data -- Dataset loading and target transformations for the training stage.

Loads CSV files produced by asl.cov_data, applies input standardization
(StandardScaler) and target transforms (log1p on registered columns + joint
standardization).  Provides inverse transforms for recovering original units.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from asl.spec import Model


def log1p_mask(model: Model) -> np.ndarray:
    """Return a boolean mask for summaries registered with log1p transform."""
    return np.array(
        [transform == "log1p" for transform in model.summary_transforms],
        dtype=bool,
    )


def summary_column_masks(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for log1p and identity summary columns."""
    rt_mask = log1p_mask(model)
    return rt_mask, ~rt_mask


def count_rt_columns(model: Model) -> int:
    """Count summary columns registered for log1p before scaling."""
    return int(log1p_mask(model).sum())


class TargetTransform:
    """Transform targets for neural network training.

    Applies log1p to registered columns, then jointly standardizes all columns.
    The transform is invertible for export and R-squared evaluation.
    """

    def __init__(self, rt_mask: np.ndarray):
        """Initialize.

        Parameters
        ----------
        rt_mask : np.ndarray of bool, shape (n_targets,)
            True for columns that receive log1p.
        """
        self.rt_mask = np.asarray(rt_mask, dtype=bool)
        self.scaler = StandardScaler()

    @classmethod
    def from_model(cls, model: Model) -> "TargetTransform":
        """Build a transform from a model's registered summary transforms."""
        return cls(log1p_mask(model))

    @property
    def n_rt_columns(self) -> int:
        """Number of log1p columns (for backward-compatible export metadata)."""
        return int(self.rt_mask.sum())

    def _apply_log1p(self, y: np.ndarray) -> np.ndarray:
        y_t = y.copy()
        y_t[:, self.rt_mask] = np.log1p(y_t[:, self.rt_mask])
        return y_t

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit on training targets and return transformed values."""
        y_t = self._apply_log1p(y)
        y_t = self.scaler.fit_transform(y_t)
        return y_t.astype(np.float32)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform new targets using already-fit statistics."""
        y_t = self._apply_log1p(y)
        y_t = self.scaler.transform(y_t)
        return y_t.astype(np.float32)

    def inverse_transform(self, y_t: np.ndarray) -> np.ndarray:
        """Map transformed predictions back to original physical units."""
        y = self.scaler.inverse_transform(y_t)
        y[:, self.rt_mask] = np.maximum(np.expm1(y[:, self.rt_mask]), 0.0)
        return y


def save_target_transform(slug: str, transform: TargetTransform) -> None:
    """Persist a fitted TargetTransform to results/<slug>/target_transform.pkl."""
    path = Path("results") / slug / "target_transform.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(transform, f)


def load_target_transform(slug: str) -> TargetTransform:
    """Load a fitted TargetTransform from results/<slug>/target_transform.pkl."""
    path = Path("results") / slug / "target_transform.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def column_transforms_for_target(target_transform: TargetTransform) -> list[str]:
    """Map ASL rt_mask to JNNX v2 column_transforms entries."""
    return [
        "log1p" if bool(target_transform.rt_mask[i]) else "identity"
        for i in range(len(target_transform.rt_mask))
    ]


def build_obs_transform_payload(
    target_transform: TargetTransform,
    summary_names: tuple[str, ...] | list[str],
) -> dict:
    """Build obs_transform.json payload for a JNNX v2 SL package."""
    return {
        "version": "1.0",
        "summary_names": list(summary_names),
        "column_transforms": column_transforms_for_target(target_transform),
        "scaler_mean": target_transform.scaler.mean_.tolist(),
        "scaler_scale": target_transform.scaler.scale_.tolist(),
    }


def write_obs_transform_json(
    slug: str,
    summary_names: tuple[str, ...] | list[str],
    package_dir: Path,
) -> Path:
    """Serialize target_transform.pkl into package obs_transform.json."""
    target_transform = load_target_transform(slug)
    payload = build_obs_transform_payload(target_transform, summary_names)
    path = package_dir / "obs_transform.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
