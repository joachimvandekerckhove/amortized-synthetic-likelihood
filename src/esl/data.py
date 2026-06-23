"""
esl.data -- Dataset loading and target transformations for the training stage.

Loads CSV files produced by esl.generate_data, applies input standardization
(StandardScaler) and target transforms (log1p on RT columns + joint
standardization).  Provides inverse transforms for recovering original units.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from esl.registry import get_model
from esl.spec import Model


def _is_proportion_summary(name: str) -> bool:
    """Return True if a summary is a bounded rate/proportion (not RT-based)."""
    low = name.lower()
    if "rt" in low:
        return False
    return any(k in low for k in ("acc", "rate", "prob"))


def summary_column_masks(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for RT and proportion summary columns.

    Parameters
    ----------
    model : Model

    Returns
    -------
    rt_mask : np.ndarray of bool, shape (n_summaries,)
    prop_mask : np.ndarray of bool, shape (n_summaries,)
    """
    rt_mask = np.array(
        [not _is_proportion_summary(name) for name in model.summary_names],
        dtype=bool,
    )
    prop_mask = ~rt_mask
    return rt_mask, prop_mask


def count_rt_columns(model: Model) -> int:
    """Count RT-based summary columns (receive log1p before scaling)."""
    rt_mask, _ = summary_column_masks(model)
    return int(rt_mask.sum())


class TargetTransform:
    """Transform targets for neural network training.

    Applies log1p to RT columns, then jointly standardizes all columns.
    The transform is invertible for export and R-squared evaluation.
    """

    def __init__(self, rt_mask: np.ndarray):
        """Initialize.

        Parameters
        ----------
        rt_mask : np.ndarray of bool, shape (n_targets,)
            True for columns that receive log1p (RT summaries).
        """
        self.rt_mask = np.asarray(rt_mask, dtype=bool)
        self.scaler = StandardScaler()

    @classmethod
    def from_model(cls, model: Model) -> "TargetTransform":
        """Build a transform from a model's summary column layout."""
        rt_mask, _ = summary_column_masks(model)
        return cls(rt_mask)

    @property
    def n_rt_columns(self) -> int:
        """Number of RT columns (for backward-compatible export metadata)."""
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


def load_dataset(
    slug: str, subsample: int | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, Model]:
    """Load training CSV for a model and return raw arrays.

    Parameters
    ----------
    slug : str
        Model identifier.
    subsample : int or None
        If given, randomly subsample to this many rows.
    seed : int
        Seed for subsampling reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_rows, n_params), float32
    y : np.ndarray of shape (n_rows, n_summaries), float32
    model : Model
    """
    import pandas as pd

    model = get_model(slug)
    data_slug = model.source_slug or slug
    data_path = Path("data") / data_slug / "train.csv"

    df = pd.read_csv(data_path)
    expected_cols = list(model.param_names) + list(model.summary_names)
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Unexpected columns in {data_path}. "
            f"Expected {expected_cols}, got {list(df.columns)}"
        )

    mask = np.isfinite(df.values).all(axis=1)
    df = df.loc[mask].reset_index(drop=True)

    if subsample is not None and len(df) > subsample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(df), size=subsample, replace=False)
        df = df.iloc[indices].reset_index(drop=True)

    X = df[list(model.param_names)].values.astype(np.float32)
    y = df[list(model.summary_names)].values.astype(np.float32)
    return X, y, model
