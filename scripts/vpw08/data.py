"""Load and preprocess Vandekerckhove, Panis, & Wagemans (2007) shape data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.ddm.ddmcollapsesig import SUMMARY_NAMES as COLLAPSE_SUMMARY_NAMES
from models.ddm.ddmcollapsesig import summaries_from_paths

from paths import DATA_PATH

COND_A = np.array([1, 1, 1, 1, 0], dtype=np.int32)
COND_B = np.array([0, 1, 0, 1, 0], dtype=np.int32)
COND_C = np.array([0, 0, 1, 1, 0], dtype=np.int32)

COND_LABELS = {
    1: "Qualitative change in convexity",
    2: "Quantitative change in convexity",
    3: "Qualitative change in concavity",
    4: "Quantitative change in concavity",
    5: "No change",
}


def load_vpw08(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load trial-level data with EZ paper preprocessing."""
    raw = pd.read_csv(path)
    raw.columns = ["sub", "change_quality", "change_type", "noChange", "response", "rt"]
    tmp = raw.loc[raw["rt"] <= 3.0].copy()

    change = 1 - tmp["noChange"].astype(int)
    cond = np.zeros(len(tmp), dtype=np.int32)
    mask = (tmp["change_quality"] == 0) & (tmp["change_type"] == 0)
    cond[mask.to_numpy()] = 1
    mask = (tmp["change_quality"] == 1) & (tmp["change_type"] == 0)
    cond[mask.to_numpy()] = 2
    mask = (tmp["change_quality"] == 0) & (tmp["change_type"] == 1)
    cond[mask.to_numpy()] = 3
    mask = (tmp["change_quality"] == 1) & (tmp["change_type"] == 1)
    cond[mask.to_numpy()] = 4
    cond[change.to_numpy() == 0] = 5

    return pd.DataFrame(
        {
            "sub": tmp["sub"].astype(int).to_numpy(),
            "cond": cond,
            "change": change.to_numpy(),
            "change_quality": tmp["change_quality"].astype(int).to_numpy(),
            "change_type": tmp["change_type"].astype(int).to_numpy(),
            "response": tmp["response"].astype(int).to_numpy(),
            "rt": tmp["rt"].astype(float).to_numpy(),
        }
    )


def ez_summaries(subset: pd.DataFrame) -> np.ndarray:
    """Accuracy, mean RT, and RT variance for one subject x condition cell."""
    if len(subset) < 2:
        return np.full(3, np.nan)
    acc = float(subset["response"].mean())
    rt_mean = float(subset["rt"].mean())
    rt_var = float(subset["rt"].var(ddof=1))
    return np.array([acc, rt_mean, rt_var], dtype=np.float64)


def ddm4_summaries(subset: pd.DataFrame) -> np.ndarray:
    """DDM4 summaries: correct/error RT moments and error rate."""
    if len(subset) < 2:
        return np.full(5, np.nan)
    rts = subset["rt"].astype(float).to_numpy()
    choices = subset["response"].astype(np.int8).to_numpy()
    rts_correct = rts[choices == 1]
    rts_error = rts[choices == 0]
    if len(rts_correct) < 2 or len(rts_error) < 2:
        return np.full(5, np.nan)
    return np.array(
        [
            float(np.mean(rts_correct)),
            float(np.var(rts_correct, ddof=1)),
            float(np.mean(rts_error)),
            float(np.var(rts_error, ddof=1)),
            float(len(rts_error) / len(rts)),
        ],
        dtype=np.float64,
    )


def collapse_summaries(subset: pd.DataFrame) -> np.ndarray:
    """Accuracy, RT tail, and tertile variance summaries for ddmcollapsesig."""
    choices = subset["response"].astype(np.int8).to_numpy()
    summaries = summaries_from_paths(subset["rt"].astype(float).to_numpy(), choices)
    return summaries.astype(np.float64)


def build_cells(df: pd.DataFrame | None = None, *, summary: str = "ez") -> dict:
    """Aggregate to subject x condition cells (45 cells for 9 x 5 design)."""
    if summary not in ("ez", "ddm4", "collapse"):
        raise ValueError(f"Unknown summary mode: {summary}")

    if df is None:
        df = load_vpw08()

    if summary == "ez":
        summarize = ez_summaries
    elif summary == "ddm4":
        summarize = ddm4_summaries
    else:
        summarize = collapse_summaries

    rows: list[dict] = []
    subjects = sorted(df["sub"].unique())
    subj_map = {sid: i + 1 for i, sid in enumerate(subjects)}

    for (sid, cond), part in df.groupby(["sub", "cond"], sort=True):
        raw = summarize(part)
        if not np.all(np.isfinite(raw)):
            continue
        rows.append(
            {
                "subj": subj_map[sid],
                "cond": int(cond),
                "n_trials": len(part),
                "obs_raw": raw,
            }
        )

    if len(rows) != 45:
        raise ValueError(f"Expected 45 cells, got {len(rows)}")

    obs_raw = np.vstack([r.pop("obs_raw") for r in rows])
    out = {
        "N_SUBJ": len(subjects),
        "K": len(rows),
        "subj": np.array([r["subj"] for r in rows], dtype=np.int32),
        "cond": np.array([r["cond"] for r in rows], dtype=np.int32),
        "n_trials": np.array([r["n_trials"] for r in rows], dtype=np.int32),
        "obs_raw": obs_raw,
        "A": COND_A,
        "B": COND_B,
        "C": COND_C,
        "cond_labels": COND_LABELS,
        "summary_mode": summary,
    }
    if summary == "ddm4":
        from models.ddm.ddm4 import DDM4

        out["summary_names"] = list(DDM4.summary_names)
    elif summary == "collapse":
        out["summary_names"] = list(COLLAPSE_SUMMARY_NAMES)
    return out
