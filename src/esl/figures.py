"""Recovery diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from esl.spec import Model


def plot_recovery_diagnostics(
    model: Model,
    true_params: np.ndarray,
    estimated_params: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    output_path: Path,
) -> None:
    """Create a PDF with true-vs-estimated parameter recovery scatter."""
    n_params = model.n_params
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.errorbar(
            true_params[:, i],
            estimated_params[:, i],
            yerr=[
                estimated_params[:, i] - ci_lower[:, i],
                ci_upper[:, i] - estimated_params[:, i],
            ],
            fmt="o",
            markersize=3,
            alpha=0.6,
            capsize=2,
        )
        lo = min(true_params[:, i].min(), estimated_params[:, i].min())
        hi = max(true_params[:, i].max(), estimated_params[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel(f"True {model.param_names[i]}")
        ax.set_ylabel(f"Estimated {model.param_names[i]}")

        coverage = np.mean(
            (true_params[:, i] >= ci_lower[:, i])
            & (true_params[:, i] <= ci_upper[:, i])
        )
        ax.set_title(f"95% CI coverage: {coverage:.2f}")

    fig.suptitle(f"{model.slug} recovery", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
