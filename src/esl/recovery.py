"""Recovery study helpers shared by scalar and multivariate pipelines."""

import os
import time

N_CHAINS = 4
N_SUBJECTS_FULL = 500
N_SUBJECTS_SMOKE = 50
N_TRIALS_RECOVERY_FULL = 500
N_TRIALS_RECOVERY_SMOKE = 500
N_MCMC_ITER_FULL = 5000
N_MCMC_ITER_SMOKE = 5000
N_BURNIN_FULL = 2000
N_BURNIN_SMOKE = 2000

COVERAGE_TARGET = 0.95
COVERAGE_LO = 0.90
COVERAGE_HI = 0.99


def _format_duration(seconds: float) -> str:
    """Format seconds as a short human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def format_recovery_progress(
    done: int,
    total: int,
    n_converged: int,
    n_failed: int,
    t0: float,
) -> str:
    """Build a progress line with elapsed time and ETA."""
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else None
    line = (
        f"[recovery] {done}/{total} done, {n_converged} converged, {n_failed} failed"
        f" | elapsed {_format_duration(elapsed)}"
        f" | {rate:.2f} subj/s"
    )
    if remaining is not None:
        line += f" | ETA ~{_format_duration(remaining)}"
    return line


def recovery_report_interval(n_subjects: int) -> int:
    """Subjects between progress reports (default: every 10)."""
    if "ESL_PROGRESS_EVERY" in os.environ:
        return max(1, int(os.environ["ESL_PROGRESS_EVERY"]))
    return max(1, min(10, n_subjects // 50))


def check_coverage_gate(coverages: list[float], param_names: tuple[str, ...]) -> None:
    """Exit non-zero if any parameter coverage is outside (LO, HI)."""
    import sys

    for i, name in enumerate(param_names):
        cov = coverages[i]
        if cov <= COVERAGE_LO or cov >= COVERAGE_HI:
            print(
                f"[recovery] FAIL: {name} 95% CI coverage {cov:.3f} "
                f"outside ({COVERAGE_LO:.0%}, {COVERAGE_HI:.0%}) "
                f"(target ~{COVERAGE_TARGET:.0%})",
                file=sys.stderr,
            )
            sys.exit(1)


def resolve_recovery_settings() -> dict:
    """Determine recovery study size from environment."""
    is_smoke = os.environ.get("ESL_SMOKE", "0") == "1"
    n_subjects = N_SUBJECTS_SMOKE if is_smoke else N_SUBJECTS_FULL
    n_trials = N_TRIALS_RECOVERY_SMOKE if is_smoke else N_TRIALS_RECOVERY_FULL
    if "ESL_N_SUBJECTS" in os.environ:
        n_subjects = int(os.environ["ESL_N_SUBJECTS"])
    if "ESL_N_TRIALS" in os.environ:
        n_trials = int(os.environ["ESL_N_TRIALS"])
    return {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "n_iter": N_MCMC_ITER_SMOKE if is_smoke else N_MCMC_ITER_FULL,
        "n_burnin": N_BURNIN_SMOKE if is_smoke else N_BURNIN_FULL,
        "n_chains": N_CHAINS,
    }
