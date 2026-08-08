"""Emulator training (dual-head, covariance-aware)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from asl.config import load_config
from asl.cov_data import SEED_DEFAULT, load_cov_dataset, load_cov_settings
from asl.data import TargetTransform, save_target_transform, summary_column_masks
from asl.export import export_onnx
from asl.mlp import DEVICE, build_architecture, resolve_training_settings
from asl.cholesky import (
    cov_stein_loss,
    debias_emulator_error_cov,
    n_chol,
    save_emulator_error_cov,
    std_cov_from_logcov,
    unpack_upper_tri,
)
from asl.spec import Model

COV_LAMBDA = 1.0
VAL_FRACTION = 0.2


class DualHeadNet(nn.Module):
    """Shared backbone with separate mean and Cholesky heads."""

    def __init__(self, base_net: nn.Module, n_summaries: int):
        super().__init__()
        width = base_net.head.in_features
        base_net.head = nn.Identity()
        self.feature_extractor = base_net
        self.mean_head = nn.Linear(width, n_summaries)
        self.chol_head = nn.Linear(width, n_chol(n_summaries))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        return self.mean_head(features), self.chol_head(features)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_target_transform(rt_mask: np.ndarray, z_mean: np.ndarray) -> TargetTransform:
    target_transform = TargetTransform(rt_mask)
    target_transform.scaler.fit(z_mean)
    return target_transform


def build_C1_std_array(
    C1_z: np.ndarray, scale: np.ndarray, n_summaries: int
) -> np.ndarray:
    rows = []
    for flat in C1_z:
        C1_log = unpack_upper_tri(flat, n_summaries)
        rows.append(std_cov_from_logcov(C1_log, scale))
    return np.stack(rows, axis=0).astype(np.float32)


def split_train_val(n_rows: int, val_fraction: float = VAL_FRACTION) -> tuple[int, int]:
    """Deterministic row split: first (1-f) rows train, remainder validation."""
    n_val = max(1, int(round(n_rows * val_fraction)))
    n_train = n_rows - n_val
    if n_train < 1:
        raise ValueError(f"Need at least 2 rows for train/val split, got {n_rows}")
    return n_train, n_val


def train_one_epoch(
    model: DualHeadNet,
    optimizer: torch.optim.Optimizer,
    X_tensor: torch.Tensor,
    mu_tensor: torch.Tensor,
    C1_tensor: torch.Tensor,
    n_summaries: int,
    batch_size: int,
    cov_lambda: float = COV_LAMBDA,
    seed: int = SEED_DEFAULT,
) -> None:
    model.train()
    n = X_tensor.shape[0]
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    indices = torch.randperm(n, device=DEVICE, generator=generator)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        xb = X_tensor[batch_idx]
        mu_target = mu_tensor[batch_idx]
        C1_batch = C1_tensor[batch_idx]

        mu_pred, chol_raw = model(xb)
        mean_loss = nn.functional.mse_loss(mu_pred, mu_target)
        cov_loss = cov_stein_loss(chol_raw, C1_batch, n_summaries)
        loss = mean_loss + cov_lambda * cov_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate_mean_r2(
    model: DualHeadNet,
    X_tensor: torch.Tensor,
    y_raw: np.ndarray,
    target_transform: TargetTransform,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        mu_std, _ = model(X_tensor)
    pred_raw = target_transform.inverse_transform(mu_std.cpu().numpy())

    ss_res = np.sum((y_raw - pred_raw) ** 2, axis=0)
    ss_tot = np.sum((y_raw - y_raw.mean(axis=0)) ** 2, axis=0)
    per_target_r2 = 1.0 - ss_res / ss_tot
    overall_r2 = 1.0 - ss_res.sum() / ss_tot.sum()
    return float(overall_r2), per_target_r2


def evaluate_cov_stein(
    model: DualHeadNet,
    X_tensor: torch.Tensor,
    C1_tensor: torch.Tensor,
    n_summaries: int,
) -> float:
    model.eval()
    with torch.no_grad():
        _, chol_raw = model(X_tensor)
        return float(cov_stein_loss(chol_raw, C1_tensor, n_summaries).item())


def compute_emulator_error_cov(
    net: DualHeadNet,
    X_tensor: torch.Tensor,
    mu_std_targets: np.ndarray,
    mean_C1_std: np.ndarray,
    n_rep: int,
    n_replicates: int,
) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        mu_pred, _ = net(X_tensor)
    residuals = mu_pred.cpu().numpy() - mu_std_targets
    if residuals.shape[0] < 2:
        raise ValueError("Need at least two rows to estimate emulator error covariance.")
    raw_cov = np.cov(residuals, rowvar=False, bias=False)
    return debias_emulator_error_cov(raw_cov, mean_C1_std, n_rep, n_replicates)


def retrain_dual_head_model(
    build_fn,
    X_train: np.ndarray,
    z_mean_train: np.ndarray,
    C1_std_train: np.ndarray,
    rt_mask: np.ndarray,
    n_summaries: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    cov_lambda: float = COV_LAMBDA,
    seed: int = SEED_DEFAULT,
) -> tuple[DualHeadNet, StandardScaler, TargetTransform]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_scaler = StandardScaler()
    X_s = x_scaler.fit_transform(X_train).astype(np.float32)

    target_transform = build_target_transform(rt_mask, z_mean_train)
    mu_std = target_transform.scaler.transform(z_mean_train).astype(np.float32)

    X_tensor = torch.from_numpy(X_s).to(DEVICE)
    mu_tensor = torch.from_numpy(mu_std).to(DEVICE)
    C1_tensor = torch.from_numpy(C1_std_train).to(DEVICE)

    base_net = build_fn().to(DEVICE)
    net = DualHeadNet(base_net, n_summaries).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    t_train = time.monotonic()
    for epoch in range(n_epochs):
        train_one_epoch(
            net,
            optimizer,
            X_tensor,
            mu_tensor,
            C1_tensor,
            n_summaries,
            batch_size,
            cov_lambda=cov_lambda,
            seed=seed + epoch,
        )
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            elapsed = time.monotonic() - t_train
            print(f"[train]   epoch {epoch + 1}/{n_epochs} ({elapsed:.0f}s)")

    net.eval()
    return net, x_scaler, target_transform


def train_emulator(model: Model) -> None:
    """Train a fixed-architecture dual-head emulator and export ONNX."""
    config = load_config()
    slug = model.slug
    settings = resolve_training_settings()
    rt_mask, _ = summary_column_masks(model)
    cov_lambda = float(config.get("training", "covariance_loss_weight", COV_LAMBDA))
    seed = int(config.get("training", "random_seed", SEED_DEFAULT))

    arch_name = config.get("training", "architecture") or model.default_architecture
    if not arch_name:
        print("[train] FAIL: No architecture specified.", file=sys.stderr)
        sys.exit(1)

    print(f"[train] Model: {slug}")
    print(f"[train] Device: {DEVICE}")
    print(f"[train] Architecture: {arch_name}")
    print(f"[train] Settings: {settings}")
    print(f"[train] Random seed: {seed}")
    print(f"[train] Covariance loss weight: {cov_lambda}")

    X, z_mean, C1_z, y_raw, _ = load_cov_dataset(model, seed=seed)
    n_train, _ = split_train_val(len(X))
    print(
        f"[train] Loaded {X.shape[0]} rows ({n_train} train, {len(X) - n_train} val), "
        f"{X.shape[1]} params, {z_mean.shape[1]} summaries"
    )

    X_train, X_val = X[:n_train], X[n_train:]
    z_mean_train, z_mean_val = z_mean[:n_train], z_mean[n_train:]
    C1_z_train, C1_z_val = C1_z[:n_train], C1_z[n_train:]
    y_raw_train, y_raw_val = y_raw[:n_train], y_raw[n_train:]

    build_fn = build_architecture(arch_name, model.n_params, model.n_summaries)
    results_dir = Path("results") / slug
    results_dir.mkdir(parents=True, exist_ok=True)

    target_transform = build_target_transform(rt_mask, z_mean_train)
    C1_std_train = build_C1_std_array(
        C1_z_train, target_transform.scaler.scale_, model.n_summaries
    )
    C1_std_val = build_C1_std_array(
        C1_z_val, target_transform.scaler.scale_, model.n_summaries
    )

    print("[train] Training dual-head model ...")
    t_train_start = time.monotonic()
    final_net, x_scaler, target_transform = retrain_dual_head_model(
        build_fn=build_fn,
        X_train=X_train,
        z_mean_train=z_mean_train,
        C1_std_train=C1_std_train,
        rt_mask=rt_mask,
        n_summaries=model.n_summaries,
        n_epochs=settings["n_epochs"],
        batch_size=settings["batch_size"],
        lr=settings["lr"],
        cov_lambda=cov_lambda,
        seed=seed,
    )
    train_time_seconds = time.monotonic() - t_train_start

    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_val_tensor = torch.from_numpy(X_val_s).to(DEVICE)
    C1_val_tensor = torch.from_numpy(C1_std_val).to(DEVICE)

    train_r2, per_train_r2 = evaluate_mean_r2(
        final_net,
        torch.from_numpy(x_scaler.transform(X_train).astype(np.float32)).to(DEVICE),
        y_raw_train,
        target_transform,
    )
    val_r2, per_val_r2 = evaluate_mean_r2(
        final_net, X_val_tensor, y_raw_val, target_transform
    )
    cov_stein_val = evaluate_cov_stein(
        final_net, X_val_tensor, C1_val_tensor, model.n_summaries
    )
    mu_std_val = target_transform.scaler.transform(z_mean_val).astype(np.float64)
    n_rep, n_replicates = load_cov_settings(slug)
    mean_C1_std = np.mean(C1_std_val, axis=0)
    sigma_emu = compute_emulator_error_cov(
        final_net,
        X_val_tensor,
        mu_std_val,
        mean_C1_std,
        n_rep,
        n_replicates,
    )
    save_emulator_error_cov(slug, sigma_emu, n_rep=n_rep, n_replicates=n_replicates)
    print(f"[train] Emulator error cov diag: {np.diag(sigma_emu).tolist()}")
    print(
        f"[train] Mean-head R^2: train={train_r2:.6f}, val={val_r2:.6f}, "
        f"per_summary_val={per_val_r2.tolist()}"
    )
    print(f"[train] Validation Stein cov loss: {cov_stein_val:.6f}")

    summary = {
        "architecture": arch_name,
        "random_seed": seed,
        "n_train": n_train,
        "n_val": len(X_val),
        "train_r2": train_r2,
        "val_r2": val_r2,
        "per_summary_r2_train": dict(zip(model.summary_names, per_train_r2.tolist())),
        "per_summary_r2_val": dict(zip(model.summary_names, per_val_r2.tolist())),
        "cov_stein_loss_val": cov_stein_val,
        "cov_lambda": cov_lambda,
        "train_time_seconds": train_time_seconds,
        "n_params": final_net.count_trainable_parameters(),
        "summary_names": list(model.summary_names),
        "output_names": list(model.output_names),
    }
    with open(results_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    onnx_path = results_dir / "model.onnx"
    export_onnx(final_net, x_scaler, target_transform, model, onnx_path)
    save_target_transform(slug, target_transform)
    print(f"[train] Exported ONNX: {onnx_path}")

    threshold = float(config.get("training", "mean_r2_threshold", 0.999))
    if val_r2 < threshold:
        print(
            f"[train] FAIL: Validation mean-head R^2 = {val_r2:.6f} < {threshold}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[train] PASS: validation mean-head R^2 = {val_r2:.6f} >= {threshold}")
