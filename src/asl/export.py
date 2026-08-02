"""ONNX export for emulators."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from asl.cholesky import upper_tri_index_pairs
from asl.data import TargetTransform, save_target_transform
from asl.spec import Model


class ExportModel(nn.Module):
    """Wraps a dual-head network with baked-in input scaling."""

    def __init__(
        self,
        trained_net: nn.Module,
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        n_summaries: int,
    ):
        super().__init__()
        self.net = trained_net
        self.n_summaries = n_summaries
        self.register_buffer("x_mean", torch.from_numpy(x_mean.astype(np.float32)))
        self.register_buffer("x_scale", torch.from_numpy(x_scale.astype(np.float32)))

    def forward(self, raw_params: torch.Tensor) -> torch.Tensor:
        x_scaled = (raw_params - self.x_mean) / self.x_scale
        mu_std, chol_raw = self.net(x_scaled)

        chol_upper = chol_raw.clone()
        for k, (i, j) in enumerate(upper_tri_index_pairs(self.n_summaries)):
            if i == j:
                chol_upper[:, k] = torch.nn.functional.softplus(chol_raw[:, k])

        return torch.cat([mu_std, chol_upper], dim=1)


def export_onnx(
    trained_net: nn.Module,
    x_scaler: StandardScaler,
    target_transform: TargetTransform,
    model: Model,
    output_path: Path,
) -> None:
    """Export an emulator to ONNX and persist its target transform."""
    trained_net_cpu = trained_net.cpu().eval()

    export_model = ExportModel(
        trained_net=trained_net_cpu,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        n_summaries=model.n_summaries,
    )
    export_model.eval()

    dummy_input = torch.zeros(1, model.n_params)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    _embed_external_data(output_path)
    save_target_transform(model.slug, target_transform)


def _embed_external_data(onnx_path: Path) -> None:
    """Convert an ONNX file with external data to a single self-contained file."""
    import onnx

    external_data_path = Path(str(onnx_path) + ".data")
    if not external_data_path.exists():
        return

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=False,
    )
    external_data_path.unlink(missing_ok=True)
