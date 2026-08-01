"""
asl.wire -- Wire a trained ONNX emulator into JAGS via JNNX.

Builds a .jnnx package from the trained ONNX model, validates it, compiles
the JAGS module, and installs it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from asl.config import load_config
from asl.data import write_obs_transform_json
from asl.mv import emulator_error_cov_path, load_emulator_error_cov, n_chol
from asl.registry import get_model
from asl.spec import Model


def _supports_sl_package(model: Model) -> bool:
    """True when the model uses multivariate synthetic-likelihood wiring."""
    return model.emulator_output_names is not None


def _output_parameters_with_groups(model: Model) -> list[dict[str, str]]:
    """Attach mean/chol groups for JNNX v2 SL packages."""
    outputs: list[dict[str, str]] = []
    n = model.n_summaries
    for i, name in enumerate(model.output_names):
        group = "mean" if i < n else "chol"
        outputs.append({"name": name, "group": group})
    return outputs


def _write_likelihood_json(model: Model, package_dir: Path, sigma_emu) -> None:
    payload = {
        "version": "1.0",
        "n_summaries": model.n_summaries,
        "sigma_emu": sigma_emu.tolist(),
        "notes": (
            "Debiased mean-head residual covariance in standardized summary space. "
            f"Source: {emulator_error_cov_path(model.slug)}"
        ),
    }
    with open(package_dir / "likelihood.json", "w") as f:
        json.dump(payload, f, indent=2)


def build_jnnx_package(model: Model, onnx_path: Path, package_dir: Path) -> None:
    """Assemble a .jnnx package directory from the ONNX model."""
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, package_dir / "model.onnx")

    function_name = f"{model.slug}_emulator"
    sl_enabled = _supports_sl_package(model)

    metadata: dict = {
        "capabilities": ["emulator"] + (["synthetic_likelihood"] if sl_enabled else []),
        "model_name": model.slug,
        "module_name": function_name,
        "function_name": function_name,
        "version": "2.0.0" if sl_enabled else "1.0.0",
        "input_parameters": [
            {"name": name, "min": float(bounds[0]), "max": float(bounds[1])}
            for name, bounds in zip(model.param_names, model.param_bounds)
        ],
        "output_parameters": (
            _output_parameters_with_groups(model)
            if sl_enabled
            else [{"name": name} for name in model.output_names]
        ),
        "transformations": {
            "input_transform": "identity",
            "output_transforms": ["identity"] * model.n_outputs,
        },
    }
    if sl_enabled:
        metadata["synthetic_likelihood"] = {
            "variant": "n_agnostic_cholesky",
            "n_summaries": model.n_summaries,
            "summary_names": list(model.summary_names),
            "onnx_layout": "concatenated",
            "distribution_name": f"{model.slug}_sl",
            "trial_count_arg": "n_trials",
            "include_sigma_emu": True,
        }
    with open(package_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    scalers = {
        "version": "1.0",
        "input_scaler": {
            "type": "identity",
            "data_min": [float(b[0]) for b in model.param_bounds],
            "data_max": [float(b[1]) for b in model.param_bounds],
        },
        "output_scaler": {
            "type": "identity",
            "data_min": [0.0] * model.n_outputs,
            "data_max": [1.0] * model.n_outputs,
        },
        "metadata": {"note": "Raw I/O - scaling baked into ONNX graph"},
    }
    with open(package_dir / "scalers.json", "w") as f:
        json.dump(scalers, f, indent=2)

    import pickle

    scalers_pkl = {
        "x_min": scalers["input_scaler"]["data_min"],
        "x_max": scalers["input_scaler"]["data_max"],
        "y_min": scalers["output_scaler"]["data_min"],
        "y_max": scalers["output_scaler"]["data_max"],
    }
    with open(package_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers_pkl, f)

    readme = (
        f"# {model.slug} JNNX Package\n\n"
        f"Raw-I/O emulator for the {model.slug} model.\n\n"
        f"## Inputs\n"
        + "\n".join(f"- {n}: [{b[0]}, {b[1]}]" for n, b in zip(model.param_names, model.param_bounds))
        + f"\n\n## Outputs\n"
        + "\n".join(f"- {n}" for n in model.output_names)
        + "\n"
    )
    with open(package_dir / "README.md", "w") as f:
        f.write(readme)

    if sl_enabled:
        sigma_emu = load_emulator_error_cov(model.slug)
        _write_likelihood_json(model, package_dir, sigma_emu)
        write_obs_transform_json(model.slug, model.summary_names, package_dir)
        expected_m = model.n_summaries + n_chol(model.n_summaries)
        if model.n_outputs != expected_m:
            raise RuntimeError(
                f"{model.slug}: expected {expected_m} ONNX outputs for SL, "
                f"got {model.n_outputs}"
            )


def validate_package(package_dir: Path) -> None:
    """Run the JNNX validation suite on a .jnnx package."""
    from jnnx.scripts.validate_jnnx import main as validate_main

    old_argv = sys.argv
    sys.argv = ["validate-jnnx", str(package_dir)]
    try:
        validate_main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(f"JNNX validation failed for {package_dir}") from e
    finally:
        sys.argv = old_argv


def compile_and_install_module(package_dir: Path) -> Path:
    """Generate, compile, and install the JAGS module."""
    ort_dir = str(load_config().get("wire", "onnxruntime_dir", "") or "")
    if not ort_dir:
        ort_dir = os.environ.get("ONNXRUNTIME_DIR", "")
    if not ort_dir:
        raise RuntimeError(
            "wire.onnxruntime_dir not set in asl.toml (or ONNXRUNTIME_DIR env)"
        )

    module_dir = _generate_jags_module_source(package_dir)
    env = os.environ.copy()
    env["ONNXRUNTIME_DIR"] = ort_dir
    _compile_jags_module(module_dir, env)
    _install_jags_module(module_dir, env)
    return module_dir


def _generate_jags_module_source(package_dir: Path) -> Path:
    """Run jnnx generate-module and return the output build directory."""
    from jnnx.scripts.generate_module import main as generate_main

    old_argv = sys.argv
    sys.argv = ["generate-module", str(package_dir)]
    try:
        generate_main()
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(f"Module generation failed for {package_dir}") from e
    finally:
        sys.argv = old_argv

    module_dir = Path("tmp") / f"{package_dir.name}_build"
    if not module_dir.exists():
        raise RuntimeError(f"Expected module directory {module_dir} not found after generation")
    return module_dir


def _compile_jags_module(module_dir: Path, env: dict) -> None:
    """Run make in the generated module directory."""
    result = subprocess.run(
        ["make"], cwd=str(module_dir), env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Module compilation failed: {result.stderr}")


def _install_jags_module(module_dir: Path, env: dict) -> None:
    """Run sudo make install; fall back to LTDL_LIBRARY_PATH if install fails."""
    result = subprocess.run(
        ["sudo", "make", "install"],
        cwd=str(module_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    print(result.stdout, file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    print("[wire] WARNING: sudo make install failed, trying LTDL_LIBRARY_PATH fallback")
    so_files = list(module_dir.glob("*.so")) + list(module_dir.glob(".libs/*.so"))
    if so_files:
        os.environ["LTDL_LIBRARY_PATH"] = str(so_files[0].parent)
        print(f"[wire] Set LTDL_LIBRARY_PATH={so_files[0].parent}")
    else:
        raise RuntimeError("Module install failed and no .so found for fallback")


def _project_fixture_path(slug: str) -> Path | None:
    """Locate SL regression fixture in the ESL repository."""
    for candidate in (
        Path("fixtures") / f"{slug}_sl_regression.json",
        Path("docs/internal") / f"{slug}_sl_regression.json",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def verify_sl_package(model: Model, package_dir: Path, module_dir: Path) -> None:
    """Run JNNX synthetic-likelihood validation when capability is enabled."""
    if not _supports_sl_package(model):
        return

    from jnnx.core import JNNXPackage
    from jnnx.sl_validation import run_sl_validation

    root = Path.cwd().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    fixture_path = _project_fixture_path(model.slug)
    if fixture_path is None:
        print(
            f"[wire] No SL regression fixture for {model.slug}; "
            "skipping SL validation (run compute_sl_logdens_ref --write-fixture)"
        )
        return

    package = JNNXPackage(package_dir)
    passed, total = run_sl_validation(package, module_dir, fixture_path=fixture_path)
    print(f"[wire] SL validation: {passed}/{total} passed")
    if passed < total:
        raise RuntimeError(
            f"SL validation failed for {package_dir}: {passed}/{total} tests passed"
        )


def wire_to_jags(slug: str) -> None:
    """Run the full wiring pipeline for a model."""
    model = get_model(slug)

    onnx_path = Path("results") / slug / "model.onnx"
    if not onnx_path.exists():
        print(f"[wire] FAIL: {onnx_path} not found. Run train first.", file=sys.stderr)
        sys.exit(1)

    package_dir = Path("models") / f"{slug}.jnnx"
    print(f"[wire] Building .jnnx package: {package_dir}")
    build_jnnx_package(model, onnx_path, package_dir)

    print("[wire] Validating package ...")
    validate_package(package_dir)
    print("[wire] Package validated.")

    print("[wire] Compiling and installing JAGS module ...")
    module_dir = compile_and_install_module(package_dir)
    print("[wire] Module installed.")

    if _supports_sl_package(model):
        print("[wire] Verifying synthetic likelihood package ...")
        verify_sl_package(model, package_dir, module_dir)
        print("[wire] PASS: SL validation")
