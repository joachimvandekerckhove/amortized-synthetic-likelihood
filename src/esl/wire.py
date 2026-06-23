"""
esl.wire -- Wire a trained ONNX emulator into JAGS via JNNX.

Builds a .jnnx package from the trained ONNX model, validates it, compiles
the JAGS module, installs it, and verifies that the JAGS deterministic node
produces the same output as the ONNX model.

Usage:
    python -m esl.wire <slug>
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from esl.registry import get_model
from esl.spec import Model


def build_jnnx_package(model: Model, onnx_path: Path, package_dir: Path) -> None:
    """Assemble a .jnnx package directory from the ONNX model.

    Creates metadata.json, copies model.onnx, and writes scalers.json with
    identity transforms (since our ONNX is already raw-I/O).

    Parameters
    ----------
    model : Model
    onnx_path : Path to the trained ONNX file
    package_dir : Path for the output .jnnx directory
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, package_dir / "model.onnx")

    function_name = f"{model.slug}_emulator"

    metadata = {
        "model_name": model.slug,
        "module_name": function_name,
        "function_name": function_name,
        "version": "1.0.0",
        "input_parameters": [
            {"name": name, "min": float(bounds[0]), "max": float(bounds[1])}
            for name, bounds in zip(model.param_names, model.param_bounds)
        ],
        "output_parameters": [
            {"name": name} for name in model.output_names
        ],
        "transformations": {
            "input_transform": "identity",
            "output_transforms": ["identity"] * model.n_outputs,
        },
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

    # Write scalers.pkl for JNNX compatibility
    import pickle

    scalers_pkl = {
        "x_min": scalers["input_scaler"]["data_min"],
        "x_max": scalers["input_scaler"]["data_max"],
        "y_min": scalers["output_scaler"]["data_min"],
        "y_max": scalers["output_scaler"]["data_max"],
    }
    with open(package_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers_pkl, f)

    # Write a minimal README
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


def validate_package(package_dir: Path) -> None:
    """Run the JNNX validation suite on a .jnnx package.

    Parameters
    ----------
    package_dir : Path

    Raises
    ------
    RuntimeError
        If validation fails.
    """
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
    """Generate, compile, and install the JAGS module.

    Parameters
    ----------
    package_dir : Path to the .jnnx package

    Returns
    -------
    module_dir : Path to the generated module source directory

    Raises
    ------
    RuntimeError
        If compilation or installation fails.
    """
    ort_dir = os.environ.get("ONNXRUNTIME_DIR", "")
    if not ort_dir:
        raise RuntimeError("ONNXRUNTIME_DIR environment variable not set")

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


def verify_jags_node_matches_onnx(model: Model, package_dir: Path) -> float:
    """Verify the JAGS module produces the same output as ONNX.

    Runs a small JAGS model that evaluates the deterministic node and
    compares to direct ONNX inference.

    Parameters
    ----------
    model : Model
    package_dir : Path to the .jnnx package

    Returns
    -------
    max_diff : float
        Maximum absolute difference between JAGS and ONNX outputs.

    Raises
    ------
    RuntimeError
        If the difference exceeds tolerance or JAGS fails.
    """
    onnx_path = package_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_path))

    # Generate test points from middle of parameter space
    rng = np.random.default_rng(123)
    n_test = 5
    test_params = np.empty((n_test, model.n_params), dtype=np.float32)
    for i, (lo, hi) in enumerate(model.param_bounds):
        test_params[:, i] = rng.uniform(lo, hi, size=n_test)

    # ONNX predictions
    onnx_preds = session.run(None, {"input": test_params})[0]

    # JAGS verification via py2jags
    function_name = f"{model.slug}_emulator"
    max_diff = _compare_via_jags(model, function_name, test_params, onnx_preds)

    print(f"[wire] JAGS vs ONNX max diff: {max_diff:.2e}")
    if max_diff > 0.01:
        raise RuntimeError(
            f"JAGS node differs from ONNX by {max_diff:.2e} (threshold: 0.01)"
        )
    return max_diff


def _compare_via_jags(
    model: Model,
    function_name: str,
    test_params: np.ndarray,
    onnx_preds: np.ndarray,
) -> float:
    """Run JAGS with fixed parameters and compare deterministic node output.

    Parameters
    ----------
    model : Model
    function_name : str
    test_params : np.ndarray of shape (n_test, n_params)
    onnx_preds : np.ndarray of shape (n_test, n_outputs)

    Returns
    -------
    max_diff : float
    """
    from py2jags import run_jags

    max_diff = 0.0
    module_name = function_name

    for row_idx in range(test_params.shape[0]):
        params = test_params[row_idx]
        param_assignments = "\n    ".join(
            f"{name} <- {float(params[i])}"
            for i, name in enumerate(model.param_names)
        )
        param_args = ", ".join(model.param_names)

        jags_model = f"""model {{
    {param_assignments}
    pred[1:{model.n_outputs}] <- {function_name}({param_args})
}}
"""
        try:
            result = run_jags(
                model_string=jags_model,
                data_dict={"dummy": 1.0},
                monitorparams=["pred"],
                nchains=1,
                nsamples=5,
                nburnin=0,
                thin=1,
                modules=[module_name],
            )
            jags_pred = np.array([
                result.get_samples(f"pred_{i+1}").mean()
                for i in range(model.n_outputs)
            ])
            diff = np.max(np.abs(jags_pred - onnx_preds[row_idx]))
            max_diff = max(max_diff, diff)
        except Exception as e:
            raise RuntimeError(f"JAGS verification failed on test row {row_idx}: {e}") from e

    return max_diff


def wire_to_jags(slug: str) -> None:
    """Run the full wiring pipeline for a model.

    Parameters
    ----------
    slug : str
        Model identifier.
    """
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
    compile_and_install_module(package_dir)
    print("[wire] Module installed.")

    print("[wire] Verifying JAGS node matches ONNX ...")
    max_diff = verify_jags_node_matches_onnx(model, package_dir)
    print(f"[wire] PASS: max_diff = {max_diff:.2e}")


def main() -> None:
    """Entry point for python -m esl.wire <slug>."""
    if len(sys.argv) != 2:
        print("Usage: python -m esl.wire <model-slug>", file=sys.stderr)
        sys.exit(1)
    wire_to_jags(sys.argv[1])


if __name__ == "__main__":
    main()
