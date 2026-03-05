"""
Raspberry Pi AI Hat+ latency test: run models of varying sizes and measure inference latency.

Model sizes follow: 20_000 params -> then x5, x2, x5, x2, ... up to ~1B.
Weights are randomized; this script measures latency only, not accuracy.

Usage:
    python latency_test_ai_hat.py [--runs 100] [--warmup 10] [--max-params 1e9] [--output results.csv]

Requirements:
    pip install torch numpy
"""

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def get_target_sizes(max_params: int = 1_000_000_000) -> list[int]:
    """Model sizes: start 20k, then x5, x2, x5, x2, ... up to max_params."""
    out = [20_000]
    mults = [5, 2]
    i = 0
    while out[-1] < max_params:
        out.append(out[-1] * mults[i % 2])
        i += 1
    # clip last if over
    if out[-1] > max_params:
        out[-1] = int(max_params)
    return out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_model_with_params(target_params: int, seed: int = 42) -> nn.Module:
    """
    Build a single-layer MLP with approximately target_params parameters.
    Uses one Linear(d, d) so that d*(d+1) ≈ target_params (d = sqrt-ish).
    Weights are randomized.
    """
    # d*(d+1) = target => d^2 + d - target = 0 => d = (-1 + sqrt(1+4*target))/2
    d = max(1, int((-1 + math.sqrt(1 + 4 * target_params)) // 2))
    actual = d * (d + 1)
    # optionally add a tiny second layer to get closer to target if we're under
    if actual < target_params and (target_params - actual) > 100:
        extra = target_params - actual  # need one more layer with ~extra params
        # out_features * (d + 1) ≈ extra => out_features ≈ extra // (d+1)
        out2 = max(1, extra // (d + 1))
        module = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, out2),
        )
    else:
        module = nn.Sequential(nn.Linear(d, d))
    torch.manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module, d


def run_latency_test(
    model: nn.Module,
    input_dim: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Run warmup then timed forward passes; return latency stats in milliseconds."""
    model.eval()
    dummy = torch.randn(1, input_dim, device=device, dtype=torch.float32)
    # warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # timed runs
    latencies_ms = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _hailo_latency_for_hef(
    hef_path: Path,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Measure latency for a compiled Hailo model (.hef) running on the NPU.

    This uses HailoRT via the `hailo_platform` Python package, which must be
    installed separately using the official Hailo SDK / `hailo-all` packages.
    """
    try:
        import hailo_platform as hpf
    except ImportError as exc:
        raise RuntimeError(
            "hailo_platform (HailoRT Python API) is not available. "
            "Make sure HailoRT is installed on this machine (e.g. `sudo apt install hailo-all`)."
        ) from exc

    hef = hpf.HEF(str(hef_path))

    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )

        input_shape = input_vstream_info.shape

        # Use a fixed random input to minimize host-side overhead.
        random_input = np.random.rand(*input_shape).astype(np.float32)
        input_data = {input_vstream_info.name: np.expand_dims(random_input, axis=0)}

        latencies_ms: list[float] = []

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params,
            ) as infer_pipeline:
                # Warmup
                for _ in range(num_warmup):
                    _ = infer_pipeline.infer(input_data)

                # Timed runs
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = infer_pipeline.infer(input_data)
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _parse_params_from_hef_name(hef_path: Path) -> int | None:
    """
    Best-effort extraction of parameter count from HEF filename.

    If your HEF files are named like `model_20000.hef` or `net-100M.hef`,
    this will pull out the first integer substring (e.g. 20000 or 100).
    """
    m = re.search(r"(\\d+)", hef_path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def export_synthetic_mlp_to_onnx(target_params: int, onnx_path: Path) -> None:
    """
    Export a synthetic MLP (matching our param-count logic) to ONNX for Hailo compilation.
    """
    model, input_dim = build_model_with_params(target_params)
    model.eval()
    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a relatively recent ONNX opset to avoid version‑conversion issues.
    # Hailo's compiler will tell you if it needs a different opset.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )


def generate_hailo_models(
    target_sizes: list[int],
    onnx_dir: Path,
    hef_dir: Path,
    compile_template: str | None,
) -> None:
    """
    Generate ONNX models for each target size, and optionally compile them to HEF.

    The compile_template is a shell command string that can contain:
        {onnx}   - full path to the ONNX file
        {hef_dir} - directory where HEF files should be written
        {params} - target parameter count (integer)

    Example (hailomz-based, adjust to your environment):
        --hailo-compile-template \\
          "hailomz compile mlp_{params} --ckpt={onnx} --hw-arch hailo8l \\
             --calib-path /path/to/calib --performance --output-dir {hef_dir}"
    """
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hef_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic ONNX models in {onnx_dir} for sizes: {target_sizes}")
    if compile_template:
        print("Will also invoke Hailo compile command template per model.")
        print("Template (placeholders: {onnx}, {hef_dir}, {params}):")
        print(f"  {compile_template}")

    for params in target_sizes:
        onnx_path = onnx_dir / f"mlp_{params}.onnx"
        export_synthetic_mlp_to_onnx(params, onnx_path)
        print(f"Exported ONNX for {params} params -> {onnx_path}")

        if compile_template:
            cmd = compile_template.format(
                onnx=str(onnx_path),
                hef_dir=str(hef_dir),
                params=params,
            )
            print(f"Running compile command: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(
                    f"WARNING: compile command failed (exit code {result.returncode}) for {onnx_path}",
                    file=sys.stderr,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test inference latency for models of different sizes (e.g. on Raspberry Pi AI Hat+)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed inference runs per model (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs per model (default: 10).",
    )
    parser.add_argument(
        "--max-params",
        type=float,
        default=1e9,
        help="Maximum model size in parameters (default: 1e9). Reduce on low-memory devices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: 'cpu', 'cuda', or 'mps'. Default: auto-detect.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "hailo"],
        default="torch",
        help="Backend to use: 'torch' (default) for CPU/GPU, or 'hailo' to run compiled .hef models on the Hailo NPU.",
    )
    parser.add_argument(
        "--hailo-hefs-dir",
        type=str,
        default="",
        help="Directory containing .hef files to benchmark when backend='hailo'.",
    )
    parser.add_argument(
        "--generate-hailo-models",
        action="store_true",
        help="Generate synthetic ONNX models for all target sizes (and optionally compile them with Hailo).",
    )
    parser.add_argument(
        "--hailo-onnx-dir",
        type=str,
        default="hailo_onnx",
        help="Directory to write synthetic ONNX models when --generate-hailo-models is used.",
    )
    parser.add_argument(
        "--hailo-compile-template",
        type=str,
        default="",
        help=(
            "Optional shell command template to compile ONNX to HEF. "
            "Placeholders: {onnx} (ONNX path), {hef_dir} (output HEF dir), {params} (target param count). "
            "Example: 'hailomz compile mlp_{params} --ckpt={onnx} --hw-arch hailo8l "
            "--calib-path /path/to/calib --performance --output-dir {hef_dir}'"
        ),
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Latency benchmark starting...")
    print(f"Backend requested: {args.backend}")
    print(f"Max parameter budget: {int(args.max_params):,}")
    print("=" * 80)

    max_params = int(args.max_params)
    target_sizes = get_target_sizes(max_params)
    print(f"Computed target parameter sizes: {target_sizes}")

    if args.generate_hailo_models:
        onnx_dir = Path(args.hailo_onnx_dir)
        if args.hailo_hefs_dir:
            hef_dir = Path(args.hailo_hefs_dir)
        else:
            hef_dir = Path("hailo_hefs")
            # propagate the default so the Hailo backend can pick it up later
            args.hailo_hefs_dir = str(hef_dir)

        compile_template = args.hailo_compile_template or None
        generate_hailo_models(target_sizes, onnx_dir, hef_dir, compile_template)
        print(f"Finished generating Hailo models for {len(target_sizes)} sizes.")

    results: list[dict] = []

    if args.backend == "hailo":
        if not args.hailo_hefs_dir:
            print("--hailo-hefs-dir is required when backend='hailo'.", file=sys.stderr)
            sys.exit(1)

        hef_dir = Path(args.hailo_hefs_dir)
        if not hef_dir.is_dir():
            print(f"HEF directory does not exist: {hef_dir}", file=sys.stderr)
            sys.exit(1)

        hef_paths = sorted(hef_dir.glob("*.hef"))
        if not hef_paths:
            print(f"No .hef files found in {hef_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Backend: Hailo NPU (HailoRT)")
        print(f"HEF directory: {hef_dir}")
        print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
        print("-" * 80)

        total = len(hef_paths)
        for idx, hef_path in enumerate(hef_paths, start=1):
            print(f"[{idx}/{total}] Running Hailo latency test for HEF: {hef_path.name}")
            try:
                stats = _hailo_latency_for_hef(
                    hef_path,
                    num_warmup=args.warmup,
                    num_runs=args.runs,
                )
                param_hint = _parse_params_from_hef_name(hef_path)
                row = {
                    "hef_name": hef_path.name,
                    "param_hint_from_name": param_hint if param_hint is not None else "",
                    **stats,
                }
                results.append(row)
                print(
                    f"HEF: {hef_path.name:<32} | "
                    f"Mean: {stats['mean_ms']:>8.2f} ms | "
                    f"Median: {stats['median_ms']:>8.2f} ms | "
                    f"P95: {stats['p95_ms']:>8.2f} ms"
                )
            except RuntimeError as e:
                print(f"HEF: {hef_path.name:<32} | ERROR: {e}", file=sys.stderr)
    else:
        if args.device:
            if args.device == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available; using CPU.", file=sys.stderr)
                device = torch.device("cpu")
            elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("MPS requested but not available; using CPU.", file=sys.stderr)
                device = torch.device("cpu")
            else:
                device = torch.device(args.device)
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        print(f"Backend: PyTorch ({device})")
        print(f"Target model sizes (params): {target_sizes}")
        print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
        print("-" * 80)

        total = len(target_sizes)
        for idx, target in enumerate(target_sizes, start=1):
            print(f"[{idx}/{total}] Building and benchmarking PyTorch MLP with ~{target:,} params...")
            try:
                model, input_dim = build_model_with_params(target)
                actual_params = count_parameters(model)
                model = model.to(device)
                stats = run_latency_test(
                    model,
                    input_dim,
                    device,
                    num_warmup=args.warmup,
                    num_runs=args.runs,
                )
                row = {
                    "target_params": target,
                    "actual_params": actual_params,
                    **stats,
                }
                results.append(row)
                print(
                    f"Params: {actual_params:>12,}  |  "
                    f"Mean: {stats['mean_ms']:>8.2f} ms  |  "
                    f"Median: {stats['median_ms']:>8.2f} ms  |  "
                    f"P95: {stats['p95_ms']:>8.2f} ms"
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Params: {target:>12,}  |  OOM (skipping larger models).")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

    return results


if __name__ == "__main__":
    main()
