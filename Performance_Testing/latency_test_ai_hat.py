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
        "--output",
        type=str,
        default="",
        help="Optional path to save CSV results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: 'cpu', 'cuda', or 'mps'. Default: auto-detect.",
    )
    args = parser.parse_args()

    max_params = int(args.max_params)
    target_sizes = get_target_sizes(max_params)

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

    print(f"Device: {device}")
    print(f"Target model sizes (params): {target_sizes}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
    print("-" * 80)

    results = []
    for target in target_sizes:
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

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            headers = ["target_params", "actual_params", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p95_ms"]
            f.write(",".join(headers) + "\n")
            for r in results:
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
        print(f"\nResults written to {out_path}")

    return results


if __name__ == "__main__":
    main()
