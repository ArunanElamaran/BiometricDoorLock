#!/usr/bin/env python3
"""
Convert PyTorch .pt model files to ONNX.

Reads .pt files from a given folder (full model saves or dict with "model" key),
infers input shape from the first nn.Linear layer when possible, and exports
to ONNX in a specified output directory.

Requires: torch
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load_model(pt_path: Path) -> nn.Module:
    """Load an nn.Module from a .pt file. Supports full model save or dict with 'model' key."""
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        return obj["model"]
    raise ValueError(
        f"Unsupported .pt format in {pt_path}. "
        "Expected a saved nn.Module or a dict with key 'model' containing the module. "
        "State-dict-only checkpoints are not supported."
    )


def _infer_input_dim(model: nn.Module) -> int:
    """Infer input feature size from the first nn.Linear layer."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError(
        "Could not infer input shape from model (no nn.Linear layer). "
        "Use a model with at least one Linear layer, or extend this script to accept --input-shape."
    )


def convert_pt_to_onnx(pt_path: Path, out_path: Path, opset_version: int = 18) -> None:
    """Load a .pt model and export it to ONNX at out_path."""
    model = _load_model(pt_path)
    model.eval()
    input_dim = _infer_input_dim(model)
    dummy = torch.randn(1, input_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pt model files to ONNX."
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Folder containing .pt files to convert.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "If set, convert only this exact .pt filename (e.g. mlp_20000.pt). "
            "Otherwise convert all .pt files in --pt-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Folder to write ONNX files into (created if it does not exist).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18).",
    )
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    if not pt_dir.is_dir():
        print(f"PT directory not found: {pt_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        single = pt_dir / args.model
        if not single.is_file():
            print(f"Model file not found: {single}", file=sys.stderr)
            sys.exit(1)
        pt_files = [single]
    else:
        pt_files = sorted(pt_dir.glob("*.pt"))
        if not pt_files:
            print(f"No .pt files found in {pt_dir}", file=sys.stderr)
            sys.exit(1)

    for pt_path in pt_files:
        onnx_path = out_dir / f"{pt_path.stem}.onnx"
        print(f"Converting {pt_path.name} -> {onnx_path}")
        try:
            convert_pt_to_onnx(pt_path, onnx_path, opset_version=args.opset)
            print(f"  -> {onnx_path}")
        except (ValueError, OSError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("Done. ONNX files in:", out_dir.resolve())


if __name__ == "__main__":
    main()
