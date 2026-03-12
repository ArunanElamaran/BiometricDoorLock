#!/usr/bin/env python3
"""
Batch-convert mlp_*.onnx → .hef for Hailo-8 / Hailo-8L using the Hailo SDK.

Examples:
    python convert_onnx_to_hef.py
    python convert_onnx_to_hef.py --remove
"""

import argparse
import subprocess
import sys
from pathlib import Path

ONNX_DIR = Path("hailo_onnx")
WORK_DIR = Path("hailo_work")      # holds parsed .har files
HEF_DIR = Path("hailo_hefs")       # final .hef output
HW_ARCH = "hailo8"                 # use "hailo8" for Hailo-8, "hailo8l" for Hailo-8L


def run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: command failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


def remove_if_exists(path: Path) -> None:
    if path.exists():
        print(f"Removing old file: {path}")
        path.unlink()


def cleanup_previous_outputs(stem: str) -> None:
    """
    Remove files from previous runs for this model.
    """
    har_path = WORK_DIR / f"{stem}.har"
    optimized_har_path = Path(f"{stem}_optimized.har")
    local_hef_path = Path(f"{stem}.hef")
    final_hef_path = HEF_DIR / f"{stem}.hef"

    remove_if_exists(har_path)
    remove_if_exists(optimized_har_path)
    remove_if_exists(local_hef_path)
    remove_if_exists(final_hef_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove previously generated HAR/optimized HAR/HEF files before processing each model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not ONNX_DIR.is_dir():
        print(f"ONNX directory not found: {ONNX_DIR}", file=sys.stderr)
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    HEF_DIR.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(ONNX_DIR.glob("mlp_*.onnx"))
    if not onnx_files:
        print(f"No mlp_*.onnx files found in {ONNX_DIR}", file=sys.stderr)
        sys.exit(1)

    for onnx_path in onnx_files:
        stem = onnx_path.stem
        har_path = WORK_DIR / f"{stem}.har"
        optimized_har_path = Path(f"{stem}_optimized.har")
        local_hef_path = Path(f"{stem}.hef")
        hef_path = HEF_DIR / f"{stem}.hef"

        print("=" * 80)
        print(f"Processing {onnx_path} → {hef_path}")
        print("=" * 80)

        if args.remove:
            cleanup_previous_outputs(stem)

        # 1) Parse ONNX -> HAR
        if not har_path.exists():
            parse_cmd = [
                "hailo",
                "parser",
                "onnx",
                str(onnx_path),
                "-y",
                "--hw-arch",
                HW_ARCH,
                "--har-path",
                str(har_path),
            ]
            if run(parse_cmd) != 0:
                continue
        else:
            print(f"{har_path} already exists, skipping parse step.")

        # 2) Optimize HAR -> optimized HAR
        optimize_cmd = [
            "hailo",
            "optimize",
            str(har_path),
            "--hw-arch",
            HW_ARCH,
            "--use-random-calib-set",
        ]
        if run(optimize_cmd) != 0:
            continue

        if not optimized_har_path.is_file():
            print(f"WARNING: Expected optimized HAR was not found: {optimized_har_path}", file=sys.stderr)
            continue

        # 3) Compile optimized HAR -> HEF
        compile_cmd = [
            "hailo",
            "compiler",
            str(optimized_har_path),
            "--hw-arch",
            HW_ARCH,
        ]
        if run(compile_cmd) != 0:
            continue

        if local_hef_path.is_file():
            local_hef_path.replace(hef_path)
            print(f"Saved HEF to {hef_path}")
        else:
            print(f"WARNING: Expected {local_hef_path} was not found after compilation.", file=sys.stderr)

    print("\nDone. HEF files should now be in:", HEF_DIR.resolve())


if __name__ == "__main__":
    main()