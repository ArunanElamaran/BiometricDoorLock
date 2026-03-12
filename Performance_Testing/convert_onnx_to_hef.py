#!/usr/bin/env python3
"""
Batch-convert mlp_*.onnx → .hef for Hailo-8L using the Hailo SDK (Dataflow Compiler).

This script is meant to run on an x86_64 Linux machine with the full Hailo SDK
installed, NOT on the Raspberry Pi. It assumes you already have ONNX files such as:
    hailo_onnx/mlp_20000.onnx, mlp_100000.onnx, ...

The CLI commands below follow the RidgeRun \"Convert ONNX Models to Hailo8L\"
guide and the Hailo SDK tools:

1) Parse:     `hailo parser onnx <model.onnx> -y --hw-arch hailo8l --har-path <model.har>`
2) Optimize:  `hailo optimize <model.har> --hw-arch hailo8l --use-random-calib-set`
3) Compile:   `hailo compiler <model.har> --hw-arch hailo8l`

You can switch to the Python `ClientRunner` API if you prefer, but this script
sticks to the documented CLI flow.
"""

import subprocess
import sys
from pathlib import Path

ONNX_DIR = Path("hailo_onnx")
WORK_DIR = Path("hailo_work")          # holds .har and intermediate files
HEF_DIR = Path("hailo_hefs")           # final .hef output
HW_ARCH = "hailo8l"                    # AI HAT+ / Hailo-8L target


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: command failed with exit code {result.returncode}", file=sys.stderr)


def main() -> None:
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
        stem = onnx_path.stem                     # e.g. "mlp_20000"
        har_path = WORK_DIR / f"{stem}.har"       # parsed / quantized
        optimized_har_path = Path(f"{stem}_optimized.har")   # produced by hailo optimize
        local_hef_path = Path(f"{stem}.hef")                 # produced by hailo compiler
        hef_path = HEF_DIR / f"{stem}.hef"        # final executable

        print("=" * 80)
        print(f"Processing {onnx_path} → {hef_path}")
        print("=" * 80)

        if not har_path.exists():
            PARSE_CMD = [
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
            run(PARSE_CMD)
        else:
            print(f"{har_path} already exists, skipping parse step.")

        OPTIMIZE_CMD = [
            "hailo",
            "optimize",
            str(har_path),
            "--hw-arch",
            HW_ARCH,
            "--use-random-calib-set",
        ]
        run(OPTIMIZE_CMD)

        if not optimized_har_path.is_file():
            print(
                f"WARNING: Expected optimized HAR was not found: {optimized_har_path}",
                file=sys.stderr,
            )
            continue

        COMPILE_CMD = [
            "hailo",
            "compiler",
            str(optimized_har_path),
            "--hw-arch",
            HW_ARCH,
        ]
        print("Running:", " ".join(COMPILE_CMD))
        result = subprocess.run(COMPILE_CMD)
        if result.returncode != 0:
            print(
                f"WARNING: hailo compiler failed with exit code {result.returncode} for {optimized_har_path}",
                file=sys.stderr,
            )
        else:
            if local_hef_path.is_file():
                local_hef_path.replace(hef_path)
                print(f"Saved HEF to {hef_path}")
            else:
                print(
                    f"WARNING: Expected {local_hef_path} was not found after compilation.",
                    file=sys.stderr,
                )

    print("\nDone. HEF files should now be in:", HEF_DIR.resolve())


if __name__ == "__main__":
    main()