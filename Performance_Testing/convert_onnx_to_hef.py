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
        hef_path = HEF_DIR / f"{stem}.hef"        # final executable

        print("=" * 80)
        print(f"Processing {onnx_path} → {hef_path}")
        print("=" * 80)

        # ------------------------------------------------------------------
        # 1) PARSE: ONNX → HAR (unquantized)
        # Per RidgeRun / Hailo docs:
        #   hailo parser onnx models/yolov5m_vehicles.onnx -y --hw-arch hailo8l --har-path models/yolov5m_vehicles.har
        # We adapt it to our paths and filenames.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2) OPTIMIZE / QUANTIZE: HAR → quantized HAR
        # The article shows (with real calib + model script):
        #   hailo optimize yolov5m_vehicles.har --hw-arch hailo8l --calib-set-path ./calib_set.npy --model-script default_model_script.all
        # For our synthetic MLPs, we use the documented `--use-random-calib-set`
        # option instead of a real calibration dataset.
        # ------------------------------------------------------------------
        OPTIMIZE_CMD = [
            "hailo",
            "optimize",
            str(har_path),
            "--hw-arch",
            HW_ARCH,
            "--use-random-calib-set",
        ]
        run(OPTIMIZE_CMD)

        # ------------------------------------------------------------------
        # 3) COMPILE: quantized HAR → HEF
        # The article shows:
        #   hailo compiler yolov5m_vehicles.har --hw-arch hailo8l
        # which produces `<name>.hef` in the working directory. Here we let
        # the compiler create the default `<stem>.hef` next to `har_path`,
        # and then move/rename it into HEF_DIR.
        # ------------------------------------------------------------------
        # Run compiler in the HAR directory so the output `.hef` lands there.
        compile_cwd = har_path.parent
        COMPILE_CMD = [
            "hailo",
            "compiler",
            str(har_path),
            "--hw-arch",
            HW_ARCH,
        ]
        print(f"Running: {' '.join(COMPILE_CMD)} (cwd={compile_cwd})")
        result = subprocess.run(COMPILE_CMD, cwd=str(compile_cwd))
        if result.returncode != 0:
            print(
                f"WARNING: hailo compiler failed with exit code {result.returncode} for {har_path}",
                file=sys.stderr,
            )
        else:
            # The compiler should have created `<stem>.hef` in `compile_cwd`.
            default_hef = compile_cwd / f"{stem}.hef"
            if default_hef.is_file():
                # Move/rename into our HEF_DIR
                default_hef.replace(hef_path)
                print(f"Saved HEF to {hef_path}")
            else:
                print(
                    f"WARNING: Expected {default_hef} was not found after compilation.",
                    file=sys.stderr,
                )

    print("\nDone. HEF files should now be in:", HEF_DIR.resolve())


if __name__ == "__main__":
    main()