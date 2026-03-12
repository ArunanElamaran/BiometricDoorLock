#!/usr/bin/env python3
"""
Batch-convert mlp_*.onnx → .hef for Hailo-8L using the Hailo SDK (Dataflow Compiler).

This script is meant to run on an x86_64 Linux machine with the full Hailo SDK
installed, NOT on the Raspberry Pi. It assumes you already have ONNX files such as:
    hailo_onnx/mlp_20000.onnx, mlp_100000.onnx, ...

You MUST adapt the CLI commands (PARSE_CMD, OPTIMIZE_CMD, COMPILE_CMD) to match
the exact DFC tools/options for your SDK, following Hailo’s docs / the RidgeRun article.
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
        # Replace this with the exact DFC CLI or Python wrapper from your SDK.
        # Example CLI shape (NOT exact – check your docs):
        #
        #   hailoc translate-onnx \
        #       --model {onnx_path} \
        #       --output-har {har_path} \
        #       --hw-arch {HW_ARCH}
        # ------------------------------------------------------------------
        if not har_path.exists():
            PARSE_CMD = [
                "hailoc", "translate-onnx",
                "--model", str(onnx_path),
                "--output-har", str(har_path),
                "--hw-arch", HW_ARCH,
            ]
            run(PARSE_CMD)
        else:
            print(f"{har_path} already exists, skipping parse step.")

        # ------------------------------------------------------------------
        # 2) OPTIMIZE / QUANTIZE: HAR → quantized HAR
        # Some SDKs do this in-place on the HAR; others write a new HAR.
        # You can either:
        #   - Use a real calibration set, OR
        #   - Use the SDK's "random calibration" option for these synthetic MLPs,
        #     as mentioned in the RidgeRun article.
        #
        # Example CLI shape (NOT exact – check your docs):
        #
        #   hailoc optimize \
        #       --har {har_path} \
        #       --hw-arch {HW_ARCH} \
        #       --use-random-calib-set
        # ------------------------------------------------------------------
        OPTIMIZE_CMD = [
            "hailoc", "optimize",
            "--har", str(har_path),
            "--hw-arch", HW_ARCH,
            "--use-random-calib-set",
        ]
        run(OPTIMIZE_CMD)

        # ------------------------------------------------------------------
        # 3) COMPILE: quantized HAR → HEF
        #
        # Example CLI shape (NOT exact – check your docs):
        #
        #   hailoc compile \
        #       --har {har_path} \
        #       --hw-arch {HW_ARCH} \
        #       --output {hef_path}
        # ------------------------------------------------------------------
        COMPILE_CMD = [
            "hailoc", "compile",
            "--har", str(har_path),
            "--hw-arch", HW_ARCH,
            "--output", str(hef_path),
        ]
        run(COMPILE_CMD)

    print("\nDone. HEF files should now be in:", HEF_DIR.resolve())


if __name__ == "__main__":
    main()