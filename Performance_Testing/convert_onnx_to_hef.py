#!/usr/bin/env python3
"""
Batch-convert .onnx → .hef for Hailo using the Hailo SDK (Dataflow Compiler).

Runs on every .onnx file in the ONNX directory (default: hailo_onnx), or on a
single file if --model is given. Meant for x86_64 Linux with the full Hailo SDK
installed, NOT on the Raspberry Pi.

The CLI commands below follow the RidgeRun \"Convert ONNX Models to Hailo8L\"
guide and the Hailo SDK tools:

1) Parse:     `hailo parser onnx <model.onnx> -y --hw-arch hailo8l --har-path <model.har>`
2) Optimize:  `hailo optimize <model.har> --hw-arch hailo8l --use-random-calib-set`
3) Compile:   `hailo compiler <model.har> --hw-arch hailo8l`

You can switch to the Python `ClientRunner` API if you prefer, but this script
sticks to the documented CLI flow.
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("hailo_work")          # holds .har and intermediate files
HEF_DIR = Path("hailo_hefs")           # final .hef output
HW_ARCH = "hailo8"                     # target device
DEFAULT_ONNX_DIR = Path("hailo_onnx")  # default folder for ONNX files


def remove_artifacts(*, preserve_hef_dir: bool) -> None:
    """
    Remove HEF/HAR artifacts from hailo_work and cwd.

    If preserve_hef_dir is False, also delete any existing HEFs in HEF_DIR.
    """
    if not preserve_hef_dir:
        for path in HEF_DIR.glob("*.hef"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    for path in WORK_DIR.glob("*.har"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for path in WORK_DIR.glob("*.log"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for path in WORK_DIR.glob("*.hef"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cwd = Path.cwd()
    for ext in ("*.log", "*.har", "*.hef"):
        for path in cwd.glob(ext):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: command failed with exit code {result.returncode}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert .onnx models to HEF for Hailo devices."
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help=(
            "Delete existing artifacts before compiling. "
            "This removes existing HEFs in the output folder plus build artifacts in the work folder and "
            "current directory."
        ),
    )
    parser.add_argument(
        "--post-remove",
        action="store_true",
        help="If --remove is also set, remove the same HEF/HAR artifacts again after conversion.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=str(DEFAULT_ONNX_DIR),
        help=f"Folder containing ONNX files (default: {DEFAULT_ONNX_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "If set, convert only this exact ONNX filename (e.g. mlp_20000.onnx). "
            "Lookup is in --onnx-dir (or default folder)."
        ),
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)

    if args.remove:
        remove_artifacts(preserve_hef_dir=False)
        print("Removed existing HEF and HAR artifacts from work/output folders and current directory.")

    if not onnx_dir.is_dir():
        print(f"ONNX directory not found: {onnx_dir}", file=sys.stderr)
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    HEF_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        single = onnx_dir / args.model
        if not single.is_file():
            print(f"Model file not found: {single}", file=sys.stderr)
            sys.exit(1)
        onnx_files = [single]
    else:
        onnx_files = sorted(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print(f"No .onnx files found in {onnx_dir}", file=sys.stderr)
            sys.exit(1)

    for onnx_path in onnx_files:
        stem = onnx_path.stem                             # e.g. "mlp_20000"
        har_path = WORK_DIR / f"{stem}.har"               # parsed / quantized
        optimized_har_path = WORK_DIR / f"{stem}_optimized.har"  # produced by hailo optimize
        local_hef_path = WORK_DIR / f"{stem}.hef"         # produced by hailo compiler
        hef_path = HEF_DIR / f"{stem}.hef"                # final executable

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
        # Run compiler in WORK_DIR so any generated *_compiled.har / .hef
        # are written there instead of cluttering the project root.
        result = subprocess.run(COMPILE_CMD, cwd=str(WORK_DIR))
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

    if args.post_remove:
        remove_artifacts(preserve_hef_dir=True)
        print(
            "Post-remove: cleaned build artifacts from work folder and current directory "
            "(preserved HEFs in output folder)."
        )


if __name__ == "__main__":
    main()