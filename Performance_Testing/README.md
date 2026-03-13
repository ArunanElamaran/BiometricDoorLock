# Performance Testing (Raspberry Pi AI Hat+)

Latency tests and model conversion for the Raspberry Pi AI Hat+ (Hailo). Model sizes: 20k → 100k → 200k → … (×5, ×2) up to 1B params. Weights are randomized; only latency is measured.

## Setup

```bash
cd Performance_Testing
pip install -r requirements.txt
```

---

## 1. `convert_pt_to_onnx.py`

Converts PyTorch `.pt` files to ONNX. Expects full-model saves (or dict with `"model"` key); infers input size from the first `nn.Linear`.

| Flag | Required | Description |
|------|----------|-------------|
| `--pt-dir DIR` | yes | Folder containing `.pt` files. |
| `--output-dir DIR` | yes | Folder for ONNX output (created if missing). |
| `--model FILE` | no | Convert only this file (e.g. `mlp_20000.pt`). Default: all `.pt` in `--pt-dir`. |
| `--opset N` | no | ONNX opset (default: 18). |

**Examples**

```bash
python convert_pt_to_onnx.py --pt-dir ./checkpoints --output-dir hailo_onnx
python convert_pt_to_onnx.py --pt-dir ./checkpoints --output-dir hailo_onnx --model mlp_20000.pt
```

---

## 2. `convert_onnx_to_hef.py`

Converts ONNX → HEF for Hailo (parse → optimize → compile). Run on x86 with Hailo SDK; not on the Pi.

| Flag | Required | Description |
|------|----------|-------------|
| `--onnx-dir DIR` | no | Folder with ONNX files (default: `hailo_onnx`). |
| `--model FILE` | no | Convert only this ONNX (e.g. `mlp_20000.onnx`). Default: all `*.onnx` in `--onnx-dir`. |
| `--remove` | no | Delete existing HEF/HAR/.log in `hailo_hefs/`, `hailo_work/` and cwd before running. |
| `--post-remove` | no | After conversion, remove the same artifacts (`.hef`, `.har`, `.log`) from those locations. |

**Examples**

```bash
python convert_onnx_to_hef.py
python convert_onnx_to_hef.py --onnx-dir my_onnx --model mlp_20000.onnx
python convert_onnx_to_hef.py --remove
python convert_onnx_to_hef.py --remove --post-remove
```

---

## 3. `latency_test_ai_hat.py`

Benchmarks inference latency: PyTorch (CPU/GPU), ONNX on CPU (onnxruntime), or Hailo NPU (.hef). Can generate synthetic ONNX (and optionally compile) for target sizes.

| Flag | Default | Description |
|------|--------|-------------|
| `--backend` | `torch` | `torch` \| `cpu` \| `hailo`. |
| `--runs` | 100 | Timed runs per model. |
| `--warmup` | 10 | Warmup runs. |
| `--max-params` | 1e9 | Max model size (params). |
| `--device` | auto | `cpu`, `cuda`, or `mps`. |
| `--hailo-hefs-dir` | - | Dir with `.hef` when `backend=hailo`. |
| `--hailo-onnx-dir` | `hailo_onnx` | ONNX dir for generation / CPU comparison. |
| `--generate-hailo-models` | - | Generate synthetic ONNX for target sizes. |
| `--only-generate` | - | Only generate ONNX (and optional compile), then exit. |
| `--compare-hailo-vs-cpu` | - | With `hailo`, also benchmark same ONNX on CPU. |
| `--cpu-max-params` | 2e8 | Cap ONNX CPU benchmark size. |
| `--hailo-compile-template` | - | Shell template to compile ONNX→HEF; placeholders: `{onnx}`, `{hef_dir}`, `{params}`. |

**Examples**

```bash
# PyTorch on CPU, cap 20M params
python latency_test_ai_hat.py --backend torch --device cpu --max-params 2e7

# Generate ONNX only (no benchmarks)
python latency_test_ai_hat.py --generate-hailo-models --only-generate --max-params 2e7

# Benchmark ONNX on CPU (expects ONNX in hailo_onnx)
python latency_test_ai_hat.py --backend cpu --hailo-onnx-dir hailo_onnx

# Benchmark Hailo .hef and compare to CPU ONNX
python latency_test_ai_hat.py --backend hailo --hailo-hefs-dir hailo_hefs --compare-hailo-vs-cpu
```

On Pi: use `--device cpu` and lower `--max-params` (e.g. `2e7`) to avoid OOM.
