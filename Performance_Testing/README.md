# Performance Testing (Raspberry Pi AI Hat+)

Latency tests for models of different sizes, e.g. when evaluating the Raspberry Pi AI Hat+.

## Model sizes

Models are created with the following parameter counts (then ×5, ×2, ×5, ×2, … up to 1B):

- 20,000 → 100,000 → 200,000 → 1,000,000 → 2,000,000 → 10,000,000 → 20,000,000 → 100,000,000 → 200,000,000 → 1,000,000,000

Weights are randomized; only inference latency is measured, not accuracy.

## Setup

```bash
cd Performance_Testing
pip install -r requirements.txt
```

## Run

```bash
# Default: CPU, up to 1B params, 100 timed runs per model
python latency_test_ai_hat.py

# Fewer runs, cap at 20M params (e.g. for Pi with limited RAM)
python latency_test_ai_hat.py --runs 50 --max-params 2e7

# Save results to CSV
python latency_test_ai_hat.py --output results.csv

# Use GPU if available (CUDA/MPS); force CPU on Pi
python latency_test_ai_hat.py --device cpu
```

## Options

| Option         | Default | Description                                      |
|----------------|--------|--------------------------------------------------|
| `--runs`       | 100    | Number of timed inference runs per model.        |
| `--warmup`     | 10     | Warmup runs before timing.                       |
| `--max-params` | 1e9    | Maximum model size; reduce on low-memory devices.|
| `--output`     | -      | Path to save CSV (target_params, mean_ms, etc.). |
| `--device`     | auto   | `cpu`, `cuda`, or `mps`.                         |

On Raspberry Pi, use `--device cpu` and a lower `--max-params` (e.g. `2e7` or `1e8`) to avoid OOM.
