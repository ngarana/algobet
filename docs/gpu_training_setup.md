# GPU-Accelerated Model Training Setup

## Overview

AlgoBet's GPU worker runs inside the `intelanalytics/ipex-llm-inference-cpp-xpu:latest` container with `/dev/dri` passed through from the host. The worker is now wired to a real Intel iGPU training path by default:

- `xgboost` is built from source with `PLUGIN_SYCL=ON` and runs on the Intel iGPU via three one-vs-rest binary boosters at `device=sycl:gpu:0`.
- `lightgbm` runs with `device=gpu` on the Intel OpenCL backend.
- startup smoke tests fail the container if either GPU path is unavailable.
- strict GPU mode rejects non-GPU fallbacks for both libraries.

## What Is Actually Accelerated

| Component | Path | Hardware |
|-----------|------|----------|
| XGBoost | `device=sycl:gpu:0`, `tree_method=hist`, one-vs-rest binary boosters | Intel Meteor Lake iGPU |
| LightGBM | `device=gpu` | Intel Meteor Lake iGPU |
| NumPy / scikit-learn | Intel MKL / oneDNN | CPU |

> [!IMPORTANT]
> The standard PyPI `xgboost` wheel still does not expose the Intel iGPU path. The GPU worker fixes this by compiling XGBoost from source with the upstream SYCL plugin, then routes AlgoBet's three-class outcome problem through a SYCL-safe one-vs-rest wrapper because the upstream multiclass SYCL booster currently crashes on Intel iGPU runtimes.

## Files

| File | Purpose |
|------|---------|
| [Dockerfile.gpu-training](file:///home/arch/Coding/algobet/Dockerfile.gpu-training) | Extends the Intel GPU base image, then source-builds SYCL-enabled XGBoost |
| [docker-compose.gpu.yml](file:///home/arch/Coding/algobet/docker-compose.gpu.yml) | Starts the strict Intel iGPU worker on port `8011` |
| [docker/gpu-training/entrypoint.sh](file:///home/arch/Coding/algobet/docker/gpu-training/entrypoint.sh) | Verifies device access, runs LightGBM and XGBoost GPU smoke tests through the real worker path, and starts the worker |
| [model_training.txt](file:///home/arch/Coding/algobet/model_training.txt) | Example API call targeting the GPU worker |

## Detected Host

| Property | Value |
|----------|-------|
| GPU | Intel Meteor Lake-P [Intel Graphics] |
| DRI Device | `/dev/dri/renderD128` |
| Render Group GID | `988` |
| Docker | 29.4.1 |
| Docker Compose | 5.1.3 |

## Quick Start

### 1. Build the GPU worker

```bash
make gpu-build
```

### 2. Verify the iGPU-backed benchmark

```bash
make gpu-benchmark
```

This now runs:

- a NumPy MKL benchmark on CPU
- an XGBoost SYCL benchmark with `device=sycl:gpu:0`
- a LightGBM benchmark with `device=gpu`
- startup smoke tests that fail if the Intel iGPU path is not usable

### 3. Start the GPU worker

```bash
make gpu-up
```

The GPU worker API is exposed on `http://localhost:8011`.

### 4. Train through the GPU worker

```bash
curl -X POST "http://localhost:8011/api/v1/ml/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "tournament_ids": [359],
    "description": "Intel iGPU-accelerated EPL model",
    "activate": true,
    "min_matches": 100,
    "require_odds": true,
    "calibrate_probabilities": true
  }'
```

Or run the one-shot container entrypoint:

```bash
make gpu-train
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALGOBET_ACCELERATION_PROFILE` | `intel_igpu` | Enables the GPU-specific training profile |
| `ALGOBET_REQUIRE_GPU` | `true` | Rejects unsupported CPU fallback paths |
| `ALGOBET_VERIFY_INTEL_GPU` | `true` | Runs LightGBM and XGBoost GPU smoke tests at startup |
| `ALGOBET_DEFAULT_MODEL_TYPE` | `xgboost` | Default API model type on the GPU worker |
| `ALGOBET_LIGHTGBM_DEVICE` | `gpu` | Forces LightGBM onto the OpenCL GPU backend |
| `ALGOBET_XGBOOST_DEVICE` | `sycl:gpu:0` | Forces XGBoost onto the Intel SYCL GPU backend |
| `ALGOBET_XGBOOST_TREE_METHOD` | `hist` | Activates the SYCL histogram updater used by each one-vs-rest XGBoost booster |
| `ALGOBET_GPU_PLATFORM_ID` | `0` | OpenCL platform id for the Intel stack on this host |
| `ALGOBET_GPU_DEVICE_ID` | `1` | OpenCL device id for the Intel iGPU on this host |
| `MKL_NUM_THREADS` | `4` | MKL threading for CPU-bound math |
| `OMP_NUM_THREADS` | `4` | OpenMP threading |
| `MODEL_TYPE` | `xgboost` | Default model type for `make gpu-train` |
| `TUNE` | `False` | Enables Optuna tuning for one-shot training |

## Troubleshooting

### GPU device missing

```bash
ls -la /dev/dri/
stat -c '%g %G %n' /dev/dri/renderD128
```

### Worker starts but training still tries CPU

- Keep `ALGOBET_REQUIRE_GPU=true` so non-GPU overrides fail instead of falling back.
- Re-run `make gpu-benchmark`; it should print `device=sycl:gpu:0` with `grow_quantile_histmaker_sycl` for XGBoost and `device=gpu` for LightGBM.
- For XGBoost, keep `tree_method=hist`; the SYCL plugin does not use the CUDA-style GPU tree methods.
- SYCL XGBoost disables in-training early stopping on the Intel iGPU worker because the upstream multiclass SYCL path crashes during fit-time evaluation.

### OpenCL device id mismatch

If the LightGBM GPU smoke test fails on another machine, inspect devices in the container:

```bash
make gpu-shell
sycl-ls
```

Then update `ALGOBET_GPU_PLATFORM_ID` / `ALGOBET_GPU_DEVICE_ID` in [docker-compose.gpu.yml](file:///home/arch/Coding/algobet/docker-compose.gpu.yml).
