#!/bin/bash
# ==============================================================================
# GPU Training Worker Entrypoint
# ==============================================================================

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       AlgoBet GPU Training Worker (Intel iGPU)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "▸ Intel oneAPI Root: ${ONEAPI_ROOT:-not set}"
echo "▸ MKL Root:          ${MKLROOT:-not set}"
echo "▸ oneDNN Root:       ${DNNLROOT:-not set}"
echo "▸ TBB Root:          ${TBBROOT:-not set}"
echo "▸ ONEAPI Selector:   ${ONEAPI_DEVICE_SELECTOR:-not set}"
echo "▸ SYCL Cache:        ${SYCL_CACHE_PERSISTENT:-not set}"
echo "▸ Accel Profile:     ${ALGOBET_ACCELERATION_PROFILE:-not set}"
echo "▸ Require GPU:       ${ALGOBET_REQUIRE_GPU:-not set}"
echo "▸ LGBM Device:       ${ALGOBET_LIGHTGBM_DEVICE:-not set}"
echo "▸ XGB Device:        ${ALGOBET_XGBOOST_DEVICE:-not set}"
echo "▸ XGB Tree Method:   ${ALGOBET_XGBOOST_TREE_METHOD:-not set}"
echo "▸ GPU Platform ID:   ${ALGOBET_GPU_PLATFORM_ID:-not set}"
echo "▸ GPU Device ID:     ${ALGOBET_GPU_DEVICE_ID:-not set}"
echo ""

echo "▸ Checking GPU device access..."
if [ -e /dev/dri/renderD128 ]; then
    echo "  ✓ /dev/dri/renderD128 is accessible"
    ls -la /dev/dri/ 2>/dev/null || true
else
    echo "  ✗ WARNING: /dev/dri/renderD128 not found!"
    echo "    GPU acceleration will NOT be available."
    echo "    Ensure --device /dev/dri is passed to docker run."
fi

if command -v sycl-ls &>/dev/null; then
    echo ""
    echo "▸ SYCL Devices:"
    sycl-ls 2>/dev/null || echo "  (sycl-ls not available)"
fi

echo ""
echo "▸ Python: $(python3 --version 2>&1)"
echo "▸ Verifying ML dependencies..."
python3 - <<'PY'
checks = []
try:
    import xgboost

    checks.append(f"  ✓ XGBoost {xgboost.__version__}")
except ImportError:
    checks.append("  ✗ XGBoost NOT installed")
try:
    import lightgbm

    checks.append(f"  ✓ LightGBM {lightgbm.__version__}")
except ImportError:
    checks.append("  ✗ LightGBM NOT installed")
try:
    import sklearn

    checks.append(f"  ✓ scikit-learn {sklearn.__version__}")
except ImportError:
    checks.append("  ✗ scikit-learn NOT installed")
try:
    import numpy

    checks.append(f"  ✓ NumPy {numpy.__version__}")
except ImportError:
    checks.append("  ✗ NumPy NOT installed")
try:
    import pandas

    checks.append(f"  ✓ Pandas {pandas.__version__}")
except ImportError:
    checks.append("  ✗ Pandas NOT installed")

print("\n".join(checks))
PY
echo ""

export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "▸ MKL Threads: ${MKL_NUM_THREADS}"
echo "▸ OMP Threads: ${OMP_NUM_THREADS}"
echo ""

VERIFY_INTEL_GPU="$(printf '%s' "${ALGOBET_VERIFY_INTEL_GPU:-0}" | tr '[:upper:]' '[:lower:]')"
if [ "$VERIFY_INTEL_GPU" = "1" ] || [ "$VERIFY_INTEL_GPU" = "true" ] || [ "$VERIFY_INTEL_GPU" = "yes" ] || [ "$VERIFY_INTEL_GPU" = "on" ]; then
    echo "▸ Verifying LightGBM Intel iGPU backend..."
    python3 - <<'PY'
import os

import lightgbm as lgb
from sklearn.datasets import make_classification

device = os.getenv("ALGOBET_LIGHTGBM_DEVICE", "gpu")
require_gpu = os.getenv("ALGOBET_REQUIRE_GPU", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if require_gpu and device != "gpu":
    raise SystemExit(
        "ALGOBET_REQUIRE_GPU=true but ALGOBET_LIGHTGBM_DEVICE is not set to gpu."
    )

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "device": device,
}

platform_id = os.getenv("ALGOBET_GPU_PLATFORM_ID")
device_id = os.getenv("ALGOBET_GPU_DEVICE_ID")
if platform_id:
    params["gpu_platform_id"] = int(platform_id)
if device_id:
    params["gpu_device_id"] = int(device_id)

X, y = make_classification(
    n_samples=1024,
    n_features=32,
    n_classes=3,
    n_informative=16,
    n_redundant=0,
    random_state=42,
)
dataset = lgb.Dataset(X, label=y)
lgb.train(params, dataset, num_boost_round=5)
print(
    "  ✓ LightGBM GPU smoke test passed "
    f"(device={params['device']}, "
    f"gpu_platform_id={params.get('gpu_platform_id', 'auto')}, "
    f"gpu_device_id={params.get('gpu_device_id', 'auto')})"
)
PY
    echo ""

    echo "▸ Verifying XGBoost Intel iGPU backend..."
    python3 - <<'PY'
import json
import os

from sklearn.datasets import make_classification

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.classifiers import ModelConfig, XGBoostPredictor

device = os.getenv("ALGOBET_XGBOOST_DEVICE", "sycl:gpu:0")
tree_method = os.getenv("ALGOBET_XGBOOST_TREE_METHOD", "hist")
require_gpu = os.getenv("ALGOBET_REQUIRE_GPU", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if require_gpu and not device.startswith("sycl:gpu"):
    raise SystemExit(
        "ALGOBET_REQUIRE_GPU=true but ALGOBET_XGBOOST_DEVICE is not a SYCL GPU."
    )
if require_gpu and tree_method != "hist":
    raise SystemExit(
        "ALGOBET_REQUIRE_GPU=true but ALGOBET_XGBOOST_TREE_METHOD is not hist."
    )

X, y = make_classification(
    n_samples=1024,
    n_features=32,
    n_classes=3,
    n_informative=16,
    n_redundant=0,
    random_state=42,
)
params = resolve_training_hyperparameters(
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 6,
        "verbosity": 2,
    },
)
predictor = XGBoostPredictor(
    ModelConfig(
        model_type="xgboost",
        hyperparameters=params,
        random_seed=42,
    )
)
predictor.fit(X, y)
config = json.loads(predictor._model[0].save_config())
resolved_device = config["learner"]["generic_param"]["device"]
updater = config["learner"]["gradient_booster"]["gbtree_train_param"]["updater_seq"]
if require_gpu and not resolved_device.startswith("sycl:gpu"):
    raise SystemExit(
        "XGBoost resolved to a non-GPU device despite strict GPU mode."
    )
if "grow_quantile_histmaker_sycl" not in updater:
    raise SystemExit(
        "XGBoost did not activate the SYCL updater on the Intel iGPU worker."
    )
print(
    "  ✓ XGBoost SYCL smoke test passed "
    f"(device={resolved_device}, updater={updater}, "
    f"binary_boosters={len(predictor._model)})"
)
PY
    echo ""
fi

MODE="${1:-worker}"
shift || true

case "$MODE" in
    worker)
        echo "▸ Starting API server with GPU training support..."
        exec uvicorn algobet.api.main:app \
            --host 0.0.0.0 \
            --port "${API_PORT:-8010}" \
            --reload \
            "$@"
        ;;

    train)
        echo "▸ Running one-shot training job..."
        exec python3 -c "
from algobet.predictions.training.pipeline import train_model
from algobet.infrastructure.database import session_scope
import json

with session_scope() as session:
    result = train_model(
        session=session,
        model_type='${MODEL_TYPE:-xgboost}',
        tune=${TUNE:-False},
        description='${DESCRIPTION:-Intel iGPU training run}',
    )
    print(json.dumps({
        'model_version': result.model_version,
        'model_type': result.model_type,
        'test_accuracy': result.test_metrics.get('accuracy', 0),
        'training_duration_seconds': result.training_duration_seconds,
    }, indent=2))
" "$@"
        ;;

    benchmark)
        echo "▸ Running GPU benchmark..."
        exec python3 - "$@" <<PY
import os
import time
import json

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

from algobet.predictions.training.acceleration import resolve_training_hyperparameters
from algobet.predictions.training.classifiers import ModelConfig, XGBoostPredictor

print("--- NumPy MKL Benchmark ---")
sizes = [1000, 2000, 4000]
for n in sizes:
    A = np.random.randn(n, n).astype(np.float64)
    B = np.random.randn(n, n).astype(np.float64)
    start = time.time()
    C = A @ B
    elapsed = time.time() - start
    gflops = (2 * n**3) / elapsed / 1e9
    print(f"  Matrix {n}x{n} multiply: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")

print()
print("--- LightGBM Intel iGPU Benchmark ---")
X, y = make_classification(
    n_samples=50000,
    n_features=50,
    n_classes=3,
    n_informative=30,
    random_state=42,
)
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "device": os.getenv("ALGOBET_LIGHTGBM_DEVICE", "gpu"),
    "num_threads": ${OMP_NUM_THREADS:-4},
    "verbose": -1,
}
platform_id = os.getenv("ALGOBET_GPU_PLATFORM_ID")
device_id = os.getenv("ALGOBET_GPU_DEVICE_ID")
if platform_id:
    params["gpu_platform_id"] = int(platform_id)
if device_id:
    params["gpu_device_id"] = int(device_id)
print(
    f"  device={params['device']} "
    f"gpu_platform_id={params.get('gpu_platform_id', 'auto')} "
    f"gpu_device_id={params.get('gpu_device_id', 'auto')}"
)
dataset = lgb.Dataset(X, label=y)
start = time.time()
lgb.train(params, dataset, num_boost_round=200)
elapsed = time.time() - start
print(f"  LightGBM 200 rounds (50k samples, 50 features): {elapsed:.2f}s")
print()

print("--- XGBoost Intel iGPU Benchmark ---")
xgb_params = resolve_training_hyperparameters(
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
    },
)
xgb_predictor = XGBoostPredictor(
    ModelConfig(
        model_type="xgboost",
        hyperparameters=xgb_params,
        random_seed=42,
    )
)
start = time.time()
xgb_predictor.fit(X, y)
elapsed = time.time() - start
config = json.loads(xgb_predictor._model[0].save_config())
print(
    f"  device={config['learner']['generic_param']['device']} "
    f"updater={config['learner']['gradient_booster']['gbtree_train_param']['updater_seq']} "
    f"binary_boosters={len(xgb_predictor._model)}"
)
print(f"  XGBoost 200 rounds (50k samples, 50 features): {elapsed:.2f}s")
print()

print("Done.")
PY
        ;;

    shell)
        echo "▸ Dropping into interactive shell..."
        exec /bin/bash "$@"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: entrypoint.sh {worker|train|benchmark|shell}"
        exit 1
        ;;
esac
