#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-.env.murp_exact_wn18rr}
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
fi

MURP_EXACT_DIR=${MURP_EXACT_DIR:-./murp_exact}
MURP_DATASET=${MURP_DATASET:-WN18RR}
MURP_MODEL=${MURP_MODEL:-poincare}
MURP_NUM_ITERATIONS=${MURP_NUM_ITERATIONS:-500}
MURP_NNEG=${MURP_NNEG:-50}
MURP_BATCH_SIZE=${MURP_BATCH_SIZE:-128}
MURP_LR=${MURP_LR:-50}
MURP_DIM=${MURP_DIM:-40}
MURP_CUDA=${MURP_CUDA:-true}
MURP_GPU=${MURP_GPU:-0}
MURP_LOG_FILE=${MURP_LOG_FILE:-}
MURP_EVAL_PREDICTIONS_PATH=${MURP_EVAL_PREDICTIONS_PATH:-}
MURP_HARD_NEGATIVES_PATH=${MURP_HARD_NEGATIVES_PATH:-}
MURP_RELATION_METRICS_PATH=${MURP_RELATION_METRICS_PATH:-}
MURP_EVAL_TOPK=${MURP_EVAL_TOPK:-5}
MURP_CONCEPT_MAP_PATH=${MURP_CONCEPT_MAP_PATH:-}
MURP_CONCEPT_DEPTH_PATH=${MURP_CONCEPT_DEPTH_PATH:-}
MURP_CONCEPT_WEIGHT=${MURP_CONCEPT_WEIGHT:-0}
MURP_CONCEPT_DEPTH_WEIGHT=${MURP_CONCEPT_DEPTH_WEIGHT:-0}
MURP_CONCEPT_WARMUP_ITERATIONS=${MURP_CONCEPT_WARMUP_ITERATIONS:-0}
MURP_CHECKPOINT_DIR=${MURP_CHECKPOINT_DIR:-}
MURP_CHECKPOINT_EVERY=${MURP_CHECKPOINT_EVERY:-0}
MURP_RESUME_CHECKPOINT=${MURP_RESUME_CHECKPOINT:-}
MURP_NNEG_SCHEDULE=${MURP_NNEG_SCHEDULE:-}
MURP_LR_SCHEDULE=${MURP_LR_SCHEDULE:-}
MURP_RELATION_CALIBRATION=${MURP_RELATION_CALIBRATION:-false}
MURP_REL_SCALE_MIN=${MURP_REL_SCALE_MIN:-}
MURP_REL_SCALE_MAX=${MURP_REL_SCALE_MAX:-}
MURP_REL_SCALE_INIT=${MURP_REL_SCALE_INIT:-}
MURP_REL_SCALE_REG=${MURP_REL_SCALE_REG:-}

if [ ! -d "$MURP_EXACT_DIR" ]; then
  echo "MuRP exact directory not found: $MURP_EXACT_DIR" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="python3"
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
  PYBIN="${REPO_ROOT}/.venv/bin/python3"
fi
mkdir -p "${REPO_ROOT}/debug"
if [ -z "${MURP_LOG_FILE}" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  MURP_LOG_FILE="${REPO_ROOT}/debug/murp_exact_${MURP_DATASET,,}_${MURP_MODEL}_${ts}.log"
fi
if [ -z "${MURP_EVAL_PREDICTIONS_PATH}" ]; then
  MURP_EVAL_PREDICTIONS_PATH="${REPO_ROOT}/debug/eval_predictions.jsonl"
fi
if [ -z "${MURP_HARD_NEGATIVES_PATH}" ]; then
  MURP_HARD_NEGATIVES_PATH="${REPO_ROOT}/debug/hard_negatives.jsonl"
fi
if [ -z "${MURP_RELATION_METRICS_PATH}" ]; then
  MURP_RELATION_METRICS_PATH="${REPO_ROOT}/debug/relation_metrics.json"
fi
if [ -z "${MURP_CHECKPOINT_DIR}" ]; then
  MURP_CHECKPOINT_DIR="${REPO_ROOT}/model_tuned/murp_exact_${MURP_DATASET,,}_${MURP_MODEL}/checkpoint"
fi

# Normalize relative output paths against repo root, then ensure parent dirs exist.
if [[ "$MURP_LOG_FILE" != /* ]]; then
  MURP_LOG_FILE="${REPO_ROOT}/${MURP_LOG_FILE#./}"
fi
if [[ "$MURP_EVAL_PREDICTIONS_PATH" != /* ]]; then
  MURP_EVAL_PREDICTIONS_PATH="${REPO_ROOT}/${MURP_EVAL_PREDICTIONS_PATH#./}"
fi
if [[ "$MURP_HARD_NEGATIVES_PATH" != /* ]]; then
  MURP_HARD_NEGATIVES_PATH="${REPO_ROOT}/${MURP_HARD_NEGATIVES_PATH#./}"
fi
if [[ "$MURP_RELATION_METRICS_PATH" != /* ]]; then
  MURP_RELATION_METRICS_PATH="${REPO_ROOT}/${MURP_RELATION_METRICS_PATH#./}"
fi
if [ -n "$MURP_CONCEPT_MAP_PATH" ] && [[ "$MURP_CONCEPT_MAP_PATH" != /* ]]; then
  MURP_CONCEPT_MAP_PATH="${REPO_ROOT}/${MURP_CONCEPT_MAP_PATH#./}"
fi
if [ -n "$MURP_CONCEPT_DEPTH_PATH" ] && [[ "$MURP_CONCEPT_DEPTH_PATH" != /* ]]; then
  MURP_CONCEPT_DEPTH_PATH="${REPO_ROOT}/${MURP_CONCEPT_DEPTH_PATH#./}"
fi
if [ -n "$MURP_CHECKPOINT_DIR" ] && [[ "$MURP_CHECKPOINT_DIR" != /* ]]; then
  MURP_CHECKPOINT_DIR="${REPO_ROOT}/${MURP_CHECKPOINT_DIR#./}"
fi
if [ -n "$MURP_RESUME_CHECKPOINT" ] && [[ "$MURP_RESUME_CHECKPOINT" != /* ]]; then
  MURP_RESUME_CHECKPOINT="${REPO_ROOT}/${MURP_RESUME_CHECKPOINT#./}"
fi
mkdir -p "$(dirname "$MURP_LOG_FILE")"
mkdir -p "$(dirname "$MURP_EVAL_PREDICTIONS_PATH")"
mkdir -p "$(dirname "$MURP_HARD_NEGATIVES_PATH")"
mkdir -p "$(dirname "$MURP_RELATION_METRICS_PATH")"
mkdir -p "$MURP_CHECKPOINT_DIR"
CHECKPOINT_LOG_FILE="${MURP_CHECKPOINT_DIR}/train.log"

EXTRA_ARGS=(
  --concept_weight "$MURP_CONCEPT_WEIGHT"
  --concept_depth_weight "$MURP_CONCEPT_DEPTH_WEIGHT"
  --concept_warmup_iterations "$MURP_CONCEPT_WARMUP_ITERATIONS"
)
if [ -n "$MURP_CONCEPT_MAP_PATH" ]; then
  EXTRA_ARGS+=(--concept_map_path "$MURP_CONCEPT_MAP_PATH")
fi
if [ -n "$MURP_CONCEPT_DEPTH_PATH" ]; then
  EXTRA_ARGS+=(--concept_depth_path "$MURP_CONCEPT_DEPTH_PATH")
fi
EXTRA_ARGS+=(--checkpoint_dir "$MURP_CHECKPOINT_DIR")
EXTRA_ARGS+=(--checkpoint_every "$MURP_CHECKPOINT_EVERY")
if [ -n "$MURP_RESUME_CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume_checkpoint "$MURP_RESUME_CHECKPOINT")
fi
if [ -n "$MURP_NNEG_SCHEDULE" ]; then
  EXTRA_ARGS+=(--nneg_schedule "$MURP_NNEG_SCHEDULE")
fi
if [ -n "$MURP_LR_SCHEDULE" ]; then
  EXTRA_ARGS+=(--lr_schedule "$MURP_LR_SCHEDULE")
fi
EXTRA_ARGS+=(--relation_calibration "$MURP_RELATION_CALIBRATION")
if [ -n "$MURP_REL_SCALE_MIN" ]; then
  EXTRA_ARGS+=(--rel_scale_min "$MURP_REL_SCALE_MIN")
fi
if [ -n "$MURP_REL_SCALE_MAX" ]; then
  EXTRA_ARGS+=(--rel_scale_max "$MURP_REL_SCALE_MAX")
fi
if [ -n "$MURP_REL_SCALE_INIT" ]; then
  EXTRA_ARGS+=(--rel_scale_init "$MURP_REL_SCALE_INIT")
fi
if [ -n "$MURP_REL_SCALE_REG" ]; then
  EXTRA_ARGS+=(--rel_scale_reg "$MURP_REL_SCALE_REG")
fi

pushd "$MURP_EXACT_DIR" >/dev/null
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$MURP_GPU" "$PYBIN" -u main.py \
  --model "$MURP_MODEL" \
  --dataset "$MURP_DATASET" \
  --num_iterations "$MURP_NUM_ITERATIONS" \
  --nneg "$MURP_NNEG" \
  --batch_size "$MURP_BATCH_SIZE" \
  --lr "$MURP_LR" \
  --dim "$MURP_DIM" \
  --cuda "$MURP_CUDA" \
  --eval_predictions_path "$MURP_EVAL_PREDICTIONS_PATH" \
  --hard_negatives_path "$MURP_HARD_NEGATIVES_PATH" \
  --relation_metrics_path "$MURP_RELATION_METRICS_PATH" \
  --eval_topk "$MURP_EVAL_TOPK" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$MURP_LOG_FILE" "$CHECKPOINT_LOG_FILE"
popd >/dev/null
echo "MuRP exact log written to: $MURP_LOG_FILE"
echo "MuRP checkpoint log written to: $CHECKPOINT_LOG_FILE"
