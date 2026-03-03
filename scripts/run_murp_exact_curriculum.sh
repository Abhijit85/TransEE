#!/usr/bin/env bash
set -euo pipefail

# MuRP exact curriculum runner:
# Stage A: easier negatives, high LR
# Stage B: harder negatives, lower LR, resume from Stage A checkpoint
# Stage C (optional): hardest negatives, lowest LR, resume from Stage B checkpoint

ENV_FILE=${ENV_FILE:-.env.murp_exact_wn18rr}
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stage A defaults
CURR_A_ITERS=${CURR_A_ITERS:-250}
CURR_A_NNEG=${CURR_A_NNEG:-50}
CURR_A_LR=${CURR_A_LR:-50}

# Stage B defaults
CURR_B_ITERS=${CURR_B_ITERS:-500}
CURR_B_NNEG=${CURR_B_NNEG:-200}
CURR_B_LR=${CURR_B_LR:-20}

# Stage C defaults (optional)
CURR_STAGE_C_ENABLE=${CURR_STAGE_C_ENABLE:-false}
CURR_C_ITERS=${CURR_C_ITERS:-600}
CURR_C_NNEG=${CURR_C_NNEG:-300}
CURR_C_LR=${CURR_C_LR:-10}

if [ -z "${MURP_CHECKPOINT_DIR:-}" ]; then
  echo "MURP_CHECKPOINT_DIR is empty. Set it in ${ENV_FILE} (or export it) before running curriculum." >&2
  exit 1
fi

if [[ "$MURP_CHECKPOINT_DIR" != /* ]]; then
  MURP_CHECKPOINT_DIR="${REPO_ROOT}/${MURP_CHECKPOINT_DIR#./}"
fi

run_stage() {
  local stage_name="$1"
  local target_iters="$2"
  local nneg="$3"
  local lr="$4"
  local resume_ckpt="$5"

  echo "=== ${stage_name} ==="
  echo "iters=${target_iters} nneg=${nneg} lr=${lr}"
  if [ -n "$resume_ckpt" ]; then
    echo "resume_checkpoint=${resume_ckpt}"
  else
    echo "resume_checkpoint=<none>"
  fi

  export MURP_NUM_ITERATIONS="$target_iters"
  export MURP_NNEG="$nneg"
  export MURP_LR="$lr"
  export MURP_RESUME_CHECKPOINT="$resume_ckpt"

  bash "${REPO_ROOT}/scripts/run_murp_exact.sh"
}

stage_a_ckpt="${MURP_CHECKPOINT_DIR}/iter_${CURR_A_ITERS}.pt"
stage_b_ckpt="${MURP_CHECKPOINT_DIR}/iter_${CURR_B_ITERS}.pt"

# Stage A: fresh start
run_stage "Stage A (easy)" "$CURR_A_ITERS" "$CURR_A_NNEG" "$CURR_A_LR" ""

if [ ! -f "$stage_a_ckpt" ]; then
  echo "Expected Stage A checkpoint missing: $stage_a_ckpt" >&2
  exit 1
fi

# Stage B: resume from Stage A
run_stage "Stage B (harder)" "$CURR_B_ITERS" "$CURR_B_NNEG" "$CURR_B_LR" "$stage_a_ckpt"

if [ "${CURR_STAGE_C_ENABLE,,}" = "true" ] || [ "${CURR_STAGE_C_ENABLE}" = "1" ]; then
  if [ ! -f "$stage_b_ckpt" ]; then
    echo "Expected Stage B checkpoint missing: $stage_b_ckpt" >&2
    exit 1
  fi
  run_stage "Stage C (hardest)" "$CURR_C_ITERS" "$CURR_C_NNEG" "$CURR_C_LR" "$stage_b_ckpt"
fi

echo "Curriculum run complete."
