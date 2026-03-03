#!/usr/bin/env bash
# Run WN18RR baselines (text-agnostic and prompt+KD) using the settings from ".env copy wn18rr".
# Outputs relation-level metrics/predictions into ./debug with variant-specific filenames.

set -euo pipefail

ENV_FILE=".env copy wn18rr"

if [ ! -f "$ENV_FILE" ]; then
    echo "[run_wn18rr_baselines] Cannot find env file: $ENV_FILE" >&2
    exit 1
fi

mkdir -p debug

run_variant() {
    local variant="$1"
    shift

    # Load the base env
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport

    # Variant-specific bookkeeping for outputs
    export SAVE_ID="$variant"
    export EVAL_PREDICTIONS_PATH="./debug/${variant}_eval_predictions.jsonl"
    export RELATION_METRICS_PATH="./debug/${variant}_relation_metrics.json"
    export HARD_NEGATIVE_LOG_PATH="./debug/${variant}_hard_negatives.jsonl"
    export TEACHER_DEBUG_LOG_PATH="./debug/${variant}_teacher_disagreements.jsonl"

    # Apply overrides (each should be KEY=VALUE)
    for override in "$@"; do
        export "$override"
    done

    echo "[run_wn18rr_baselines] Launching $variant ..."
    bash run.sh train "${MODEL_NAME:-RelatE}" wn18rr "${GPU_DEVICE:-0}" "${SAVE_ID}"
}

# Text-agnostic baseline: disable prompts and knowledge distillation.
run_variant "wn18rr_baseline_text_agnostic" \
    "USE_REL_PROMPT_EMB=false" \
    "REL_PROMPT_PATH=" \
    "REL_PROMPT_WEIGHT=0" \
    "REL_PROMPT_WARMUP_STEPS=0" \
    "USE_ENTITY_PROMPT_EMB=false" \
    "ENTITY_PROMPT_WEIGHT=0" \
    "ENTITY_PROMPT_WARMUP_STEPS=0" \
    "TEACHER_TYPE=" \
    "TEACHER_CHECKPOINT=" \
    "TEACHER_REPO=" \
    "TEACHER_DEVICE=" \
    "KD_LAMBDA=0" \
    "KD_RELATION_WEIGHTS=" \
    "KD_LOSS=" \
    "KD_WARMUP_STEPS=0" \
    "KD_DECAY_START=0" \
    "KD_DECAY_DURATION=0" \
    "KD_HYPER_WEIGHT=0" \
    "HYPER_KD_WARMUP_STEPS=0"

# Prompt-augmented distillation run (current default settings).
run_variant "wn18rr_prompts_kd" \
    "USE_REL_PROMPT_EMB=true" \
    "USE_ENTITY_PROMPT_EMB=true"
