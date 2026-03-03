#!/bin/bash

# Load env file (default .env) if present
is_truthy() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

ENV_FILE=${ENV_FILE:-.env}
if [ -f "$ENV_FILE" ]; then
    # Preserve externally provided COMPILE_* overrides
    PRE_COMPILE_OUTPUT_DIR=${COMPILE_OUTPUT_DIR-}
    PRE_COMPILE_PHASE_DIM=${COMPILE_PHASE_DIM-}
    PRE_COMPILE_EVAL_SPLIT=${COMPILE_EVAL_SPLIT-}
    PRE_COMPILE_EVAL_MODE=${COMPILE_EVAL_MODE-}
    PRE_COMPILE_ANYBURL_RULES=${COMPILE_ANYBURL_RULES-}
    PRE_COMPILE_CANDIDATE_MAX=${COMPILE_CANDIDATE_MAX-}
    PRE_COMPILE_ALPHA=${COMPILE_ALPHA-}
    PRE_COMPILE_FALLBACK_TOPK=${COMPILE_FALLBACK_TOPK-}
    PRE_COMPILE_RULE_TOPK=${COMPILE_RULE_TOPK-}

    set -o allexport
    # shellcheck disable=SC1091
    source "$ENV_FILE"
    set +o allexport

    # Restore any overrides passed in the environment
    if [ -n "${PRE_COMPILE_OUTPUT_DIR+x}" ] && [ -n "${PRE_COMPILE_OUTPUT_DIR}" ]; then COMPILE_OUTPUT_DIR=$PRE_COMPILE_OUTPUT_DIR; fi
    if [ -n "${PRE_COMPILE_PHASE_DIM+x}" ] && [ -n "${PRE_COMPILE_PHASE_DIM}" ]; then COMPILE_PHASE_DIM=$PRE_COMPILE_PHASE_DIM; fi
    if [ -n "${PRE_COMPILE_EVAL_SPLIT+x}" ] && [ -n "${PRE_COMPILE_EVAL_SPLIT}" ]; then COMPILE_EVAL_SPLIT=$PRE_COMPILE_EVAL_SPLIT; fi
    if [ -n "${PRE_COMPILE_EVAL_MODE+x}" ] && [ -n "${PRE_COMPILE_EVAL_MODE}" ]; then COMPILE_EVAL_MODE=$PRE_COMPILE_EVAL_MODE; fi
    if [ -n "${PRE_COMPILE_ANYBURL_RULES+x}" ] && [ -n "${PRE_COMPILE_ANYBURL_RULES}" ]; then COMPILE_ANYBURL_RULES=$PRE_COMPILE_ANYBURL_RULES; fi
    if [ -n "${PRE_COMPILE_CANDIDATE_MAX+x}" ] && [ -n "${PRE_COMPILE_CANDIDATE_MAX}" ]; then COMPILE_CANDIDATE_MAX=$PRE_COMPILE_CANDIDATE_MAX; fi
    if [ -n "${PRE_COMPILE_ALPHA+x}" ] && [ -n "${PRE_COMPILE_ALPHA}" ]; then COMPILE_ALPHA=$PRE_COMPILE_ALPHA; fi
    if [ -n "${PRE_COMPILE_FALLBACK_TOPK+x}" ] && [ -n "${PRE_COMPILE_FALLBACK_TOPK}" ]; then COMPILE_FALLBACK_TOPK=$PRE_COMPILE_FALLBACK_TOPK; fi
    if [ -n "${PRE_COMPILE_RULE_TOPK+x}" ] && [ -n "${PRE_COMPILE_RULE_TOPK}" ]; then COMPILE_RULE_TOPK=$PRE_COMPILE_RULE_TOPK; fi
fi

# Check PyTorch version
if [ -x ".venv/bin/python3" ]; then
    .venv/bin/python3 -u -c 'import torch; print(torch.__version__)'
else
    python3 -u -c 'import torch; print(torch.__version__)'
fi

# Paths (allow override via env)
CODE_PATH=${CODE_PATH:-'Code'}
DATA_ROOT=${DATA_ROOT:-'data'}
SAVE_ROOT=${SAVE_ROOT:-'model_tuned'}

# Allow .env to provide defaults for key arguments
MODE=${1:-${MODE:-}}
MODEL=${2:-${MODEL_NAME:-}}
DATASET=${3:-${DATASET_NAME:-}}
GPU_DEVICE_ARG=${4:-}

if [ -n "$DATASET" ]; then
    dataset_normalized=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
    case "$dataset_normalized" in
        fb15k-237|fb15k237)
            if [ -z "${DATA_PATH+x}" ]; then DATA_PATH="${DATA_ROOT}/FB15k-237"; fi
            if [ -z "${SAVE_ID+x}" ]; then SAVE_ID="fb15k237_rel"; fi
            if [ -z "${BATCH_SIZE+x}" ]; then BATCH_SIZE=1024; fi
            if [ -z "${NEGATIVE_SAMPLE_SIZE+x}" ]; then NEGATIVE_SAMPLE_SIZE=1024; fi
            if [ -z "${HIDDEN_DIM+x}" ]; then HIDDEN_DIM=768; fi
            if [ -z "${GAMMA+x}" ]; then GAMMA=14; fi
            if [ -z "${ALPHA+x}" ]; then ALPHA=1.2; fi
            if [ -z "${LEARNING_RATE+x}" ]; then LEARNING_RATE=2e-5; fi
            if [ -z "${TYPE_MAP_PATH+x}" ]; then TYPE_MAP_PATH="./fb15k237_entity_type_map.json"; fi
            if [ -z "${TYPE_LAMBDA+x}" ]; then TYPE_LAMBDA=0.15; fi
            ;;
        wn18rr)
            if [ -z "${DATA_PATH+x}" ]; then DATA_PATH="${DATA_ROOT}/wn18rr"; fi
            if [ -z "${SAVE_ID+x}" ]; then SAVE_ID="wn18rr_rel"; fi
            if [ -z "${BATCH_SIZE+x}" ]; then BATCH_SIZE=512; fi
            if [ -z "${NEGATIVE_SAMPLE_SIZE+x}" ]; then NEGATIVE_SAMPLE_SIZE=3072; fi
            if [ -z "${HIDDEN_DIM+x}" ]; then HIDDEN_DIM=1024; fi
            if [ -z "${GAMMA+x}" ]; then GAMMA=16; fi
            if [ -z "${ALPHA+x}" ]; then ALPHA=1.5; fi
            if [ -z "${LEARNING_RATE+x}" ]; then LEARNING_RATE=2.2e-4; fi
            if [ -z "${TYPE_MAP_PATH+x}" ]; then TYPE_MAP_PATH="./wn18rr_entity_type_map.json"; fi
            if [ -z "${TYPE_LAMBDA+x}" ]; then TYPE_LAMBDA=0.2; fi
            ;;
        yago3-10|yago310)
            if [ -z "${DATA_PATH+x}" ]; then DATA_PATH="${DATA_ROOT}/YAGO3-10"; fi
            if [ -z "${SAVE_ID+x}" ]; then SAVE_ID="yago310_rel"; fi
            if [ -z "${BATCH_SIZE+x}" ]; then BATCH_SIZE=512; fi
            if [ -z "${NEGATIVE_SAMPLE_SIZE+x}" ]; then NEGATIVE_SAMPLE_SIZE=2048; fi
            if [ -z "${HIDDEN_DIM+x}" ]; then HIDDEN_DIM=1024; fi
            if [ -z "${GAMMA+x}" ]; then GAMMA=20; fi
            if [ -z "${ALPHA+x}" ]; then ALPHA=1.5; fi
            if [ -z "${LEARNING_RATE+x}" ]; then LEARNING_RATE=7e-5; fi
            if [ -z "${TYPE_MAP_PATH+x}" ]; then TYPE_MAP_PATH="./yago3_entity_type_map.json"; fi
            if [ -z "${TYPE_LAMBDA+x}" ]; then TYPE_LAMBDA=0.15; fi
            ;;
    ogbl-biokg|ogbl_biokg|biokg)
        if [ -z "${DATA_PATH+x}" ]; then DATA_PATH="./data/ogb/ogbl_biokg_kge"; fi
        if [ -z "${SAVE_ID+x}" ]; then SAVE_ID="ogblbiokg_rel"; fi
        if [ -z "${BATCH_SIZE+x}" ]; then BATCH_SIZE=4096; fi
        if [ -z "${NEGATIVE_SAMPLE_SIZE+x}" ]; then NEGATIVE_SAMPLE_SIZE=4096; fi
        if [ -z "${HIDDEN_DIM+x}" ]; then HIDDEN_DIM=2048; fi
        if [ -z "${GAMMA+x}" ]; then GAMMA=12; fi
        if [ -z "${ALPHA+x}" ]; then ALPHA=1.2; fi
        if [ -z "${LEARNING_RATE+x}" ]; then LEARNING_RATE=1e-4; fi
        if [ -z "${TYPE_MAP_PATH+x}" ]; then TYPE_MAP_PATH="./data/ogb/ogbl_biokg_kge/entity_type_map.json"; fi
        if [ -z "${TYPE_LAMBDA+x}" ]; then TYPE_LAMBDA=0.1; fi
        if [ -z "${INVERSE_MAP_PATH+x}" ]; then INVERSE_MAP_PATH="./data/ogb/ogbl_biokg_kge/relation_inverse_map.json"; fi
        ;;
        *)
            echo "[run.sh] DATASET_NAME='$DATASET' not recognized. Supported presets: FB15k-237, WN18RR, YAGO3-10, ogbl-biokg." >&2
            ;;
    esac
fi

SAVE_ID=${5:-${SAVE_ID:-'default_run'}}

if [ -n "$GPU_DEVICE_ARG" ]; then
    GPU_DEVICE="$GPU_DEVICE_ARG"
else
    GPU_DEVICE=${GPU_DEVICE:-0}
fi

if [ -x ".venv/bin/python3" ]; then
    PYTHON_LAUNCHER=(.venv/bin/python3 -u)
else
    PYTHON_LAUNCHER=(python3 -u)
fi

# Compile-only mode: RELATE-Compile (no training)
if [ "$MODE" = "compile" ]; then
    if [ -z "$DATASET" ]; then
        echo "Missing DATASET_NAME for compile mode." >&2
        exit 1
    fi
    if [ -z "$FULL_DATA_PATH" ]; then
        if [ -n "${DATA_PATH:-}" ]; then
            FULL_DATA_PATH=$DATA_PATH
        else
            FULL_DATA_PATH=$DATA_ROOT/$DATASET
        fi
    fi
    COMPILE_OUTPUT_DIR=${COMPILE_OUTPUT_DIR:-"$SAVE_ROOT/RelatE_compile_${DATASET}"}
    COMPILE_PHASE_DIM=${COMPILE_PHASE_DIM:-128}
    COMPILE_EVAL_SPLIT=${COMPILE_EVAL_SPLIT:-valid}
    COMPILE_EVAL_MODE=${COMPILE_EVAL_MODE:-full}
    COMPILE_ANYBURL_RULES=${COMPILE_ANYBURL_RULES:-}
    COMPILE_CANDIDATE_MAX=${COMPILE_CANDIDATE_MAX:-15000}
    COMPILE_ALPHA=${COMPILE_ALPHA:-0.9}
    COMPILE_FALLBACK_TOPK=${COMPILE_FALLBACK_TOPK:-300}
    COMPILE_RULE_TOPK=${COMPILE_RULE_TOPK:-5000}
    COMPILE_SYM_THRESH=${COMPILE_SYM_THRESH:-0.8}
    COMPILE_INV_THRESH=${COMPILE_INV_THRESH:-0.8}
    COMPILE_COMP_THRESH=${COMPILE_COMP_THRESH:-0.5}
    COMPILE_COMP_TOPK=${COMPILE_COMP_TOPK:-5}
    COMPILE_COMP_MAX_PAIRS=${COMPILE_COMP_MAX_PAIRS:-200000}

    "${PYTHON_LAUNCHER[@]}" "$CODE_PATH/compile_driver.py" \
        --data_path "$FULL_DATA_PATH" \
        --output_dir "$COMPILE_OUTPUT_DIR" \
        --phase_dim "$COMPILE_PHASE_DIM" \
        --symmetry_threshold "$COMPILE_SYM_THRESH" \
        --inverse_threshold "$COMPILE_INV_THRESH" \
        --composition_threshold "$COMPILE_COMP_THRESH" \
        --composition_topk "$COMPILE_COMP_TOPK" \
        --composition_max_pairs "$COMPILE_COMP_MAX_PAIRS" \
        --eval_split "$COMPILE_EVAL_SPLIT" \
        --eval_mode "$COMPILE_EVAL_MODE" \
        ${COMPILE_ANYBURL_RULES:+--anyburl_rules "$COMPILE_ANYBURL_RULES"} \
        --candidate_max "$COMPILE_CANDIDATE_MAX" \
        --alpha "$COMPILE_ALPHA" \
        --fallback_topk "$COMPILE_FALLBACK_TOPK" \
        --rule_topk "$COMPILE_RULE_TOPK"
    exit $?
fi

if is_truthy "${REQUIRE_CUDA:-false}"; then
    if ! CUDA_VISIBLE_DEVICES=$GPU_DEVICE "${PYTHON_LAUNCHER[@]}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        echo "CUDA not available for GPU_DEVICE=$GPU_DEVICE; refusing to run on CPU." >&2
        exit 1
    fi
fi
DISTRIBUTED_FLAG=()

# Optional model/training flags from environment
MODEL_FLAG_ARGS=()
if is_truthy "${DOUBLE_ENTITY_EMBEDDING:-false}"; then
    MODEL_FLAG_ARGS+=(--double_entity_embedding)
fi
if is_truthy "${DOUBLE_RELATION_EMBEDDING:-false}"; then
    MODEL_FLAG_ARGS+=(--double_relation_embedding)
fi

ADV_FLAG=()
if is_truthy "${USE_ADVERSARIAL_SAMPLING:-true}"; then
    ADV_FLAG+=(-adv)
fi

# Remaining hyperparameters follow original positional order, but fall back to env defaults
BATCH_SIZE=${6:-${BATCH_SIZE:-}}
NEGATIVE_SAMPLE_SIZE=${7:-${NEGATIVE_SAMPLE_SIZE:-}}
HIDDEN_DIM=${8:-${HIDDEN_DIM:-}}
GAMMA=${9:-${GAMMA:-}}
ALPHA=${10:-${ALPHA:-}}
LEARNING_RATE=${11:-${LEARNING_RATE:-}}
MAX_STEPS=${12:-${MAX_STEPS:-}}
TEST_BATCH_SIZE=${13:-${TEST_BATCH_SIZE:-}}

# Ensure required values exist
if [ -z "$MODE" ] || [ -z "$MODEL" ] || [ -z "$DATASET" ] || [ -z "$BATCH_SIZE" ] || [ -z "$NEGATIVE_SAMPLE_SIZE" ] || [ -z "$HIDDEN_DIM" ] || [ -z "$GAMMA" ] || [ -z "$ALPHA" ] || [ -z "$LEARNING_RATE" ] || [ -z "$MAX_STEPS" ] || [ -z "$TEST_BATCH_SIZE" ]; then
    echo "Missing required parameters. Provide CLI arguments or set defaults via .env."
    exit 1
fi

# Determine full data path: .env DATA_PATH overrides default root/dataset combo
if [ -n "${DATA_PATH:-}" ]; then
    FULL_DATA_PATH=$DATA_PATH
else
    FULL_DATA_PATH=$DATA_ROOT/$DATASET
fi

SAVE=${SAVE_ROOT}/"$MODEL"_"$DATASET"_"$SAVE_ID"

# Default per-run output paths if not provided
RELATION_METRICS_PATH=${RELATION_METRICS_PATH:-"$SAVE/relation_metrics.json"}
EVAL_PREDICTIONS_PATH=${EVAL_PREDICTIONS_PATH:-"$SAVE/eval_predictions.jsonl"}
HARD_NEGATIVE_LOG_PATH=${HARD_NEGATIVE_LOG_PATH:-"$SAVE/hard_negatives.jsonl"}
TEACHER_DEBUG_LOG_PATH=${TEACHER_DEBUG_LOG_PATH:-"$SAVE/teacher_disagreements.jsonl"}

ENV_VALID_STEPS=${VALID_STEPS:-}
ENV_SAVE_STEPS=${SAVE_CHECKPOINT_STEPS:-}
ENV_LOG_STEPS=${LOG_STEPS:-}
ENV_TEST_LOG_STEPS=${TEST_LOG_STEPS:-}

OPTIONAL_ARGS=()
if [ -n "${TYPE_MAP_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--type_map_path "$TYPE_MAP_PATH")
fi
if [ -n "${TYPE_LAMBDA:-}" ]; then
    OPTIONAL_ARGS+=(--type_lambda "$TYPE_LAMBDA")
fi
if [ -n "${INVERSE_MAP_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--inverse_map_path "$INVERSE_MAP_PATH")
fi
if [ -n "${INIT_CHECKPOINT:-}" ]; then
    OPTIONAL_ARGS+=(--init_checkpoint "$INIT_CHECKPOINT")
fi
if [ -n "${COMPILED_INIT_DIR:-}" ]; then
    OPTIONAL_ARGS+=(--compiled_init_dir "$COMPILED_INIT_DIR")
fi
if [ -n "${PATH_LOSS_WEIGHT:-}" ]; then
    OPTIONAL_ARGS+=(--path_loss_weight "$PATH_LOSS_WEIGHT")
fi
if [ -n "${PATH_BATCH_SIZE:-}" ] && [ "${PATH_BATCH_SIZE}" -gt 0 ]; then
    OPTIONAL_ARGS+=(--path_batch_size "$PATH_BATCH_SIZE")
fi
if [ -n "${PATH_NEGATIVE_SIZE:-}" ] && [ "${PATH_NEGATIVE_SIZE}" -gt 0 ]; then
    OPTIONAL_ARGS+=(--path_negative_size "$PATH_NEGATIVE_SIZE")
fi
if [ -n "${PATH_HOPS:-}" ]; then
    read -ra PATH_HOPS_ARR <<< "$PATH_HOPS"
    OPTIONAL_ARGS+=(--path_hops "${PATH_HOPS_ARR[@]}")
fi
if [ -n "${PATH_MAX_PER_HOP:-}" ] && [ "${PATH_MAX_PER_HOP}" -gt 0 ]; then
    OPTIONAL_ARGS+=(--path_max_per_hop "$PATH_MAX_PER_HOP")
fi
if [ -n "${PATH_CONSISTENCY_WEIGHT:-}" ] && [ "$(printf '%.6f' "${PATH_CONSISTENCY_WEIGHT}")" != "0.000000" ]; then
    OPTIONAL_ARGS+=(--path_consistency_weight "$PATH_CONSISTENCY_WEIGHT")
fi
if [ -n "${PATH_MARGIN:-}" ] && [ "$(printf '%.6f' "${PATH_MARGIN}")" != "0.000000" ]; then
    OPTIONAL_ARGS+=(--path_margin "$PATH_MARGIN")
fi
if [ -n "${PATH_CONSISTENCY_MARGIN:-}" ] && [ "$(printf '%.6f' "${PATH_CONSISTENCY_MARGIN}")" != "0.000000" ]; then
    OPTIONAL_ARGS+=(--path_consistency_margin "$PATH_CONSISTENCY_MARGIN")
fi
if [ -n "${PATH_CURRICULUM_STEPS:-}" ]; then
    read -ra PATH_CURR_ARR <<< "$PATH_CURRICULUM_STEPS"
    OPTIONAL_ARGS+=(--path_curriculum_steps "${PATH_CURR_ARR[@]}")
fi
if is_truthy "${USE_REL_PROMPT_EMB:-false}"; then
    OPTIONAL_ARGS+=(--use_rel_prompt_emb)
fi
if [ -n "${REL_PROMPT_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--rel_prompt_path "$REL_PROMPT_PATH")
fi
if [ -n "${REL_PROMPT_WEIGHT:-}" ]; then
    OPTIONAL_ARGS+=(--rel_prompt_weight "$REL_PROMPT_WEIGHT")
fi
if [ -n "${REL_PROMPT_WARMUP_STEPS:-}" ]; then
    OPTIONAL_ARGS+=(--rel_prompt_warmup_steps "$REL_PROMPT_WARMUP_STEPS")
fi
if is_truthy "${USE_ENTITY_PROMPT_EMB:-false}"; then
    OPTIONAL_ARGS+=(--use_entity_prompt_emb)
fi
if [ -n "${ENTITY_PROMPT_WEIGHT:-}" ]; then
    OPTIONAL_ARGS+=(--entity_prompt_weight "$ENTITY_PROMPT_WEIGHT")
fi
if [ -n "${ENTITY_PROMPT_WARMUP_STEPS:-}" ]; then
    OPTIONAL_ARGS+=(--entity_prompt_warmup_steps "$ENTITY_PROMPT_WARMUP_STEPS")
fi
if [ -n "${KD_WARMUP_STEPS:-}" ]; then
    OPTIONAL_ARGS+=(--kd_warmup_steps "$KD_WARMUP_STEPS")
fi
if [ -n "${KD_DECAY_START:-}" ]; then
    OPTIONAL_ARGS+=(--kd_decay_start "$KD_DECAY_START")
fi
if [ -n "${KD_DECAY_DURATION:-}" ]; then
    OPTIONAL_ARGS+=(--kd_decay_duration "$KD_DECAY_DURATION")
fi
if [ -n "${RELATION_SAMPLING_WEIGHTS:-}" ]; then
    OPTIONAL_ARGS+=(--relation_sampling_weights "$RELATION_SAMPLING_WEIGHTS")
fi
if [ -n "${EXTRA_HARD_RELATIONS:-}" ]; then
    read -ra EXTRA_HARD_REL_ARR <<< "$EXTRA_HARD_RELATIONS"
    OPTIONAL_ARGS+=(--extra_hard_relations "${EXTRA_HARD_REL_ARR[@]}")
fi
if [ -n "${TRAIN_ANYBURL_RULES:-}" ]; then
    OPTIONAL_ARGS+=(--train_anyburl_rules "$TRAIN_ANYBURL_RULES")
fi
if [ -n "${CANDIDATE_NEGATIVE_FRACTION:-}" ]; then
    OPTIONAL_ARGS+=(--candidate_negative_fraction "$CANDIDATE_NEGATIVE_FRACTION")
fi
if [ -n "${CANDIDATE_RULE_TOPK:-}" ]; then
    OPTIONAL_ARGS+=(--candidate_rule_topk "$CANDIDATE_RULE_TOPK")
fi
if [ -n "${CANDIDATE_FALLBACK_TOPK:-}" ]; then
    OPTIONAL_ARGS+=(--candidate_fallback_topk "$CANDIDATE_FALLBACK_TOPK")
fi
if [ -n "${CANDIDATE_CACHE_MAX:-}" ]; then
    OPTIONAL_ARGS+=(--candidate_cache_max "$CANDIDATE_CACHE_MAX")
fi
if [ -n "${EMU_NEGATIVE_FRACTION:-}" ]; then
    OPTIONAL_ARGS+=(--emu_negative_fraction "$EMU_NEGATIVE_FRACTION")
fi
if [ -n "${EMU_NUM_WALKS:-}" ]; then
    OPTIONAL_ARGS+=(--emu_num_walks "$EMU_NUM_WALKS")
fi
if [ -n "${EMU_WALK_LENGTH:-}" ]; then
    OPTIONAL_ARGS+=(--emu_walk_length "$EMU_WALK_LENGTH")
fi
if [ -n "${EMU_CACHE_SIZE:-}" ]; then
    OPTIONAL_ARGS+=(--emu_cache_size "$EMU_CACHE_SIZE")
fi
if [ -n "${EMU_RELATION_QUOTA:-}" ]; then
    OPTIONAL_ARGS+=(--emu_relation_quota "$EMU_RELATION_QUOTA")
fi
if [ -n "${PLM_ENTITY_EMB_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--plm_entity_emb_path "$PLM_ENTITY_EMB_PATH")
fi
if [ -n "${PLM_REL_EMB_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--plm_relation_emb_path "$PLM_REL_EMB_PATH")
fi
if [ -n "${PLM_ENTITY_REG_WEIGHT:-}" ]; then
    OPTIONAL_ARGS+=(--plm_entity_reg_weight "$PLM_ENTITY_REG_WEIGHT")
fi
if [ -n "${PLM_REL_REG_WEIGHT:-}" ]; then
    OPTIONAL_ARGS+=(--plm_relation_reg_weight "$PLM_REL_REG_WEIGHT")
fi
if is_truthy "${FULL_RANKING_CE:-false}"; then
    OPTIONAL_ARGS+=(--full_ranking_ce)
fi
if [ -n "${FULL_RANKING_CHUNK_SIZE:-}" ]; then
    OPTIONAL_ARGS+=(--full_ranking_chunk_size "$FULL_RANKING_CHUNK_SIZE")
fi
if [ -n "${FULL_RANKING_LABEL_SMOOTHING:-}" ]; then
    OPTIONAL_ARGS+=(--full_ranking_label_smoothing "$FULL_RANKING_LABEL_SMOOTHING")
fi

if [ $MODE == "train" ]
then
    echo "Start Training......"
    mkdir -p "$SAVE"
    LOG_FILE="$SAVE/train.log"
    if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ -z "${PYTORCH_ALLOC_CONF:-}" ]; then
        export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    fi
    export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
    unset PYTORCH_CUDA_ALLOC_CONF
    {
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE "${PYTHON_LAUNCHER[@]}" $CODE_PATH/driver.py --do_train \
            --cuda \
            --do_valid \
            --do_test \
            --data_path $FULL_DATA_PATH \
            --model $MODEL \
            "${MODEL_FLAG_ARGS[@]}" \
            -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
            -g $GAMMA -a $ALPHA "${ADV_FLAG[@]}" \
            -lr $LEARNING_RATE --max_steps $MAX_STEPS \
            -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
            "${OPTIONAL_ARGS[@]}" \
            ${ENV_VALID_STEPS:+--valid_steps $ENV_VALID_STEPS} \
            ${ENV_SAVE_STEPS:+--save_checkpoint_steps $ENV_SAVE_STEPS} \
            ${ENV_LOG_STEPS:+--log_steps $ENV_LOG_STEPS} \
            ${ENV_TEST_LOG_STEPS:+--test_log_steps $ENV_TEST_LOG_STEPS} \
            "${@:14}"
    } 2>&1 | tee -a "$LOG_FILE"

elif [ $MODE == "valid" ]
then
    echo "Start Evaluation on Valid Data Set......"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/driver.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]
then
    echo "Start Evaluation on Test Data Set......"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/driver.py --do_test --cuda -init $SAVE

else
    echo "Unknown MODE" $MODE
fi
