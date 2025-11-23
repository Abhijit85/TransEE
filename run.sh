#!/bin/bash

# Load .env if present
is_truthy() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -f ".env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# Check PyTorch version
python3 -u -c 'import torch; print(torch.__version__)'

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

PYTHON_LAUNCHER=(python3 -u)
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

if [ $MODE == "train" ]
then
    echo "Start Training......"
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
