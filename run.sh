#!/bin/bash

# Load .env if present
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
SAVE_ID=${5:-${SAVE_ID:-'default_run'}}

if [ -n "$GPU_DEVICE_ARG" ]; then
    GPU_DEVICE="$GPU_DEVICE_ARG"
else
    GPU_DEVICE=${GPU_DEVICE:-0}
fi

# Remaining hyperparameters follow original positional order
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}

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

if [ $MODE == "train" ]
then
    echo "Start Training......"
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/driver.py --do_train \
        --cuda \
        --do_valid \
        --do_test \
        --data_path $FULL_DATA_PATH \
        --model $MODEL \
        -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
        -g $GAMMA -a $ALPHA -adv \
        -lr $LEARNING_RATE --max_steps $MAX_STEPS \
        -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
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
