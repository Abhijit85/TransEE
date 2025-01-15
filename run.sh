#!/bin/sh

# Check PyTorch version
python -u -c 'import torch; print(torch.__version__)'

# Paths
CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

# Required parameters
MODE=$1                # Mode: train, valid, or test
MODEL=$2               # Model: TransEEnhanced, HAKE, ModE
DATASET=$3             # Dataset name (e.g., FB15k-237)
GPU_DEVICE=$4          # GPU device ID
SAVE_ID=$5             # Identifier for the save directory

# Full paths
FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

# Model parameters
BATCH_SIZE=$6          # Batch size for training
NEGATIVE_SAMPLE_SIZE=$7 # Number of negatives per positive
HIDDEN_DIM=$8          # Embedding dimension
GAMMA=$9               # Margin for margin-based loss
ALPHA=${10}            # Adversarial sampling temperature
LEARNING_RATE=${11}    # Learning rate
MAX_STEPS=${12}        # Maximum training steps
TEST_BATCH_SIZE=${13}  # Batch size for testing
MODULUS_WEIGHT=${14}   # Modulus weight (for TransEEnhanced/HAKE)
PHASE_WEIGHT=${15}     # Phase weight (for TransEEnhanced/HAKE)

# Execute based on mode
if [ "$MODE" = "train" ]; then
    echo "Starting Training..."
    python $CODE_PATH/driver.py \
        --mode train \
        --model $MODEL \
        --data_path $FULL_DATA_PATH \
        --save_path $SAVE \
        -n $NEGATIVE_SAMPLE_SIZE \
        -b $BATCH_SIZE \
        -d $HIDDEN_DIM \
        -g $GAMMA \
        -a $ALPHA \
        -lr $LEARNING_RATE \
        --max_steps $MAX_STEPS \
        --test_batch_size $TEST_BATCH_SIZE \
        -mw $MODULUS_WEIGHT \
        -pw $PHASE_WEIGHT \
        --gpu $GPU_DEVICE

elif [ "$MODE" = "valid" ]; then
    echo "Starting Validation..."
    python $CODE_PATH/driver.py \
        --mode valid \
        --model $MODEL \
        --data_path $FULL_DATA_PATH \
        --save_path $SAVE \
        --gpu $GPU_DEVICE

elif [ "$MODE" = "test" ]; then
    echo "Starting Testing..."
    python $CODE_PATH/driver.py \
        --mode test \
        --model $MODEL \
        --data_path $FULL_DATA_PATH \
        --save_path $SAVE \
        --gpu $GPU_DEVICE

else
    echo "Unknown MODE: $MODE"
    exit 1
fi
