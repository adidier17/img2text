#!/bin/bash
# Workspace
WORK_SPACE="$(pwd)"
TR_RECORD_DIR="${WORK_SPACE}/TFRecords"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${WORK_SPACE}/Inception_V3/inception_v3.ckpt"
#try using the 1M iters checkpoint file
#INCEPTION_CHECKPOINT="${WORK_SPACE}/1M_iters_ckpt/model.ckpt-1000000.data-00000-of-00001"

# Directory to save the model.
MODEL_DIR="${WORK_SPACE}/Model"

# Build the model.
cd im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${TR_RECORD_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=8360
