#!/bin/bash
# Workspace
WORK_SPACE=$(pwd)
TR_RECORD_DIR="${WORK_SPACE}/TFRecords"

MODEL_DIR="${WORK_SPACE}/Model"

cd im2txt

# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/evaluate \
  --input_file_pattern="${TR_RECORD_DIR}/val-?????-of-00004" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval"