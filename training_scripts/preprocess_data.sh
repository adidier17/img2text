#!/bin/bash
WORK_SPACE="$(pwd)"
BASE_PATH="/Users/akdidier/Documents/Maars/compare_to_tanvir"
TRAIN_IMAGES="${BASE_PATH}/train"
TRAIN_CAPTIONS="${BASE_PATH}/captions/train.txt"
TEST_IMAGES="${BASE_PATH}/test"
TEST_CAPTIONS="${BASE_PATH}/captions/test.txt"
TF_RECORDS_OUTPUT_DIR="TFRecords"
PROCESSED_CAPTIONS_DIR="${WORK_SPACE}/processed_captions"

mkdir -p ${PROCESSED_CAPTIONS_DIR}
mkdir -p ${TF_RECORDS_OUTPUT_DIR}

python build_image_data.py \
  --train_image_dir="${TRAIN_IMAGES}" \
  --train_captions="${TRAIN_CAPTIONS}" \
  --train_processed_captions="${PROCESSED_CAPTIONS_DIR}/train.json" \
  --test_image_dir="${TEST_IMAGES}" \
  --test_captions="${TEST_CAPTIONS}" \
  --test_processed_captions="${PROCESSED_CAPTIONS_DIR}/test.json" \
  --output_dir="${TF_RECORDS_OUTPUT_DIR}" \
  --word_counts_output_file="${TF_RECORDS_OUTPUT_DIR}/word_counts.txt" \
  --train_shards=256 \
  --test_shards=4 \
  #--test_shards=8