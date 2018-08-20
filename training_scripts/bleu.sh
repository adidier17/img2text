WORK_SPACE="$(pwd)"
BASE_PATH="/Users/akdidier/Documents/Maars/compare_to_tanvir"
TEST_IMAGES="${BASE_PATH}/test"
MODEL_DIR="${WORK_SPACE}/Model"
TF_RECORDS_OUTPUT_DIR="${WORK_SPACE}/TFRecords"
PROCESSED_CAPTIONS_DIR="${WORK_SPACE}/processed_captions"

#Build
#cd im2txt/im2txt
cd im2txt
#bazel build -c opt //im2txt:calculate_bleu
bazel build -c opt //im2txt/...
ls
#python calculate_bleu.py
# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/calculate_bleu \
  --test_images_dir="${TEST_IMAGES}" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --vocab_file="${TF_RECORDS_OUTPUT_DIR}/word_counts.txt" \
  --eval_dir="${MODEL_DIR}/eval" \
  --test_captions_file="${PROCESSED_CAPTIONS_DIR}/test.json" \
