#!/bin/bash

export PYTHONPATH=${PROJECT_ROOT:-.}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="6"

IMAGE_ROOT=${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR}
DATA_DIR=${HALLU_DATA_ROOT:-YOUR_HALLU_DATA_ROOT}/IMIS_data_support_and_contradict_queries_test_qwen_format
BATCH_SIZE=8
DTYPE=bfloat16

# 模型 & checkpoint 配置
TEST_NAMES=(
"Qwen3-VL-2B-Instruct"
)

# 遍历
for i in "${!TEST_NAMES[@]}"; do
  TEST_NAME="${TEST_NAMES[$i]}"

  MODEL_PATH="${MODEL_ROOT:-YOUR_MODEL_ROOT}/${TEST_NAME}"
  OUTPUT_PATH="${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/${TEST_NAME}"

  echo "=========================================="
  echo "Evaluating: ${TEST_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Output dir: ${OUTPUT_PATH}"
  echo "=========================================="

  python src/eval/eval_batch.py \
    --model_path "${MODEL_PATH}" \
    --image_root_path "${IMAGE_ROOT}" \
    --data_dir "${DATA_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}"

done