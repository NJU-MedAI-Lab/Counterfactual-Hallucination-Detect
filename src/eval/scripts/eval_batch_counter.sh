#!/bin/bash

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="7"

IMAGE_ROOT=${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR}
DATA_DIR=${DATA_DIR:-YOUR_QWEN_FORMAT_DATA_DIR}
BATCH_SIZE=8
DTYPE=bfloat16

# 模型 & checkpoint 配置
TEST_NAMES=(
"/Hallu-Esti/qwen3vl-2b-lora-finetune-term1add-unvisible-SYSFalse-batch16-lr1e-4-newphrase-withwronglesion"
)

CHECKPOINT_NAMES=(
  "best_model_ckpt"
)

UNCERTAINTY_TYPE="all"  # "logits", "sample", "all", "none"
SAMPLE_NUM=5  # 仅在 sample 或 all 模式下使用
TEMPERATURE=0.7  # 采样温度，仅在 sample 或 all 模式下使用
TOP_P=0.9  # 采样 top-p，仅在 sample 或 all 模式下使用


# 遍历
for i in "${!TEST_NAMES[@]}"; do
  TEST_NAME="${TEST_NAMES[$i]}"
  CHECKPOINT_NAME="${CHECKPOINT_NAMES[$i]}"

  MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/outputs/checkpoints}/${TEST_NAME}/${CHECKPOINT_NAME}/hf"
  OUTPUT_PATH="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/batch_eval_counter}/${TEST_NAME}-${CHECKPOINT_NAME}-${UNCERTAINTY_TYPE}-uncertainty"

  echo "=========================================="
  echo "Evaluating: ${TEST_NAME} | ${CHECKPOINT_NAME}"
  echo "Model path: ${MODEL_PATH}"
  echo "Output dir: ${OUTPUT_PATH}"
  echo "=========================================="

  python ${PROJECT_ROOT}/src/eval/eval_batch.py \
    --model_path "${MODEL_PATH}" \
    --image_root_path "${IMAGE_ROOT}" \
    --data_dir "${DATA_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --uncertainty_type "${UNCERTAINTY_TYPE}" \
    --sample_num "${SAMPLE_NUM}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --test_contradictory

done
