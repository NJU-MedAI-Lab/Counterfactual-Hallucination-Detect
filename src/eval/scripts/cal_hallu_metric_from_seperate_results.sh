#!/bin/bash
export PYTHONPATH=${PROJECT_ROOT:-.}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

SYS_ENABLE=False
CHECKPOINT_NAME="best_model_ckpt"
TEST_NAME="qwen3vl-2b-lora-finetune-term1add-unvisible-SYSFalse-batch16-lr1e-4-newphrase-withwronglesion"
TAU=0.8
RESULT_TYPE="seperate"

# TAU for loop
# for TAU in $(seq 0 0.05 1)
# do

# echo "Running with tau=$TAU"

UNCERTAINTY_TYPE="logits"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="All"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --con_root_dir ${COUNTER_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval_counter}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --hallu_data_dir ${HALLU_DATA_ROOT:-YOUR_HALLU_DATA_ROOT}/${HALLU_NUM}_hallu_test_qwen_format_new \
  --output_dir ${HALLU_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/hallu_metrics}/${TAU}/${TEST_NAME}-${CHECKPOINT_NAME}-${UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual


UNCERTAINTY_TYPE="logits"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="Half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --con_root_dir ${COUNTER_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval_counter}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --hallu_data_dir ${HALLU_DATA_ROOT:-YOUR_HALLU_DATA_ROOT}/${HALLU_NUM}_hallu_test_qwen_format_new \
  --output_dir ${HALLU_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/hallu_metrics}/${TAU}/${TEST_NAME}-${CHECKPOINT_NAME}-${UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual

UNCERTAINTY_TYPE="sample"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="Half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --con_root_dir ${COUNTER_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval_counter}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --hallu_data_dir ${HALLU_DATA_ROOT:-YOUR_HALLU_DATA_ROOT}/${HALLU_NUM}_hallu_test_qwen_format_new \
  --output_dir ${HALLU_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/hallu_metrics}/${TAU}/${TEST_NAME}-${CHECKPOINT_NAME}-${UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual

UNCERTAINTY_TYPE="sample"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="All"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --con_root_dir ${COUNTER_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval_counter}/Hallu-Esti/${TEST_NAME}-${CHECKPOINT_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --hallu_data_dir ${HALLU_DATA_ROOT:-YOUR_HALLU_DATA_ROOT}/${HALLU_NUM}_hallu_test_qwen_format_new \
  --output_dir ${HALLU_RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/hallu_metrics}/${TAU}/${TEST_NAME}-${CHECKPOINT_NAME}-${UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual

# done
