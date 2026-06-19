#!/bin/bash
export PYTHONPATH=${PROJECT_ROOT:-.}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

TEST_NAME="qwen3vl-2b-lora-finetune-term1add-unvisible-SYSTrue-batch16-lr1e-4-newphrase-withwronglesion"
TAU=0.6
RESULT_TYPE="combined"
ENTITY_ESTRACT=False
CONTERFACTUAL_TEST=True
SYS_ENABLE=True

UNCERTAINTY_TYPE="logits"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/results.json \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --output_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual

python src/eval/cal_metric_permodality.py \
  --results_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/combined_hallu_metric_uncertainty-${UNCERTAINTY_TYPE}_tau${TAU}-CON${CONTERFACTUAL_TEST}.json

UNCERTAINTY_TYPE="sample"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/results.json \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --output_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU \
  --use_counterfactual

python src/eval/cal_metric_permodality.py \
  --results_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/combined_hallu_metric_uncertainty-${UNCERTAINTY_TYPE}_tau${TAU}-CON${CONTERFACTUAL_TEST}.json

UNCERTAINTY_TYPE="logits"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/results.json \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --output_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU 

python src/eval/cal_metric_permodality.py \
  --results_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/combined_hallu_metric_uncertainty-${UNCERTAINTY_TYPE}_tau${TAU}-CONFalse.json

UNCERTAINTY_TYPE="sample"  # "logits", "sample"
RESULT_UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
HALLU_NUM="half"  # "Half", "All"

python src/eval/cal_hallu_metric_from_results.py \
  --result_type $RESULT_TYPE \
  --sup_root_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/results.json \
  --image_root_dir ${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR} \
  --output_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM} \
  --uncertainty_type $UNCERTAINTY_TYPE \
  --tau $TAU 

python src/eval/cal_metric_permodality.py \
  --results_dir ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/results}/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${RESULT_UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}/combined_hallu_metric_uncertainty-${UNCERTAINTY_TYPE}_tau${TAU}-CONFalse.json