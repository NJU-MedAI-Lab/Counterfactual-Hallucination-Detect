#!/bin/bash
export PYTHONPATH=${PROJECT_ROOT:-.}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

python src/eval/cal_metric.py \
  --result_root ${RESULT_ROOT:-${PROJECT_ROOT:-.}/outputs/batch_eval}/Hallu-Esti/2b-sft-finetune-term1-add-unvisible-sys-batch8-lr1e-5-best_model_ckpt-logits-uncertainty/ \
  --uncertainty_type all