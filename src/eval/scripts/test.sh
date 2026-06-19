#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export TRANSFORMERS_VERBOSITY=error
export DS_LOG_LEVEL=error

# auto-detect number of GPUs
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

CHECKPOINT_NAME="best_model_ckpt"
TEST_NAME="qwen3vl-2b-lora-finetune-term1add-unvisible-SYSTrue-batch16-lr1e-4-newphrase-withwronglesion"
HALLU_NUM="half"  # "half", "all"
UNCERTAINTY_TYPE="all"  # "logits", "sample", "all"
ENTITY_ESTRACT=True
CONTERFACTUAL_TEST=True
SYS_ENABLE=True

# Model configuration
llm=${MODEL_NAME_OR_PATH:-"${PROJECT_ROOT}/outputs/checkpoints/${TEST_NAME}/${CHECKPOINT_NAME}/hf"}

# Test entry point
entry_file=${PROJECT_ROOT}/src/eval/test.py

# Dataset configuration (replace with public dataset names)
datasets=imis_train_grounding
eval_datasets=hallu_esti_test_${HALLU_NUM}

# Output configuration
output_dir=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/results/Hallu_Esti_results_entity${ENTITY_ESTRACT}_CON${CONTERFACTUAL_TEST}_SYS${SYS_ENABLE}/${TEST_NAME}-${UNCERTAINTY_TYPE}-uncertainty-${HALLU_NUM}"}

# Test arguments
args="
    --test_only True\
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten False \
    --bf16 \
    --output_dir ${output_dir} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --dataloader_num_workers 8 \
    --test_batch_size 4 \
    --test_result_file "results.json" \
    --image_root_dir "${IMAGE_ROOT_DIR:-YOUR_IMAGE_ROOT_DIR}" \
    --sample_num 5 \
    --temperature 0.7 \
    --top_p 0.9 \
    --uncertainty_type "${UNCERTAINTY_TYPE}" \
    --do_entity_extract ${ENTITY_ESTRACT} \
    --entity_extract_model "gpt-4.1-mini" \
    --cf_model "gpt-4.1-mini" \
    --do_counterfactual_test ${CONTERFACTUAL_TEST} \
    --sys_enable ${SYS_ENABLE}
"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

