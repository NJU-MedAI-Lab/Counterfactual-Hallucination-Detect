#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
export WANDB_PROJECT=Qwen3-VL-PROJECTS
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

# auto-detect number of GPUs
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=${MODEL_NAME_OR_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}

# Training hyperparameters
lr=1e-4
disrupt_weight=0.05
adv_weight=3.0
batch_size=16
grad_accum_steps=8
SYS_ENABLE=False

# Training entry point
entry_file=${PROJECT_ROOT}/src/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=hallu_esti_training,hallu_esti_training_unvisible_medgemma,hallu_esti_wrong_lesion
eval_datasets=hallu_esti_eval

# Output configuration
run_name="qwen3vl-2b-lora-finetune-term1-add-unvisible-SYS${SYS_ENABLE}-batch${batch_size}-lr${lr}-newphrase-withwronglesion"
output_dir=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/checkpoints/${run_name}"}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --sys_enable ${SYS_ENABLE} \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --learning_rate ${lr} \
    --weight_decay 0.0001 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
