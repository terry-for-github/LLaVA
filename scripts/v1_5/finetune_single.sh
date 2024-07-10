#!/bin/bash

MODEL_NAME=llava-llama3-graph
PRETRAIN_MODEL_NAME=${MODEL_NAME}-pretrain

accelerate launch \
    --config_file single_node_zero3.yaml \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version llama_v3 \
    --data_path ./playground/data/llava_v1_5_mix128.json \
    --image_folder ./playground/data \
    --vision_tower moe-vision-tower \
    --vision_experts_list microsoft/layoutlmv3-large facebook/dinov2-giant openai/clip-vit-large-patch14-336 \
    --m_token_one_patch 2 2 1 \
    --pretrain_mm_mlp_adapter ./checkpoints/${PRETRAIN_MODEL_NAME}-toy/mm_projector.bin \
    --mm_projector_type mousi \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/${MODEL_NAME}-toy \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
