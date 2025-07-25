#!/bin/bash
echo HF_HOME=/userhome/huggingface > .deepspeed_env
echo https_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo http_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo TRANSFORMERS_OFFLINE=1 >> .deepspeed_env
echo WANDB_PROJECT=qwen_ex >> .deepspeed_env

RUN_NAME=finetune_qwen25_7b_sgemf_ovmmi_0118
NUM_TRIAL=1
    # --data_path ./playground/finetune/llava_v1_5_mix665k.json \

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --version qwen \
    --data_path_list ./playground/onevision \
    playground/MMInstruct-GPT4V/jsons_all/qa_en_clean.json \
    --image_folder ./playground/finetune \
    --vision_tower google/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter ./ckpts/llava-qwen25-7b-pretrain-sgemf-0115/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./ckpts/llava-qwen25-7b-sgemf-ovmmi-0118\
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "no" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 0.5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME | tee logs/$RUN_NAME-$NUM_TRIAL.log
