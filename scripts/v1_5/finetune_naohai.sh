#!/bin/bash
echo HF_HOME=/userhome/huggingface > .deepspeed_env
echo https_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo http_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo TRANSFORMERS_OFFLINE=1 >> .deepspeed_env
echo WANDB_PROJECT=naohai >> .deepspeed_env

RUN_NAME=finetune_naohai_7b
NUM_TRIAL=8

deepspeed -H /hostfile -i node_13:0,1,2,3,4,5,6,7@node_05:0,1,2,3,4,5,6,7 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json  \
    --model_name_or_path ./naohai_7b \
    --version baichuan \
    --data_path ./playground/finetune/llava_v1_5_mix665k.json \
    --image_folder ./playground/finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./ckpts/llava-naohai-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./ckpts/llava-naohai-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 20000 \
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
    --report_to wandb \
    --disable_tqdm True \
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME-$NUM_TRIAL.log
