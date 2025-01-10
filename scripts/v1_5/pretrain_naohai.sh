#!/bin/bash
echo HF_HOME=/userhome/huggingface > .deepspeed_env
echo https_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo http_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo TRANSFORMERS_OFFLINE=1 >> .deepspeed_env
echo WANDB_PROJECT=naohai >> .deepspeed_env
    

RUN_NAME=pretrain_naohai_7b_gbc4
NUM_TRIAL=1

deepspeed \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero0.json \
    --model_name_or_path ./naohai_7b \
    --version plain \
    --data_path_list ./playground/pretrain/blip_laion_cc_sbu_558k.json \
    playground/image_caption/DCI/ANNO/DCI_8K_exist.json \
    playground/image_caption/DenseFusion/ANNO/DF_1M_exist.json \
    playground/image_caption/DOCCI/ANNO/DOCCI_15K_exist.json \
    playground/image_caption/GBC-10M/train_4_clean_exist.json \
    playground/image_caption/MMInstruct/ANNO/MMInstruct-18K_exist.json \
    playground/image_caption/ShareGPT4V/ANNO/ShareGPT4V_102K_exist.json \
    --image_folder ./playground/pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./ckpts/llava-naohai-7b-pretrain-gbc4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME-$NUM_TRIAL.log
