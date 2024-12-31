#!/bin/bash
#!/bin/bash
echo HF_HOME=/userhome/huggingface > .deepspeed_env
echo https_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo http_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo TRANSFORMERS_OFFLINE=1 >> .deepspeed_env
echo WANDB_PROJECT=qwen_ex >> .deepspeed_env
echo NCCL_TIMEOUT=1200000 >> .deepspeed_env

RUN_NAME=finetune_qwen25_7b_sgemf_ovmmi_1219_short_6k_lmu05
NUM_TRIAL=1
    # --data_path ./playground/finetune/llava_v1_5_mix665k.json \

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ckpts/llava-qwen25-7b-sgemf-ovmmi-1219-short/checkpoint-6000 \
    --version qwen \
    --data_path ./playground/finetune/lmu_data.json \
    --image_folder /userhome/Dataset/LMUData/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./ckpts/llava-qwen25-7b-sgemf-ovmmi-1219-short-6k-lmu05 \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --continue_finetune true \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
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
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME-$NUM_TRIAL.log
