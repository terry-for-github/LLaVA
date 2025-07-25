#!/bin/bash

CONFIG=llava-llama32-3b-pretrain
RESUME_FROM_CHECKPOINT=false

echo HF_HOME=/userhome/huggingface > .deepspeed_env
echo https_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo http_proxy=http://127.0.0.1:7890 >> .deepspeed_env
echo TRANSFORMERS_OFFLINE=1 >> .deepspeed_env
echo WANDB_PROJECT=$CONFIG >> .deepspeed_env

RUN_NAME=$CONFIG-$(date +"%y-%m%d-%H%M")
deepspeed -i localhost:4,5,6,7 llava/train/train_mem.py \
    --config configs/$CONFIG.toml \
    --report_to wandb \
    --run_name $RUN_NAME \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    $@ | tee logs/$RUN_NAME.log