#!/bin/bash

SPLIT="mmbench_${2}_cn_20231003"
CKPT="llava-v1.5-13b-lora-$1"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/$CKPT \
    --model-base lmsys/vicuna-13b-v1.5 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
