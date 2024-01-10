#!/bin/bash

CKPT="llava-v1.5-13b-lora-$1"
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$CKPT \
    --model-base lmsys/vicuna-13b-v1.5 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/mm-vet-$CKPT.json

