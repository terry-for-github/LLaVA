#!/bin/bash

CKPT="$1"
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file /userhome/eval_answers/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode llava_llama_3

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /userhome/eval_answers/textvqa/answers/$CKPT.jsonl
