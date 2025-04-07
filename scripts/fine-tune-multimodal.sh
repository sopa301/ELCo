#!/bin/bash

# Please define your own path here
huggingface_path=./huggingface-emoji/

# Multimodal models
for model_name in bert-base
do
    for seed in 43 44 45 46 47
    do
        CUDA_VISIBLE_DEVICES=0 python scripts/emote_multimodal.py --finetune 1 --model_name $model_name --portion 1 --seed $seed --hfpath $huggingface_path --use_images
    done
done 