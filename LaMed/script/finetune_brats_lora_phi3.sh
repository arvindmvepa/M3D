#!/bin/bash

# run "accelerate config" first!
output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-finetune-0000
accelerate launch --gpu_ids $1 LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type phi3 \
    --vqa_data_train_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_v2.json \
    --vqa_data_val_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_v2.json \
    --vqa_data_test_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard

PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python LaMed/src/utils/merge_lora_weights_and_save_hf_model.py \
--version="" --model_type="phi3" \
--model_with_lora="$output_dir"/model_with_lora.bin \
--output_dir="$output_dir"/hf

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--model_name_or_path "$output_dir"/hf \
--vqa_data_test_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json \

