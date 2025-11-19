#!/bin/bash

# run "accelerate config" first!
output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-combined-finetune-freeze-viz-again-new-dataset-v11-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json
test_path=/local2/amvepa91/clinical_validation_gli_test_set.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--output_dir $output_dir/eval_vqa2 \
--model_name_or_path "$output_dir"/hf \
--vqa_data_test_path $test_path \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa2 \
--gt_file $test_path \