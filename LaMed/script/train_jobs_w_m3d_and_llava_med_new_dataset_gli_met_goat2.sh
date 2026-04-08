#!/bin/bash

# m3d job
output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-combined-finetune-freeze-viz-again-new-dataset-gli_met_goat_no_pretrain_v11-0000
train_path=/local2/amvepa91/MedTrinity-25M/brats_gli_met_goat_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json
val_path=/local2/amvepa91/MedTrinity-25M/brats_gli_met_goat_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json
test_path=/local2/amvepa91/MedTrinity-25M/brats_gli_met_goat_3d_vqa_subjTrue_test_updated_v11_seed0_multitask_fixed.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--output_dir $output_dir/eval_vqa1 \
--model_name_or_path "$output_dir"/hf \
--vqa_data_test_path $test_path \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa1 \
--gt_file $test_path \