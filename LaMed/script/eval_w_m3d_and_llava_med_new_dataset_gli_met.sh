#!/bin/bash

# m3d job
output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-combined-finetune-freeze-viz-again-new-dataset-gli_met_v11-0000
test_path=/local2/amvepa91/MedTrinity-25M/brats_gli_met_3d_vqa_subjTrue_train_updated_v11_seed0.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--output_dir $output_dir/eval_vqa2 \
--model_name_or_path "$output_dir"/hf \
--vqa_data_test_path $test_path \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa2 \
--gt_file $test_path \


# llava-med job
output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-combined-finetune-freeze-viz-llava-med-again-new-dataset-gli_met-v11-0000
test_path=/local2/amvepa91/MedTrinity-25M/brats_gli_met_3d_vqa_subjTrue_train_updated_v11_seed0.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--model_name_or_path "$output_dir"/hf \
--output_dir $output_dir/eval_vqa2 \
--vqa_data_test_path $test_path \

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa_utils.py \
--output_dir $output_dir/eval_vqa2 \
--gt_file $test_path \