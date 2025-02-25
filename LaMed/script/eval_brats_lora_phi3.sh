#!/bin/bash

output_dir=./LaMed/output/LaMed-Phi3-4B-finetune-freeze-brats-pretrained-ped-viz-0000
test_path=/local2/amvepa91/MedTrinity-25M/brats_ped_3d_vqa_subjTrue_test_v1.json

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--model_name_or_path $output_dir/hf \
--vqa_data_test_path $test_path \
--output_dir $output_dir/eval_vqa/
