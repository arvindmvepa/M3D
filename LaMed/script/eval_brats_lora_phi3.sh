#!/bin/bash

output_dir=./LaMed/output/LaMed-Phi3-4B-multimodal-finetune-0000
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python Bench/eval/eval_vqa.py \
--model_name_or_path "$output_dir"/hf \
--vqa_data_test_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json \
--output_dir $output_dir/eval_vqa/