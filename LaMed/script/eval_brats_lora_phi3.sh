#!/bin/bash

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python Bench/eval/eval_vqa.py \
--model_name_or_path LaMed/output/LaMed-Phi3-4B-finetune-0000/hf \
--vqa_data_test_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v2.json \
