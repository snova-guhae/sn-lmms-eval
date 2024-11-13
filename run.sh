#!/bin/bash

cd /import/ml-sc-scratch1/matte/sn-lmms-eval
source activate /import/snvm-sc-scratch2/matte/miniconda3/envs/chart_testing

# Set HF_HOME cache directory
export HF_HOME=/import/snvm-sc-scratch1/matte/cache

# Run the command using accelerate
/import/snvm-sc-scratch2/matte/miniconda3/envs/chart_testing/bin/accelerate launch --num_processes 1 -m lmms_eval \
  --model mllama \
  --model_args pretrained="/import/ml-sc-scratch5/viekashv/llama_finetune_output/",device_map="auto" \
  --num_fewshot 0 \
  --log_samples \
  --batch_size 1 \
  --output_path /import/ml-sc-scratch2/matte/llava_logs \
  --tasks adi-eval \
  --limit 10
