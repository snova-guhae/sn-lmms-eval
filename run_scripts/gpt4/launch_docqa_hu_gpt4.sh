source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh
conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval
export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH

python3 -m lmms_eval --model gpt4v \
  --model_args model_version="gpt-4o",modality=image \
  --tasks docqa_hu_syn,docqa_hu_ifeval \
  --limit 2000 \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix gpt4o_hu \
  --output_path ./logs/ 
