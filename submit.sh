sngpu --jobname lllama_sambajudge \
  --partition gpuonly \
  --image nvcr.io/nvidia/pytorch:22.10-py3 \
  --cpu 1 \
  --mem 65536 \
  --gpu 1 \
  --wrap "/import/ml-sc-scratch1/matte/sn-lmms-eval/run.sh"