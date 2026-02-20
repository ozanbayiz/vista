#!/bin/bash
cd /home/ozanbayiz/idarve
export CUDA_VISIBLE_DEVICES=3
python3 -m src.vlm_generate \
  --hdf5 /scratch/current/ozanbayiz/fairface_paligemma2.hdf5 \
  --output-dir /scratch/current/ozanbayiz/results/causal_paligemma2 \
  --task caption --attribute gender --mode leace \
  --n-samples 500 --batch-size 4 --max-new-tokens 256
