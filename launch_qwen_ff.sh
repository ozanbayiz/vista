#!/bin/bash
cd /home/ozanbayiz/idarve
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=6 python3 -m src.setup_datasets.ve_latent_dataset --model-name qwen3vl --dataset fairface --output data/fairface_qwen3vl.hdf5 --batch-size 16 --num-workers 4
