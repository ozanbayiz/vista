#!/bin/bash
cd /home/ozanbayiz/idarve
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=4 python3 -m src.setup_datasets.ve_latent_dataset --model-name dinov3 --dataset imagenet --output data/imagenet_dinov3.hdf5 --batch-size 64 --num-workers 8 --imagenet-path /tmp/imagenet_val
