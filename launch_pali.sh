#!/bin/bash
cd /home/ozanbayiz/idarve
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=0 python3 -m src.setup_datasets.ve_latent_dataset --model-name paligemma2 --dataset imagenet --output data/imagenet_paligemma2.hdf5 --batch-size 64 --num-workers 8 --imagenet-path /tmp/imagenet_val
