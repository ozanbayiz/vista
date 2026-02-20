#!/bin/bash
cd /home/ozanbayiz/idarve
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=7 python3 -m src.setup_datasets.ve_latent_dataset --model-name internvl35 --dataset fairface --output data/fairface_internvl35.hdf5 --batch-size 32 --num-workers 4
