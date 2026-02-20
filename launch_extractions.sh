#!/bin/bash
# Launch PaliGemma2 ImageNet extraction on GPU 0
cd /home/ozanbayiz/idarve

CUDA_VISIBLE_DEVICES=0 nohup python3 -m src.setup_datasets.ve_latent_dataset \
    --model-name paligemma2 \
    --dataset imagenet \
    --output data/imagenet_paligemma2.hdf5 \
    --batch-size 64 \
    --num-workers 8 \
    --imagenet-path /tmp/imagenet_val \
    > /tmp/paligemma2_imagenet.log 2>&1 &
echo "PaliGemma2 ImageNet PID: $!"
