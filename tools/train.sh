#!/bin/bash

python train.py \
    --wandb_project "rai-audit" \
    --lr 5e-5 \
    --max_steps 10000 \
    --batch_size 128 \
    --log_interval 50 \
    --val_interval 500
