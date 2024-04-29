#!/bin/bash

# "white_male_conservative"
# "white_male_liberal"
# "white_female_liberal"
# "black_female_moderate_liberal"
# "white_female_conservative"
# "all"

subset="all"

python train.py \
    --wandb_project "rai-audit" \
    --lr 2e-5 \
    --max_steps 450 \
    --batch_size 64 \
    --log_interval 50 \
    --model_save_dir "./models/${subset}" \
    --seed 42 \
    --subset $subset