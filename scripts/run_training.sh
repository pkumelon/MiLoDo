#!/bin/bash

# Check if config file is provided
if [ $# -eq 1 ]; then
    CONFIG_FILE=$1
    python train.py --stages_config "$CONFIG_FILE"
else
    # Run with default multi-stage configuration
    python train.py --stages_config "./scripts/stages_config.json" \
        --batch_size 2 \
        --agents 10 \
        --optimizee lasso \
        --features 300 \
        --l1_weight 0.1 \
        --noise 0.1 \
        --use_bias \
        --lr_init 0.03 \
        --optimizer adam \
        --weight_decay 1e-4 \
        --seed 42 \
        --save_path "./model/multi_stage_model.pth" \
        --save_intermediate
fi