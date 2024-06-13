#!/bin/bash

# Array of seeds
seeds=(0 1 2 3 4)

# Configuration file path
config_file="./cfg/shac_alpha/ant.yaml"

# Device to use
device="cpu"

# Loop through each seed and run the Python script
for seed in "${seeds[@]}"
do
    echo "Running with seed $seed"
    python3 train_shac_alpha.py --cfg $config_file --device $device --seed $seed
done

