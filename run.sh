#!/bin/bash

# Set default values
dataset=${1:-movielens}
model=${2:-NeuMF-end}
mode=${3:-flip}
drop_rate=${4:-0.2}
num_gradual=${5:-30000}
test_batch_size=${6:-2048}

python -u main.py --dataset=$dataset --model=$model --mode=$mode --drop_rate=$drop_rate --num_gradual=$num_gradual --test_batch_size=$test_batch_size > log/$dataset/$model-$mode_$drop_rate-$num_gradual.log 2>&1 &