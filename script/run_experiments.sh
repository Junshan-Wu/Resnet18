#!/bin/bash

# Activate Python environment
source ../venv/Scripts/activate

# Define hyperparameter values
learning_rates=(1e-4 5e-5 1e-5 1e-6)
batch_sizes=(1024 2048 2500)
schedulers=(cosine step exponential)
cutouts=(0 1)

# Iterate over all combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for scheduler in "${schedulers[@]}"; do
      for cutout in "${cutouts[@]}"; do
        echo "Running experiment with lr=$lr, batch_size=$bs, scheduler=$scheduler, cutout=$cutout"
        python ../main.py \
          --learning_rate $lr \
          --batch_size $bs \
          --lr_scheduler $scheduler \
          --use_cutout $cutout
      done
    done
  done
done

echo "All experiments completed."