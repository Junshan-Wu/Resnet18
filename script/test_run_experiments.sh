#!/bin/bash

# Activate Python environment
source ../venv/Scripts/activate

# Define hyperparameter values for testing
learning_rates=(1e-4 5e-5)
batch_sizes=(1024)
schedulers=(cosine)
cutouts=(0)

# Iterate over test combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for scheduler in "${schedulers[@]}"; do
      for cutout in "${cutouts[@]}"; do
        echo "Running test experiment with lr=$lr, batch_size=$bs, scheduler=$scheduler, cutout=$cutout"
        python ../main.py \
          --learning_rate $lr \
          --batch_size $bs \
          --lr_scheduler $scheduler \
          --use_cutout $cutout
      done
    done
  done
done

echo "Test experiments completed."