#!/bin/bash

# Activate Python environment
source ./venv/Scripts/activate

# Define hyperparameter values
learning_rates=( 5e-4 7e-4 8e-4 )
batch_sizes=(1024 850)
schedulers=(cosine)
cutouts=(0 1)
valid_sizes=(0 0.2)
warmup_epochs_list=(3 5 0)

# Iterate over all combinations of hyperparameters
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for scheduler in "${schedulers[@]}"; do
      for cutout in "${cutouts[@]}"; do
        for valid_size in "${valid_sizes[@]}"; do
          for warmup_epochs in "${warmup_epochs_list[@]}"; do
            echo "Running experiment with lr=$lr, batch_size=$bs, scheduler=$scheduler, cutout=$cutout, valid_size=$valid_size, warmup_epochs=$warmup_epochs"
            python ./main.py \
              --learning_rate $lr \
              --batch_size $bs \
              --lr_scheduler $scheduler \
              --use_cutout $cutout \
              --valid_size $valid_size \
              --warmup_epochs $warmup_epochs
          done
        done
      done
    done
  done
done

echo "All experiments completed."