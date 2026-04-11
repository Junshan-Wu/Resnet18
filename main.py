from test import test
from train import train
from train_full import train_full
from model import Model
from model_32 import Model_32
import torch
import os
import numpy as np
from plot import plot_training_overview, plot_training_overview_train_only, plot_learning_rate_comparison
import parameters
from utils import set_seed

# Set the random seed for reproducibility
set_seed(3407)

# model = Model()
model = Model_32()
params = parameters.get_parameters()
device = torch.device(params.device if torch.cuda.is_available() else "cpu")
model = model.to(device)

if params.device == 'cuda':
    model = torch.nn.DataParallel(model)


def _name_safe(value):
    return str(value).replace('.', 'p').replace('-', 'm')

if params.valid_size != 0:
    model, loss_history, success_history, val_loss_history, val_success_history, train_step_losses, val_step_losses = train(model, params.num_epoch)
    test_accuracy, test_loss = test(model)

    # Visualize training results and save plots to results/plot
    os.makedirs('results/plot', exist_ok=True)

    plot_filename = (
        f"lr_{_name_safe(params.learning_rate)}_"
        f"bs_{params.batch_size}_"
        f"sch_{params.lr_scheduler}_"
        f"cutout_{params.use_cutout}_"
        f"warmup_{params.warmup_epochs}_"
        f"valid_{_name_safe(params.valid_size)}.png"
    )
    plot_save_path = os.path.join('results/plot', plot_filename)

    plot_training_overview(
        success_history,
        val_success_history,
        loss_history,
        val_loss_history,
        test_accuracy=test_accuracy,
        save_path=plot_save_path
    )
else:
    model, loss_history, success_history, train_step_losses = train_full(model, params.num_epoch)
    test_accuracy, test_loss = test(model)
    os.makedirs('results/plot', exist_ok=True)
    plot_filename = (
        f"fulltrain_lr_{_name_safe(params.learning_rate)}_"
        f"bs_{params.batch_size}_"
        f"sch_{params.lr_scheduler}_"
        f"cutout_{params.use_cutout}_"
        f"warmup_{params.warmup_epochs}_"
        f"valid_{_name_safe(params.valid_size)}.png"
    )
    plot_save_path = os.path.join('results/plot', plot_filename)

    plot_training_overview_train_only(
        success_history,
        loss_history,
        test_accuracy=test_accuracy,
        save_path=plot_save_path
    )

# 保存step-loss数据到results/data/loss目录
# os.makedirs('results/data/loss', exist_ok=True)
# np.save(f'results/data/loss/train_step_losses_lr_{params.learning_rate}.npy', train_step_losses)
# np.save(f'results/data/loss/val_step_losses_lr_{params.learning_rate}.npy', val_step_losses)
# print(f"Step-loss data saved for learning rate {params.learning_rate}")

# 保存训练数据到 results/data
# os.makedirs('results/data/success', exist_ok=True)
# np.save('results/data/loss/loss_history.npy', loss_history)
# np.save('results/data/success/success_history.npy', success_history)
# np.save('results/data/loss/val_loss_history.npy', val_loss_history)
# np.save('results/data/success/val_success_history.npy', val_success_history)
# print("Training data saved to results/data/")

