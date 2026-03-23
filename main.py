from test import test
from train import train
from model import Model
import torch
import os
import numpy as np
from plot import plot_training_loss, plot_validation_loss, plot_success_rate, plot_validation_success_rate, plot_step_loss_curves
import parameters

model = Model()
params = parameters.get_parameters()
device = torch.device(params.device if torch.cuda.is_available() else "cpu")
model = model.to(device)

if params.device == 'cuda':
    model = torch.nn.DataParallel(model)


model, loss_history, success_history, val_loss_history, val_success_history, train_step_losses, val_step_losses = train(model, params.num_epoch)

# 保存step-loss数据到results/data
os.makedirs('results/data', exist_ok=True)
np.save(f'results/data/train_step_losses_lr_{params.learning_rate}.npy', train_step_losses)
np.save(f'results/data/val_step_losses_lr_{params.learning_rate}.npy', val_step_losses)
print(f"Step-loss data saved for learning rate {params.learning_rate}")

# 绘制step-loss曲线
plot_step_loss_curves(data_dir='results/data', save_path='results/plot/step_loss_curves.png')

# 保存训练数据到 results/data
os.makedirs('results/data', exist_ok=True)
np.save('results/data/loss_history.npy', loss_history)
np.save('results/data/success_history.npy', success_history)
np.save('results/data/val_loss_history.npy', val_loss_history)
np.save('results/data/val_success_history.npy', val_success_history)
print("Training data saved to results/data/")

# Visualize training results and save plots to results/plot
os.makedirs('results/plot', exist_ok=True)
plot_training_loss(loss_history, save_path='results/plot/training_loss.png')
plot_validation_loss(val_loss_history, save_path='results/plot/validation_loss.png')
plot_success_rate(success_history, save_path='results/plot/success_rate.png')
plot_validation_success_rate(val_success_history, save_path='results/plot/validation_success_rate.png')
