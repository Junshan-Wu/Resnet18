import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_loss(loss_history, save_path=None, show=True):
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='b', label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss of Training Set')
    plt.title('Average Loss vs. Epoch (Training Set)')
    plt.grid(True)
    plt.legend()

    _save_and_show(save_path, show)

def plot_validation_loss(val_loss_history, save_path=None, show=True):
    epochs = [3 * (i + 1) for i in range(len(val_loss_history))]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_loss_history, marker='o', linestyle='-', color='r', label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss of Validation Set')
    plt.title('Average Loss vs. Epoch (Validation Set)')
    plt.grid(True)
    plt.legend()

    _save_and_show(save_path, show)

def plot_success_rate(success_history, save_path=None, show=True):
    epochs = range(1, len(success_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, success_history, marker='o', linestyle='-', color='g', label='Success Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate of Training Set')
    plt.title('Success Rate vs. Epoch (Training Set)')
    plt.grid(True)
    plt.legend()

    _save_and_show(save_path, show)

def plot_validation_success_rate(val_success_history, save_path=None, show=True):
    epochs = [3 * (i + 1) for i in range(len(val_success_history))]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_success_history, marker='o', linestyle='-', color='m', label='Success Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate of Validation Set')
    plt.title('Success Rate vs. Epoch (Validation Set)')
    plt.grid(True)
    plt.legend()

    _save_and_show(save_path, show)


def _save_and_show(save_path, show):
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()