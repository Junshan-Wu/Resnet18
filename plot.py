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
    epochs = range(1, len(val_loss_history) + 1)
    
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
    epochs = range(1, len(val_success_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_success_history, marker='o', linestyle='-', color='m', label='Success Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate of Validation Set')
    plt.title('Success Rate vs. Epoch (Validation Set)')
    plt.grid(True)
    plt.legend()

    _save_and_show(save_path, show)

def plot_learning_rate_comparison(file_paths, save_path=None, show=True):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))  # 使用颜色映射生成不同颜色

    # 根据文件名判断数据集类型
    if file_paths and "train_step_losses_lr_" in file_paths[0]:
        dataset_type = "(trainset)"
    elif file_paths and "val_step_losses_lr_" in file_paths[0]:
        dataset_type = "(validset)"
    else:
        dataset_type = ""

    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过...")
            continue

        # 从文件名中提取学习率
        try:
            learning_rate = float(file_path.split('_lr_')[1].split('.npy')[0])
        except (IndexError, ValueError):
            print(f"无法从文件名中提取学习率: {file_path}")
            learning_rate = f"未知_{i}"

        # 加载数据并绘制曲线
        step_losses = np.load(file_path)
        steps = range(1, len(step_losses) + 1)
        plt.plot(steps, step_losses, label=f"LR={learning_rate}", color=colors[i])

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Step for Different Learning Rates {dataset_type}")
    plt.legend()
    plt.grid(True)

    _save_and_show(save_path, show)

def _save_and_show(save_path, show):
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()