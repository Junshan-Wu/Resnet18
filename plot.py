import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_overview(success_history, val_success_history, loss_history, val_loss_history, test_accuracy=None, save_path=None, show=True):
    epochs_acc = range(1, len(success_history) + 1)
    epochs_loss = range(1, len(loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: accuracy curves for train and valid sets
    axes[0].plot(epochs_acc, success_history, marker='o', linestyle='-', color='g', label='Train Accuracy')
    axes[0].plot(epochs_acc, val_success_history, marker='o', linestyle='-', color='m', label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs. Epoch')
    axes[0].grid(True)
    axes[0].legend()

    # Right subplot: loss curves for train and valid sets
    axes[1].plot(epochs_loss, loss_history, marker='o', linestyle='-', color='b', label='Train Loss')
    axes[1].plot(epochs_loss, val_loss_history, marker='o', linestyle='-', color='r', label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss vs. Epoch')
    axes[1].grid(True)
    axes[1].legend()

    if test_accuracy is not None:
        fig.suptitle(f"Training Overview | Test Accuracy: {test_accuracy:.3f}", fontsize=14)
    else:
        fig.suptitle("Training Overview", fontsize=14)

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    _save_and_show(save_path, show)


def plot_training_overview_train_only(success_history, loss_history, test_accuracy=None,save_path=None, show=True):
    epochs_acc = range(1, len(success_history) + 1)
    epochs_loss = range(1, len(loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: train accuracy curve
    axes[0].plot(epochs_acc, success_history, marker='o', linestyle='-', color='g', label='Train Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Train Accuracy vs. Epoch')
    axes[0].grid(True)
    axes[0].legend()

    # Right subplot: train loss curve
    axes[1].plot(epochs_loss, loss_history, marker='o', linestyle='-', color='b', label='Train Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Train Loss vs. Epoch')
    axes[1].grid(True)
    axes[1].legend()

    if test_accuracy is not None:
        fig.suptitle(f"Training Overview | Test Accuracy: {test_accuracy:.3f}", fontsize=14)
    else:
        fig.suptitle("Training Overview", fontsize=14)
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
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