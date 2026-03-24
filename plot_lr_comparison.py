import os
from plot import plot_learning_rate_comparison

# 定义存储 step-loss 数据的目录
data_dir = "results/data/loss"

# 获取所有以 train_step_losses_lr_ 开头的 .npy 文件; 1->train step-loss数据, 2->val step-loss数据
file_paths_1 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("train_step_losses_lr_") and f.endswith(".npy")]
file_paths_2 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("val_step_losses_lr_") and f.endswith(".npy")]
# 调用绘图函数
plot_learning_rate_comparison(file_paths_1, save_path="results/plot/lr_comparison_train.png", show=True)
plot_learning_rate_comparison(file_paths_2, save_path="results/plot/lr_comparison_val.png", show=True)
