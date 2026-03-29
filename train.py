import torch
import numpy as np
import torch.nn.functional as F
import DataLoader
from test import test
import os
import parameters

def train(model, num_epoch):
    print("Start training")
    model.train()
    trainset, validset = DataLoader.Train_data_Loader()
    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    loss_history = [] # 记录每个 epoch 的平均训练损失
    success_history = [] # 记录每个 epoch 的训练成功率
    val_loss_history = []  # 记录每个 epoch 的平均验证损失
    val_success_history = [] # 记录每个 epoch 的验证成功率
    train_step_losses = []  # 记录每个训练step的损失值
    val_step_losses = []  # 记录每个验证step的损失值
    saveDir = params.model_save_dir
    os.makedirs(saveDir, exist_ok=True)

    global_step = 0  # 全局训练步数
    global_val_step = 0  # 全局验证步数

    best_val_loss = float('inf')  # 初始化最佳验证损失值
    best_model_path = None  # 初始化最佳模型路径

    for epoch in range(num_epoch):
        # init the parameters of trainset
        correct_preds_all = 0
        train_num_all = 0
        total_loss = 0

        # train
        for step, (images, labels) in enumerate(trainset):
            global_step += 1  # 更新全局训练步数
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            optimizer.zero_grad()
            preds = model(images) # 得到batchsize*clas(10)的张量
            loss = F.cross_entropy(preds, labels) 
            total_loss += loss.item() * batch_size
            train_step_losses.append(loss.item())  # 记录当前step的损失值
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs, label_preds = preds.max(dim=1)
                correct_preds = torch.sum(labels==label_preds)
                success_rate = correct_preds/batch_size
                correct_preds_all += correct_preds
                train_num_all += batch_size
                if step % 10 == 0:
                    print(f'epoch: {epoch} train step: {step} success rate: {success_rate:.3f}, loss: {loss.item():.3f}')

        avg_loss = total_loss / train_num_all
        success_rate_epoch = (correct_preds_all/train_num_all).item()
        print('epoch {} training finish! The success rate is {}, and the average loss is {}'.format(epoch,success_rate_epoch, avg_loss))
        success_history.append(success_rate_epoch)
        loss_history.append(avg_loss)   

        # validate after each training epoch
        val_correct = 0
        val_num_all = 0
        val_total_loss = 0
        with torch.no_grad():
            for val_step, (val_images, val_labels) in enumerate(validset):
                global_val_step += 1  # 更新全局验证步数
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_batch_size = val_labels.size(0)
                preds = model(val_images) # 得到batchsize*clas(10)的张量
                val_loss = F.cross_entropy(preds, val_labels) 
                val_total_loss += val_loss.item() * val_batch_size
                val_step_losses.append(val_loss.item())  # 记录当前验证step的损失值
                probs, val_label_preds = preds.max(dim=1)
                val_correct_preds = torch.sum(val_labels==val_label_preds)
                val_batch_acc = val_correct_preds.item() / val_batch_size
                val_correct += val_correct_preds
                val_num_all += val_batch_size
                if val_step % 10 == 0:
                    print(f'epoch: {epoch} valid step: {val_step} batch success rate: {val_batch_acc:.3f}, loss: {val_loss.item():.3f}')

        val_avg_loss = val_total_loss / val_num_all
        val_success_rate = (val_correct / val_num_all).item()
        print("epoch {} validation finish! The final success rate is {}, and the average loss is {}".format(epoch, val_success_rate, val_avg_loss))
        val_loss_history.append(val_avg_loss)
        val_success_history.append(val_success_rate)

        # Checkpoint
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_model_path = os.path.join(saveDir, f"best_model_lr_{params.learning_rate}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path} with validation loss {best_val_loss:.4f}")

    # patience=3, threshold=1e-3
    # if len(loss_history) >= patience + 1:
    #    # 检查最近 patience 轮的相邻损失差值
    #    recent_losses = loss_history[-patience-1:]  # 取最后 patience+1 个损失
    #    diffs = [abs(recent_losses[i+1] - recent_losses[i]) for i in range(len(recent_losses)-1)]
    #    if all(d < threshold for d in diffs):
    #        print(f'Early stopping triggered: loss change < {threshold} for {patience} consecutive epochs.')
    #        break
    
    # model.train()
    # 如果存在最佳模型路径，加载最佳模型权重
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model weights from {best_model_path}")
        
    return model, loss_history, success_history, val_loss_history, val_success_history, train_step_losses, val_step_losses



