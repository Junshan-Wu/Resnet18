import torch
import numpy as np
import torch.nn.functional as F
import DataLoader
from test import test
import os

def train(model, num_epoch):
    print("Start training")
    model.train()
    trainset, validset = DataLoader.Train_data_Loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters())
    loss_history = []
    success_history = []
    val_loss_history = []
    val_success_history = []
    saveDir = "./model_weights/"
    # 递归创建目录（如果不存在）
    os.makedirs(saveDir, exist_ok=True)
    for epoch in range(num_epoch):
        # init the parameters of trainset
        correct_preds_all = 0
        train_num_all = 0
        total_loss = 0

        # init the parameters of validset
        val_correct = 0
        val_num_all = 0
        val_total_loss = 0

        # train
        for step, (images, labels) in enumerate(trainset):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            optimizer.zero_grad()
            preds = model(images) # 得到batchsize*clas(10)的张量
            # cross_entropy=nll_loss+log_softmax，返回的是一个batch内所有样本的平均损失值，preds是logits，labels是类别标签（整数）
            loss = F.cross_entropy(preds, labels) 
            total_loss += loss.item() * batch_size
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs, label_preds = preds.max(dim=1)
                correct_preds = torch.sum(labels==label_preds)
                success_rate = correct_preds/batch_size
                correct_preds_all += correct_preds
                train_num_all += batch_size
                print('epoch: {} train step: {} success rate: {:.3}'.format(epoch, step, success_rate))

        avg_loss = total_loss / train_num_all
        success_rate_epoch = (correct_preds_all/train_num_all).item()
        print('epoch {} training finish! The success rate is {}, and the average loss is {}'.format(epoch,success_rate_epoch, avg_loss))
        success_history.append(success_rate_epoch)
        loss_history.append(avg_loss)   

        # valid
        with torch.no_grad():
            for val_step, (val_images, val_labels) in enumerate(validset):
                if epoch % 3 != 0:
                    break
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_batch_size = val_labels.size(0)
                preds = model(val_images) # 得到batchsize*clas(10)的张量
                val_loss = F.cross_entropy(preds, val_labels) 
                val_total_loss += val_loss.item() * val_batch_size
                probs, val_label_preds = preds.max(dim=1)
                val_correct_preds = torch.sum(val_labels==val_label_preds)
                val_batch_acc = val_correct_preds.item() / val_batch_size
                val_correct += val_correct_preds
                val_num_all += val_batch_size
                print("epoch: {} valid step: {}  batch success rate: {:.3}".format(epoch, val_step, val_batch_acc))

        if epoch % 3 == 0:
            val_avg_loss = val_total_loss / val_num_all
            val_success_rate = (val_correct/val_num_all).item()
            print("epoch {} validation finish! The final success rate is {}, and the average loss is {}".format(epoch, val_success_rate, val_avg_loss))
            val_loss_history.append(val_avg_loss)
            val_success_history.append(val_success_rate)



    model_path = os.path.join(saveDir, f"final_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # patience=3, threshold=1e-3
    # if len(loss_history) >= patience + 1:
    #    # 检查最近 patience 轮的相邻损失差值
    #    recent_losses = loss_history[-patience-1:]  # 取最后 patience+1 个损失
    #    diffs = [abs(recent_losses[i+1] - recent_losses[i]) for i in range(len(recent_losses)-1)]
    #    if all(d < threshold for d in diffs):
    #        print(f'Early stopping triggered: loss change < {threshold} for {patience} consecutive epochs.')
    #        break
    
    # model.train()

    return model, loss_history, success_history, val_loss_history, val_success_history
            


