import torch
import numpy as np
import torch.nn.functional as F
import DataLoader
from test import test

def train(model, max_epoch, patience=3, threshold=1e-3):
    print("Start training")
    model.train()
    trainset = DataLoader.Train_data_Loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters())
    loss_history = []
    for epoch in range(max_epoch):
        correct_preds_all = 0
        train_num_all = 0
        for step, (images, labels) in enumerate(trainset):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(images) # 得到batchsize*clas(10)的张量
            # cross_entropy=nll_loss+log_softmax，返回的是一个batch内所有样本的平均损失值，preds是logits，labels是类别标签（整数）
            loss = F.cross_entropy(preds, labels) 
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_size = labels.size(0)
                probs, label_preds = preds.max(dim=1)
                correct_preds = torch.sum(labels==label_preds)
                success_rate = correct_preds/batch_size
                correct_preds_all += correct_preds
                train_num_all += batch_size
                print('epoch: {} step: {} success rate: {:.3}'.format(epoch, step, success_rate))

        success_rate_epoch = correct_preds_all/train_num_all
        print('epoch {} finish! The success rate is {}'.format(epoch,success_rate_epoch))

        avg_loss = test(model)           
        loss_history.append(avg_loss)    

        if len(loss_history) >= patience + 1:
           # 检查最近 patience 轮的相邻损失差值
           recent_losses = loss_history[-patience-1:]  # 取最后 patience+1 个损失
           diffs = [abs(recent_losses[i+1] - recent_losses[i]) for i in range(len(recent_losses)-1)]
           if all(d < threshold for d in diffs):
               print(f'Early stopping triggered: loss change < {threshold} for {patience} consecutive epochs.')
               break
        
        model.train()


    return model, loss_history
            


