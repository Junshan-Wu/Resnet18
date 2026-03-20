import torch
import numpy as np
import torch.nn.functional as F
import DataLoader


def test(model):
    print("Start testing")
    model.eval()
    testset = DataLoader.Test_data_Loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_correct = 0
    test_num_all = 0
    total_loss = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(testset):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            preds = model(images) # 得到batchsize*clas(10)的张量
            loss = F.cross_entropy(preds, labels) 
            total_loss += loss.item() * batch_size
            probs, label_preds = preds.max(dim=1)
            correct_preds = torch.sum(labels==label_preds)
            batch_acc = correct_preds.item() / batch_size
            test_correct += correct_preds
            test_num_all += batch_size
            print("step: {}  batch success rate: {:.3}".format(step, batch_acc))

    avg_loss = total_loss / test_num_all
    success_rate = test_correct/test_num_all
    print("Finish! The final success rate is", success_rate) 
    print("Finish! The final average loss is", avg_loss)
    return avg_loss
        
       
