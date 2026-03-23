import torch
import numpy as np
import torch.nn.functional as F
import DataLoader
import parameters


def test(model):
    print("Start testing")
    model.eval()
    testset = DataLoader.Test_data_Loader()
    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"
    test_correct = 0
    test_num_all = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(testset):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            preds = model(images) # 得到batchsize*clas(10)的张量
<<<<<<< HEAD
            loss = F.cross_entropy(preds, labels)
            total_loss += loss.item() * batch_size
=======
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8
            probs, label_preds = preds.max(dim=1)
            correct_preds = torch.sum(labels==label_preds)
            batch_acc = correct_preds.item() / batch_size
            test_correct += correct_preds
            test_num_all += batch_size
            if step % 10 == 0:
                print(f"step: {step} batch success rate: {batch_acc:.3f}, loss: {loss.item():.3f}")

<<<<<<< HEAD
    avg_loss = total_loss / test_num_all
    success_rate = (test_correct/test_num_all).item()
    print(f"Finish! The final success rate is {success_rate:.3f}, and the average loss is {avg_loss:.3f}")




=======
    success_rate = (test_correct/test_num_all).item()
    print("Finish! The final success rate is", success_rate) 


        
       
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8
