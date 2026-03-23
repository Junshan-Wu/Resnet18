import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def Train_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = './data'

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    
    valid_size = 0.2
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=64, 
            sampler=train_sampler,  
            num_workers=2,  # 加载数据所开启的进程数量
        )
    
    valid_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=64, 
            sampler=valid_sampler,  
            num_workers=2,  # 加载数据所开启的进程数量
        )
    
    return train_loader, valid_loader



def Test_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = './data'

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)

    test_loader = Data.DataLoader(
            dataset=testset,  
            batch_size=64, 
            shuffle=True,  
            num_workers=2,  
        )
 
    return test_loader