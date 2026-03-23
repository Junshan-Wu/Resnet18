import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
<<<<<<< HEAD
import parameters
=======
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8

def Train_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = parameters.get_parameters().data_dir

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    
<<<<<<< HEAD
    valid_size = parameters.get_parameters().valid_size
=======
    valid_size = 0.2
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

<<<<<<< HEAD
=======
    # 无放回地按照给定的索引列表采样样本元素
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = Data.DataLoader(
            dataset=trainset,  
<<<<<<< HEAD
            batch_size=parameters.get_parameters().batch_size, 
            sampler=train_sampler,  
            num_workers=parameters.get_parameters().num_workers,
=======
            batch_size=64, 
            sampler=train_sampler,  
            num_workers=2,  # 加载数据所开启的进程数量
>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8
        )
    
    valid_loader = Data.DataLoader(
            dataset=trainset,  
<<<<<<< HEAD
            batch_size=parameters.get_parameters().batch_size, 
            sampler=valid_sampler,  
            num_workers=parameters.get_parameters().num_workers,
        )
    
    return train_loader, valid_loader
=======
            batch_size=64, 
            sampler=valid_sampler,  
            num_workers=2,  # 加载数据所开启的进程数量
        )
    
    return train_loader, valid_loader

>>>>>>> d8ed29394c46c98cace1aac5dca1649459e99ab8


def Test_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = parameters.get_parameters().data_dir

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)

    test_loader = Data.DataLoader(
            dataset=testset,  
            batch_size=parameters.get_parameters().batch_size, 
            shuffle=True,  
            num_workers=parameters.get_parameters().num_workers,
        )
 
    return test_loader