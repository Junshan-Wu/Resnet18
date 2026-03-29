import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import parameters

def Train_data_Loader():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    data_dir = parameters.get_parameters().data_dir

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    
    valid_size = parameters.get_parameters().valid_size
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=parameters.get_parameters().batch_size, 
            sampler=train_sampler,  
            num_workers=parameters.get_parameters().num_workers,
        )
    
    valid_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=parameters.get_parameters().batch_size, 
            sampler=valid_sampler,  
            num_workers=parameters.get_parameters().num_workers,
        )
    
    return train_loader, valid_loader


def Test_data_Loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

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