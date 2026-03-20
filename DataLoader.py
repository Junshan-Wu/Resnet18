import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

def Train_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = './data'

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)

    train_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=64, 
            shuffle=False,  # 不对数据集重新排序
            num_workers=2,  # 加载数据所开启的进程数量
        )
    
    return train_loader



def Test_data_Loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=224)])

    data_dir = './data'

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)

    test_loader = Data.DataLoader(
            dataset=testset,  
            batch_size=1, 
            shuffle=True,  
            num_workers=2,  
        )
 
    return test_loader