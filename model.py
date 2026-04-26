import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
该模型对应的是3*224*224的输入图像，输出是10类的概率分布

'''
def _get_activation(name):
    name = name.lower()
    if name == 'relu':
        return F.relu
    if name in ('sigmoid', 'sigma'):
        return torch.sigmoid
    if name == 'tanh':
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


class Basic_block(nn.Module):
    def __init__(self, in_chan, out_chan, activation):
        super(Basic_block, self).__init__()
        if in_chan != out_chan:
            str = 2
        else:
            str = 1    

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=str, padding=1)        
        self.bn1 = nn.BatchNorm2d(num_features=out_chan) # 第一个conv后需要进行bn
        self.activation = activation
        self.conv2 = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_chan) # 第二个conv后需要进行bn
        self.ds = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_chan) # downsample后需要进行bn
    
    def downsample(self,x):
        if self.in_chan == self.out_chan:
            output = x
        elif self.in_chan != self.out_chan:
            output = self.ds(x)
            output = self.bn3(output)

        return output


    def forward(self,x):
        re = self.downsample(x)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.activation(x1+re)

        return x1
    


class Model(nn.Module):
    def __init__(self, activation='relu'):
        super(Model, self).__init__()
        # 卷积核：N*C*H*W -> 64*3*7*7
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.activation = _get_activation(activation)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Layer1 = self.Layer(64, 64)
        self.Layer2 = self.Layer(64, 128)
        self.Layer3 = self.Layer(128, 256)
        self.Layer4 = self.Layer(256, 512)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(512, 10)


    def forward(self,x):
        x = self.Layer0(x)

        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x # 返回的是batchsize*10的张量


    def Layer0(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x1 = self.maxpool1(x1)

        return x1
        

    def Layer(self, in_chan, out_chan):
        block_1 = Basic_block(in_chan, out_chan, self.activation)
        block_2 = Basic_block(out_chan, out_chan, self.activation)


        return nn.Sequential(block_1, block_2)


       
