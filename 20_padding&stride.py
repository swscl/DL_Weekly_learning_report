import torch
from torch import nn

def comp_conv2d(conv2d,x):
    x=x.reshape((1,1)+x.shape)#添加上传统的conv2d需要的前两个维度，输入和输出的通道数
    y=conv2d(x)
    return y.reshape(y.shape[2:])#除去前两个维度

conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
x=torch.rand(8,8)
y=comp_conv2d(conv2d,x).shape

print(y)