##从零开始实现池化层
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(x,pool_size,mode='max'):
    p_h,p_w = pool_size
    y=torch.zeros((x.shape[0]-p_h+1,x.shape[1]-p_w+1))#为什么要提前定义好输出的大小呢
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if mode =='max':
                y[i,j]=x[i:i+p_h,j:j+p_w].max()
            elif mode =='avg':
                y[i, j] = x[i:i + p_h, j:j + p_w].mean()
    return y

x=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
# print(x)
#
# pool2d = nn.MaxPool2d(3)#表示窗口的大小是3,默认的stride和窗口的大小是相同的，就是两个窗口是没有重叠的。
# y=pool2d(x)
# print('y:',y)

x=torch.cat((x,x+1),1)#在1这个维度上扩充
print(x)
pool2d  = nn.MaxPool2d(3,padding=1,stride=2)
y=pool2d(x)
print(y)