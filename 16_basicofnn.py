
import torch
from torch import nn
from torch.nn import functional as F

# #回顾多层感知机
# #使用Sequential定义了一个module
# net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
#
# X = torch.rand(2,20)
# Y =net(X)

#自定义一个块
# class MLP (nn.Module):
#     #用MLP来继承Module,就可以在当前的类使用继承类的一些好用的函数
#     #所有的块都有两个函数，一个是init函数，在这里指定需要哪些类和参数
#     def __init__(self):
#         super().__init__()#这一步保存继承类的init
#         self.hidden = nn.Linear(20,256)
#         self.out = nn.Linear(256,10)
#
#     #定义前向函数
#     def forward(self,x):
#         return self.out(F.relu(self.hidden(x)))
#
# #使用这个模块
# net = MLP()
# X = torch.rand(2,20)
# y = net(X)
# print(y)

class mysequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    def forward(self,x):
        for block in self._modules.values():
            x = block(x)
        return x
net = mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
x=torch.rand(2,20)
net(x)
