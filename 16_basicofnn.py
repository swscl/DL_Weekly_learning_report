
import torch
from torch import nn
from torch.nn import functional as F

# 回顾多层感知机
# 使用Sequential定义了一个module
# net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
#
# X = torch.rand(2,4)
# Y =net(X)

# def init_normal(m):  # 参数初始化成正态分布 ，m是传入的module
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, mean=0, std=0.01)
#     nn.init.zeros_(m.bias)
#
# #net.apply(init_normal)
# def init_constant(m):
#     if type(m) == nn.Linear:
#         nn.init.constant_(m.weight, 1)
#         nn.init.zeros_(m.bias)
# def xavier(m):
# 	if type(m) == nn.Linear:
# 		nn.init.xavier_uniform_(m.weight)

# def my_init(m):
#     if type(m) == nn.Linear:
#         print(
#             "原本：",
#            [(name,param) for name,param in m.named_parameters()][0]
#         )
#         nn.init.uniform_(m.weight,-10,10)
#         print(
#             "第一次：",
#            [(name,param) for name,param in m.named_parameters()][0]
#         )
#         m.weight.data *=( m.weight.data.abs()>5)
#         print(
#             "第二次：",
#            [(name,param) for name,param in m.named_parameters()][0]
#         )
#
# net.apply(my_init)
#print(net[0].weight)
#print(net[0].bias)
# print(net[0].weight)
# print(net[0].bias)
# print(net[0].state_dict())
# print(net[2].state_dict())#输出第三层的状态
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].weight)
# for name ,pram in net.named_parameters():
# 	print((name,pram.shape))
# print(net.state_dict()['2.bias'].data)

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

# class mysequential(nn.Module):
#     def __init__(self,*args):
#         super().__init__()
#         for block in args:
#             self._modules[block] = block
#     def forward(self,x):
#         for block in self._modules.values():
#             x = block(x)
#         return x
# net = mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
# x=torch.rand(2,20)
# net(x)
# def block1():
#     return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
#
# def block2():
#     net=nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block{i}',block1())
#     return net
# rgnet = nn.Sequential(block2(),nn.Linear(4,1))
# x = torch.rand(2,4)
# rgnet(x)
# print(rgnet)

# shared =nn.Linear(8,8)
# net = nn.Sequential(nn.Linear(8,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
# x=torch.rand(8,8)
# net(x)
#
# print(net[2].state_dict()['weight'][0])
# print(net[4].state_dict()['weight'][0])

# class CenteredLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self,x):
#         return x-x.mean()
# lay = CenteredLayer()
# #print(lay(torch.FloatTensor([1,2,3,4,5])))
# x = torch.FloatTensor([1, 2, 3, 4, 5])
# net = nn.Sequential(nn.Linear(5,10),lay)
# print(net(x))

# class mylinear(nn.Module):
#     def __init__(self,in_units,units):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(in_units,units))
#         self.bias = nn.Parameter(torch.randn(units,))
# #将自定义的参数作为实例传入Parameter后，这个类会给自定义的参数加上名字和梯度信息
#     def forward(self,x):
#         linear = torch.matmul(x,self.weight)+self.bias
#         return F.relu(linear)
#
# dense= mylinear(5,3)
# print(dense.weight)
# y=dense(torch.rand(1,5))
# print(y)

x=torch.randn(2,20)
x1=torch.randn(size=(2,20))

print(x)
print(x1)