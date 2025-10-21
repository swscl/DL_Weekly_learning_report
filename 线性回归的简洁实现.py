import numpy as np
import torch
from torch.utils import data##导入处理数据的模块
from d2l import torch as d2l

'''生成数据集'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

##synthetic_data是人工数据合成的函数
'''读取数据集，每次找一个batchsize'''
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

'''定义模型'''
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
##等价于手动实现w和b。
'''定义损失函数：均方误差'''
loss = nn.MSELoss()
'''定义优化算法：SGD'''
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

'''训练部分:与手动部分几乎相同'''
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

'''比较参数'''
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)