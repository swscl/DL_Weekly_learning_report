import torch
from torch import nn
from d2l import torch as d2l
#将数据拿到迭代器中
batch_size = 256


train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''
初始化模型参数
'''
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten）：展开成0维的向量，
# 来调整网络输入的形状
# nn.Linear是一个线性层，输入是784，输出是10
# Sequential是一个模型构造器：这是 PyTorch 中的一个层容器，
# 用于按顺序组合多个神经网络层。它会将传入的层按顺序串联起来，
# 前一层的输出会自动作为后一层的输入，简化模型搭建流程。
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction='none')
'''
优化算法
'''
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

'''
训练：
'''
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)














