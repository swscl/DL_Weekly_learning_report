import torch
from torch import nn
from d2l import torch as d2l
from data_loader import load_fashion_mnist  # 导入封装的函数
from softmax_train import train_ch3


if __name__ == '__main__':
 batch_size = 256
train_iter, test_iter = load_fashion_mnist(batch_size=batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256#隐藏层是256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

'''实现relu'''
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

'''模型'''
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)


train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)




