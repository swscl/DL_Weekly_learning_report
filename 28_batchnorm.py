#batchnorm从零开始实现
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，只有一个样本，直接使用全局的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        #2 代表全连接层 (batch_size, feature)
        #4 代表2D卷积层 (batch_size, channels, height, width)
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)#按行求均值，就是每一列算出来一个均值
            var = ((X - mean) ** 2).mean(dim=0)#方差，方差和均值都是行向量
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。沿着通道维度求均值，通道的dim=1
            # 这里我们需要保持X的形状以便后面可以做广播运算
            #每一个样本打平高和宽，所有样本在该通道的均值和标准差
            #触发广播机制的情形
            # 1.tensor维度相等。
            # 2.tensor维度不等且其中一个维度为1。
            # 3.tensor维度不等且其中一个维度不存在。
            # 这里mean的维度为1，故即使维度不一样也可以触发广播机制
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            #计算后的 mean 形状会变成 (1, C, 1, 1)。
            '''
            keepdim=True 的作用如果不加这个参数，返回的均值形状会变成一维的 $(C,)$。使用了 keepdim=True 后，形状保持为 $(1, C, 1, 1)$。为什么要这样做？ 为了方便后续的广播机制 (Broadcasting)。在执行 $x - \mu$ 这一步时，$(1, C, 1, 1)$ 的均值可以直接与原图 $(N, C, H, W)$ 相减，程序会自动把这个均值“平铺”到每一个像素点上去。
            '''
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位,实现那个公式
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))#要拟合的方差：全1
        self.beta = nn.Parameter(torch.zeros(shape))#要拟合的均值：全0
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
'''
使用批量归一化的lenet'''

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))


lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()