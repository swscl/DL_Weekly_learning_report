import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import torchvision.datasets
from torchvision import datasets
from torch.utils.data import DataLoader
class Reshape(torch.nn.Module):
    '''给输入的x进行处理，-1表示批量数不变，通道数为1，大小为28*28'''
    def forward(self,x):
        return x.view(-1,1,28,28)

net =torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1,6,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Conv2d(6,16,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Flatten(),

    nn.Linear(16*6*6,120),#16个通道，图片大小6*6
    nn.ReLU(),
    nn.Linear(120,84),
    nn.ReLU(),
    nn.Linear(84,10)
)

#test
x=torch.rand((1,1,28,28),dtype=torch.float32)
for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,"输出的大小：\t",x.shape)


batch_size=256
#封装好的加载数据和迭代器
#train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
#这里只是加载数据
train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
#创建迭代器
train_iter = DataLoader(train_data, batch_size=256, shuffle=True)
test_iter = DataLoader(test_data, batch_size=256, shuffle=True)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    #配合sigmoid:
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         nn.init.xavier_uniform_(m.weight)

    #配合relu:
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # He初始化核心参数：nonlinearity='relu'
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            # 偏置初始化（可选，提升稳定性）
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
lr, num_epochs = 0.003, 15
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()