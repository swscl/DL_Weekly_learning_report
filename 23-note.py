import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载MNIST数据集
train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2), nn.ReLU(),
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(6,16,kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2), nn.Flatten(),
            nn.Linear(16*5*5,120), nn.ReLU(),
            nn.Linear(120,84), nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

net = Net()
net = net.cpu()

# 打印数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据迭代器
train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True)

# 损失函数与优化器
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.cpu()
lr = 0.05
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练参数初始化
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数
epoch = 20  # 训练轮次
writer = SummaryWriter("./logs")  # tensorboard日志

# 训练循环
for i in range(epoch):
    print("-----第{}轮训练开始----".format(i+1))
    net.train()  # 训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cpu()
        targets = targets.cpu()
        outputs = net(imgs)
        loss = loss_fun(outputs, targets)
        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 200 == 0:
            print("训练次数：{}，loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试阶段
    net.eval()  # 评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cpu()
            targets = targets.cpu()
            outputs = net(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 记录测试集指标
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuary", total_accuracy.item() / test_data_size, total_test_step)
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy.item() / test_data_size))
    total_test_step += 1

writer.close()