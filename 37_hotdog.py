import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体（Windows自带，必存在）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示方块问题

data_dir = r'D:\pycharm-code\data\hotdog'
#创建两个实例来分别读取训练和测试数据集中的所有图像文件。
#train_augs（训练增强）
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
#test_augs（测试预处理）
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

d2l.plt.show()
# 使用RGB通道的均值和标准差，以标准化每个通道。左边是RGB通道的mean，右边是RGB通道的std（因为imagenet做了这样的标准化，）
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# pretrained_net = torchvision.models.resnet18(pretrained=True)
# print(pretrained_net.fc)
#Linear(in_features=512, out_features=1000, bias=True)
finetune_net = torchvision.models.resnet18(pretrained=True)
#自行修改的输出层
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
#只对最后一层进行随机初始化
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率,
def train_fine_tuning(net, learning_rate, batch_size=32, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    #devices = d2l.try_all_gpus()
    devices = d2l.try_all_gpus() or [torch.device('cpu')]  # 有GPU用GPU，没有则用CPU
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

train_fine_tuning(finetune_net, 5e-5)