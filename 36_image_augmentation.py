import torch
import torchvision
from torch import nn
from d2l import torch as d2l

def load_cifar10(is_train, augs, batch_size):
    '''
    把 “原始 CIFAR10 数据”→“经过增强 / 转换的 Tensor 数据”→“按批次迭代的格式”，适配模型训练。
    '''
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

# 训练集的数据增强策略train_augs
train_augs = torchvision.transforms.Compose([
     #随机水平反转
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])
#测试集的数据转换策略test_augs
test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])


