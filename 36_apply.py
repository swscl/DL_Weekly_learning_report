import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F  # 新增：BatchNorm计算需要

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
def load_cifar10(is_train, augs, batch_size):
    '''
    把 “原始 CIFAR10 数据”→“经过增强 / 转换的 Tensor 数据”→“按批次迭代的格式”，适配模型训练。
    '''
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=0)
    return dataloader

# 训练集的数据增强策略train_augs
train_augs = torchvision.transforms.Compose([
     #随机水平反转
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])
#测试集的数据转换策略test_augs
test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# ========== 新增：适配CIFAR10的LeNet+BatchNorm ==========
net = nn.Sequential(
    # 改1：输入通道从1→3（CIFAR10是RGB 3通道）
    nn.Conv2d(3, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 32×32→16×16

    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 16×16→8×8

    nn.Flatten(),
    # 改2：全连接层输入维度（16×5×）
    nn.Linear(16 * 5 * 5, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),

    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    # 改3：输出维度保持10（CIFAR10正好10类，不用改）
    nn.Linear(84, 10)
)

# ========== 新增：加载CIFAR10数据迭代器 ==========
# 定义超参数（可根据显卡显存调整batch_size，比如显存小就改成128）
batch_size = 256
lr = 1.0  # 学习率（和你之前训练LeNet的参数一致）
num_epochs = 10  # 训练轮数

# 加载训练集：传入train_augs（带水平翻转的增强）
train_iter = load_cifar10(is_train=True, augs=train_augs, batch_size=batch_size)
# 加载测试集：传入test_augs（只转Tensor，无增强）
test_iter = load_cifar10(is_train=False, augs=test_augs, batch_size=batch_size)

# 可选：验证数据形状（确保适配模型）
X, y = next(iter(train_iter))
print(f"训练集单批次形状：X={X.shape}（批次×通道×高×宽），y={y.shape}（批次）")
# 正确输出：X=torch.Size([256, 3, 32, 32])，y=torch.Size([256])

# ========== 新增：训练模型 ==========
# 自动选择GPU（没有就用CPU）
device = d2l.try_gpu()
print(f"使用设备：{device}")

# 开始训练（逻辑和FashionMNIST完全一致）
d2l.train_ch6(
    net,                # 适配CIFAR10的LeNet+BatchNorm模型
    train_iter,         # CIFAR10训练迭代器（带增强）
    test_iter,          # CIFAR10测试迭代器（无增强）
    num_epochs,         # 训练轮数
    lr,                 # 学习率
    device              # 训练设备（GPU/CPU）
)

# ========== 可选：验证增强效果和预测结果 ==========
import matplotlib.pyplot as plt

# 1. 显示增强后的训练集图片
samples = [train_iter.dataset[i][0].permute(1,2,0).numpy() for i in range(8)]
plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(samples[i])
    plt.axis('off')
plt.title("增强后的CIFAR10训练样本")
plt.show()

# 2. 预测测试集前10张图
X, y = next(iter(test_iter))
preds = net(X.to(device)).argmax(dim=1)  # 预测类别

# 显示图片+真实标签+预测标签
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i].permute(1,2,0).numpy())
    plt.title(f"真实：{y[i]}\n预测：{preds[i].cpu().numpy()}")
    plt.axis('off')
plt.show()
