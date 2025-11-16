import torch
from IPython import display
from d2l import torch as d2l
import torchvision#用于可视化？ 导入PyTorch的计算机视觉工具库，包含数据集、模型和图像变换等
from torch.utils import data#导入数据加载工具，主要用于构建数据迭代器
from torchvision import transforms## 导入图像预处理变换工具
batch_size = 256

#这就是加载图像分类数据的load_data_fashion_mnist函数，这也在d2l中封装好了
#train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trans = transforms.ToTensor()
#下载训练数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True, #加载训练集
    transform=trans, #表示应用上述的图像变换
    download=True
)
#下载测试数据集
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False, ##加载测试集
    transform=trans,
    download=True
)
def get_dataloader_workers():  #@save，创建4个数据集
    """使用4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
test_iter  = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers())
print('数据集加载完成\n')


'''初始化模型参数'''
num_inputs = 784#模型输入：softmax回归的输入是一个向量，所以要将图像1*28*28展开成一个向量
num_outputs = 10#模型输出的维度：10个类

#初始化参数的维度，input看成X：1*784,W:784*10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

'''定义softmax操作'''
# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
#

def softmax(X):
    """其实就是softmax的公式实现"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# X_prob, X_prob.sum(1)

'''定义模型：实现softmax的回归模型'''
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


'''定义损失函数：交叉熵损失 '''
#这是一个补充操作
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y]
def cross_entropy(y_hat, y):
    """实现交叉熵的公式"""
    return - torch.log(y_hat[range(len(y_hat)), y])

# print(y_hat[range(len(y_hat)), y])
# print("****交叉熵：：",cross_entropy(y_hat, y))


'''分类精度：将预测的类别与真实的Y进行比较'''
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:#是一个2维矩阵，列数也>1
        y_hat = y_hat.argmax(axis=1)#按照每一行求argmax,返回的是每一行的最大值的下标，作为预测的类别
    cmp = y_hat.type(y.dtype) == y#作比较
    return float(cmp.type(y.dtype).sum())
#计算正确率：
# print('预测的正确率：',accuracy(y_hat, y) / len(y))

'''评估任意模型net的准确率'''
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print("任意数据集上的准确度：\n",evaluate_accuracy(net, test_iter))


'''softmax回归的训练：对整个数据迭代一次'''
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    #isinstance(对象, 类/类型)，作用是：判断第一个参数（对象）是否是第二个参数（类或类型）的实例（包括该类的子类实例），返回布尔值 True 或 False。
    if isinstance(net, torch.nn.Module):#用nn.module来实现的
        net.train()# 将模型设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)#用长度为3的迭代器，来累加信息
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)#计算y_hat
        l = loss(y_hat, y)#计算l,用交叉熵
        if isinstance(updater, torch.optim.Optimizer):#updater是一个优化器，比如sgd
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()#清楚历史梯度
            l.mean().backward()
            updater.step()#调整参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


'''修改后的训练函数：只打印不绘图'''
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并打印每个epoch的指标"""
    for epoch in range(num_epochs):
        # 训练一个周期并获取指标
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        # 计算测试集准确率
        test_acc = evaluate_accuracy(net, test_iter)
        # 打印当前epoch的指标
        print(f"epoch {epoch + 1}/{num_epochs}")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  train acc:  {train_acc:.4f}")
        print(f"  test acc:   {test_acc:.4f}\n")
    # 最终验证指标是否达标
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

#num_epochs = 10
#显示
#train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

'''
预测
'''
'''
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
'''




