import torch
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms

# 定义函数和类（这些可以放在全局作用域，不影响）
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader_workers():
    """使用4个进程来读取数据（Windows下建议调试时设为0）"""
    return 4  # 若仍报错，可改为0

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch + 1}/{num_epochs}")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  train acc:  {train_acc:.4f}")
        print(f"  test acc:   {test_acc:.4f}\n")
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:  # 取第一批测试数据
        break
    trues = d2l.get_fashion_mnist_labels(y)  # 真实标签的文本描述
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  # 预测标签的文本描述
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]  # 组合真实标签和预测标签
    # 显示前n个样本的图像和标签
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
    d2l.plt.show()

# ********** 关键：所有执行逻辑必须放在这个块中（Windows多进程要求）**********
if __name__ == '__main__':
    batch_size = 256

    # 加载数据集
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=trans,
        download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=trans,
        download=True
    )

    # 创建数据迭代器（这里会用到多进程）
    train_iter = data.DataLoader(
        mnist_train, batch_size, shuffle=True,
        num_workers=get_dataloader_workers()
    )
    test_iter = data.DataLoader(
        mnist_test, batch_size, shuffle=False,
        num_workers=get_dataloader_workers()
    )
    print('数据集加载完成\n')

    # 初始化模型参数
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 打印初始准确率（随机参数，准确率应接近1/10）
    print("初始模型在测试集上的准确度：\n", evaluate_accuracy(net, test_iter))

    # 训练参数
    lr = 0.1
    num_epochs = 10

    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)



# 启动训练
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)
