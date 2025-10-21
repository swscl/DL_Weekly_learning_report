import torch
import torchvision#用于可视化？ 导入PyTorch的计算机视觉工具库，包含数据集、模型和图像变换等
from torch.utils import data#导入数据加载工具，主要用于构建数据迭代器
from torchvision import transforms## 导入图像预处理变换工具
from d2l import torch as d2l

#用于展示图片
d2l.use_svg_display()#设置d2l库的图像显示格式为SVG（矢量图），比位图更清晰

"""
读取数据集：Fashion-MNIST由10个类别的图像组成， 
每个类别由训练数据集（train dataset）中的6000张图像 
和测试数据集（test dataset）中的1000张图像组成。 
因此，训练集和测试集分别包含60000和10000张图像。 
测试数据集不会用于训练，只用于评估模型性能。
"""

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式的张量，
# 并除以255使得所有像素的数值均在0～1之间
'''
transforms.ToTensor() 是关键预处理：图像原始格式为 PIL 库的Image类型
（像素值 0-255 整数），转换为 PyTorch 的Tensor类型后，像素值会被归一化
到 0-1（便于模型训练时的数值稳定性）。
'''
trans = transforms.ToTensor()
#下载训练数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data",
    train=True, #加载训练集
    transform=trans, #表示应用上述的图像变换
    download=True
)
#下载测试数据集
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",
    train=False, ##加载测试集
    transform=trans,
    download=True
)

'''
查看数据的基本信息：
每个输入图像的高度和宽度均为28像素。 数据集由灰度图像组成，其通道数为1。
像素图像的形状记为 ℎ×𝑤
'''
#可以查看tensor的长度:训练集和测试集分别是60000张和10000张
print(len(mnist_train), len(mnist_test))
#查看一张图片的形状：[1,28,28],第一个1表示的是黑白图片？？对，表示通道等于1，
# 1 个通道（灰度图，彩色图为 3 通道）、28 像素高、28 像素宽。
#mnist_train[0]是元组（图像张量，标签），[0][0]就是第一张图像的张量，形状是如上
print(mnist_train[0][0].shape)

'''
定义两个可视化数据集的函数
'''
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    #10类标签的文本对应
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 将数字标签（0-9）转换为文本标签
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表，不多解释"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

'''
可视化训练集样本
'''
#以下是训练数据集中前[几个样本的图像及其相应的标签]
# 加载一个批次（18张图片）:data.DataLoader(mnist_train, batch_size=18) 创建一个数据迭代器，每次返回 18 张图片和对应的标签。
# iter(...) 转为迭代器，next(...) 取第一个批次。
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
#X 是批次图像张量，形状为 [18, 1, 28, 28]，通过 reshape(18, 28, 28)
# 去除通道维度（因为是灰度图，单通道可省略），便于显示。
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
d2l.plt.show()

'''
构建高效数据迭代器
读取小批量的数据，与上一节几乎相同的思想
'''
batch_size = 256

def get_dataloader_workers():  #@save，创建4个数据集
    """使用4个进程来读取数据"""
    return 4

#DataLoader 是 PyTorch 加载数据的核心工具，负责按批次读取数据，支持多进程加速。
#num_workers=4 表示用 4 个进程并行加载数据（加快读取速度）
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

#用来测试数据，读一次数据的时间是多少
timer = d2l.Timer()#创建计时器
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
##看看上面的for执行了多长时间，for执行就是将数据在
# 遍历的过程中实施加载，可以体现上面的4个进程并行的作用

#整合以上部分
'''
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
'''

'''
数据迭代器是获得更高性能的关键组件。
依靠实现良好的数据迭代器，利用高性能计算来避免减慢训练过程。
'''

'''
1.减少batch_size（如减少到1）是否会影响读取性能？
答：减小batch_size会增加批次数量，导致数据加载的 IO 次数增多，
   可能降低读取效率（尤其是多进程加载时，过小的批次无法充分利用并行性）。
2.数据迭代器的性能非常重要。当前的实现足够快吗？探索各种选择来改进它。
答：数据迭代器性能是训练效率的关键。可通过调整num_workers（进程数）、
    使用更大的batch_size、预处理数据缓存到内存

'''







