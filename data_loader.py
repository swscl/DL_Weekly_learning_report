# data_loader.py（修改后）
import torchvision
from torchvision import transforms
import torch.utils.data as data


def get_dataloader_workers(num_workers=0):  # 新增参数，默认0
    return num_workers


def load_fashion_mnist(batch_size, root="./data", transform=None, num_workers=0):  # 新增num_workers参数
    if transform is None:
        transform = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=transform, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=transform, download=True
    )

    train_iter = data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True,
        num_workers=get_dataloader_workers(num_workers)  # 传递参数
    )
    test_iter = data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False,
        num_workers=get_dataloader_workers(num_workers)  # 传递参数
    )

    print("数据集加载完成")
    return train_iter, test_iter