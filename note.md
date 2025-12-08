### 21：卷积的多输入输出通道.

#### 21-1：通道

计算：

![image-20251205140649115](https://gitee.com/shuchangluo/md_images/raw/master/image-20251205140649115.png)

这说明：

卷积核的通道数（深度）是要核和入图片的深度数对应的，一个卷积核可以得到一个特征响应图，这就是单输出通道。

卷积核的个数（层数）是随意的，指定有多少个不同的卷积核对输入图片进行计算，响应就会产生多少个特征相应图，这些相应图堆叠在一起就是多输出通道。

总结：

- 输出通道数是卷积层的超参数
- 每个输入通道有独立的二维卷积核，所有通道结果相加得到一个输出通道结果
- 每个输出通道有独立的三维卷积核

#### 21-2：多输入多输出通道

##### 1.多输入通道的互相关运算

```python
import torch
from d2l import torch as d2l

#多输入通道的互相关运算
def corr2d_muti_in(x,k):
    return sum (d2l.corr2d(x,k) for x ,k in zip(x,k))
```

** 输入参数的含义**

- **`x` (输入)**: 一个具有多个通道的输入张量。它的形状通常是 $(\text{通道数}, \text{高度}, \text{宽度})$。
- **`k` (核)**: 一个具有多个通道的卷积核张量。它的形状通常是 $(\text{输入通道数}, \text{核高度}, \text{核宽度})$

**核心运算步骤**

该函数通过 `zip` 和 `sum` 实现了多通道互相关运算的两个关键步骤：

1. **`zip(x, k)`**: 将输入的**每个通道**（$x$ 的第一个维度）与其对应的卷积核通道（$k$ 的第一个维度）配对。
   - 比如，如果输入 $x$ 有 $C_{in}$ 个通道，核 $k$ 也有 $C_{in}$ 个通道，`zip` 会生成 $C_{in}$ 对 $(x_i, k_i)$，其中 $x_i$ 和 $k_i$ 都是二维张量（$\text{高度} \times \text{宽度}$ 和 $\text{核高度} \times \text{核宽度}$）。
2. **`d2l.corr2d(x, k)`**: 对配对后的**每个 $2D$ 通道**进行标准的互相关运算。这会得到 $C_{in}$ 个独立的 $2D$ 输出特征图。
3. **`sum(...)`**: 将所有通道独立计算得到的 $2D$ 输出结果**相加（求和）**。

##### 2.多个通道的输出的互相关函数

注意，多个通道输出意味着由多个深度和输入相同深度的卷积核，才能得出多个输出通道，卷积核的个数决定了输出的通道，所以再这个例子中k是4d的。

```python
def corr2d_muti_in_out(x,k):
    return torch.stack([corr2d_muti_in(x,k) for k in k],0)

#k = torch.stack((k,k+1,k+2),0)
```

##### 问题：`for k in k` 为什么取的是第一个维度 ($\mathbf{C_{out}}$)？

在 PyTorch (以及 Python 科学计算库，如 NumPy) 中，当直接对一个多维张量进行迭代 (`for item in tensor_name:`) 时，默认的行为是按第一个轴（维度 0）进行切片和迭代。

**问题：`stack` 执行的堆叠具体是什么内涵：**

循环 `[corr2d_muti_in(x, k) for k in k]` 结束后，得到的是一个包含 $C_{out}$ 个元素的列表，每个元素都是一个 $2D$ 矩阵（即一个输出特征图）。

当执行 `torch.stack(..., 0)` 时：

1. 新建维度： `stack` 新建了一个维度 0，用来存放所有这些 $2D$ 矩阵。

2. 新维度含义： 这个新建的维度 0 就是输出通道 (Output Channel) 维度。

3. 最终形状： 最终返回的 $3D$ 张量形状是：

   

   $$(\mathbf{C_{out}}, \text{H}, \text{W})$$

##### 3.1*1的卷积就等价于一个全连接

```python
def corr2d_multi_in_out_1x1(x,k):
    c_i,h,w=x.shape
    c_o =k.shape[0]
    x=x.reshape((c_i,h*w))
    k=k.reshape((c_o,c_i))
    y=torch.matmul((k,x))

    return y.reshape((c_o,h,w))

x=torch.normal(0,1,size=(3,3,3))
k = torch.normal(0,1,size=(2,3,1,1))

y1 = corr2d_multi_in_out_1x1(x,k)
y2 = corr2d_muti_in_out(x,k)
```

打个草稿就明白是怎么回事了

