import random#随机
import torch
from d2l import torch as d2l

'''
构造数据集：
w=[2,-3.4]T,b=4.2,噪声项
y=Xw+b+噪声
'''
# 定义生成数据集的函数
#给定w,b和生成的n个样本：
def synthetic_data(w,b,num_examples):
    '''生成y=Xw+b+噪声'''
    #生成特征矩阵X：size =(样本数，特征数)，每个元素是均值0方差1的正态分布
    #torch.normal用于生成正态分布（高斯分布）随机数的函数
    X=torch.normal(0,1,size=(num_examples,len(w)))
    # matmul是矩阵乘法，按张量维度自动匹配矩阵乘法逻辑:X与w的矩阵乘法 + 偏置b
    y=torch.matmul(X,w)+b
    # 给y添加噪声：噪声服从均值0、标准差0.01的正态分布，形状与y一致
    y+=torch.normal(0,0.01,size=y.shape)
    # 返回特征X和标签y（将y reshape为列向量，方便后续计算）
    return X,y.reshape(-1,1)#-1表示自动计算出多少行，1表示弄成1列


# 定义真实的权重w和偏置b（我们希望模型最终能学到这两个值）
true_w = torch.tensor([2, -3.4])  # 真实权重：2个特征对应的系数
true_b = 4.2  # 真实偏置

features,labels = synthetic_data(true_w,true_b,100)

"""features中每一行都包含了一个二维数据样本，labels中的每一行都包含了一个一维标签"""
#print('features:',features[0],'\nlabel:',labels[0])

'''
定义数据迭代器（生成小批量数据）
定义一个data_iter函数，函数接受批量的大小，特征矩阵x和标签向量y作为输入，生成大小为batch_size的小批量'''
def data_iter(batch_size,features,labels):
    num_examples=len(features)#总的样本数100个
    indices = list(range(num_examples))#生成样本索引，从0到99
    random.shuffle(indices)#将上面按顺序的整数打乱，将0到99打乱

    #按照batchsize的大小提取小批量的数据
    for i in range(0,num_examples,batch_size):
        """每一次跳batch_size的大小，就是i的自增单位是batchsize"""
        # 每一次取一个批量，但是最后一组可能会数量不足，所以可能会不够
        # indices是列表切片的语法，torch.tensor是将这个列表转化成tensor格式
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
        '''
        features[batch_indices] 和 labels[batch_indices]：根据当前批次的索引，
        从特征数据 features 和标签数据 labels 中取出对应样本，组成一个批次。
        yield：这是 Python 生成器（generator）的语法，作用是 “逐个返回” 每个批次的数据，
        而不是一次性生成所有批次。每次循环到 yield 时，会返回当前批次，暂停执行；下次迭代时，
        从暂停的位置继续运行，直到所有批次处理完。
        '''

batch_size = 10

# for X,y in data_iter(batch_size,features,labels):
#     print('X:',X,'\n','y:',y)
#     break

'''定义初始化模型参数，w和b'''
#requires_grad=True表示需要计算梯度
w=torch.normal(0,0.01,size = (2,1),requires_grad=True)#要求梯度就加上requires_grad=True
b=torch.zeros(1,requires_grad=True)

"""定义模型:其实就是线性回归的公式"""
def linreg(X,w,b):
    '''线性回归模型'''
    '''前向计算，跟踪了梯度吗？是的'''
    return torch.matmul(X,w)+b#返回10*1

"""定义损失函数（均方损失）"""
def squared_loss(y_hat,y):
    """均方损失"""
    """均方损失：(预测值-真实值)² / 2"""
    # 先将y的形状调整为与y_hat一致（避免维度不匹配），再计算损失
    return (y_hat-y.reshape(y_hat.shape))**2/2

'''定义优化算法'''
def sgd(params,lr,batch_size):
    """小批量随机梯度下降，params是所有的参数列表，需要更新参数"""
    with torch.no_grad():# 该上下文内的操作不跟踪梯度（参数更新不需要求导）
        for param in params:#遍历所有的参数(w和b)
            ## 参数更新公式：param = param - 学习率 * 梯度均值（除以batch_size是因为损失是总和）
            param-=lr * param.grad / batch_size#之前的损失函数没有求均值，这里需要加上均值
            param.grad.zero_()#手动将梯度设置成0
'''
在你的代码中，w 和 b 的梯度已经通过 l.sum().backward() 计算好了
（存储在 w.grad 和 b.grad 中），sgd 函数只需要用这些梯度来更新参数，
不需要再对 “更新参数” 的过程求导，因此必须用 with torch.no_grad()
 来禁用跟踪
'''
lr = 0.03
num_epochs = 3 #将整个数据扫描三遍
net = linreg #定义的模型
loss = squared_loss #损失是均方损失

'''模型的训练流程'''
#每一轮训练
for epoch in range(num_epochs):
    #data_iter用来返回小批量的数据，遍历小批量的数据
    for X,y in data_iter(batch_size,features,labels):
        ###看看！net(X,w，b)返回的是10*1的张良，就是y_hat（预测值），loss，
        l = loss(net(X,w,b),y) #小批量的损失：计算当前批量的预测损失（l是每个样本的损失，形状为(batch_size, 1)）
        ## 对损失求和后反向传播，计算w和b的梯度，sum()就是吧损失从向量转化为标量
        #backward()方法默认需要一个标量（单个数值）作为起点来触发反向传播（因为梯度是 “标量对变量的导数”）
        #.backward()：沿计算图反向传播，计算梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    #每轮结束后，计算整个训练集的损失（不跟踪梯度，节省资源）
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        # 打印当前轮次和平均损失（损失逐渐减小，说明模型在学习）
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')


'''
关于计算图：
具体来说，当你调用 l.sum().backward() 时：
1.框架会从损失l（计算图的输出节点）出发，沿着计算图的反向边（与数据流向相反）遍历；
2.对每个运算节点，根据其对应的求导规则（比如+的导数是 1，*的导数是另一个因子等），
计算当前节点对前一个节点的局部导数；
3.通过链式法则，将局部导数相乘，最终得到损失l对每个参数（w、b）的梯度（存储在w.grad、
b.grad中）。
'''

'''
关于跟踪梯度：
当我们对 requires_grad=True 的张量进行运算时（比如计算 y_hat = Xw + b 或
 l = loss(y_hat, y)），PyTorch 会在前向传播过程中实时跟踪每一步运算：
 记录参与运算的张量（如 X、w、b、y）；
 记录运算类型（如 matmul、+、^2 等）；
 按顺序将这些 “张量” 和 “运算” 连接成计算图（无需重复计算，只是记录依赖关系）。
这个 “跟踪” 过程是伴随第一次前向计算同步进行的，而不是 “重新计算一遍计算图”。
计算图只在第一次前向时构建一次（针对当前输入），后续如果输入或参数变化，
会动态更新计算图，但核心是 “记录” 而非 “重复计算”。
'''

'''
总结:
需要特别关注 “不需要跟踪梯度” 的场景，本质是区分 “需要求导的计算” 和 “不需要求导的操作”：
凡是与 “参数更新” 无关的步骤（推理、评估、参数查看、手动更新参数等），都应禁用梯度跟踪，以节省资源并避免逻辑错误。
只有 “前向传播计算损失” 的过程需要跟踪梯度，为后续反向传播（求导）做准备
'''
#print("w的估计误差：",true_w-w.reshape(true_w.shape))
#print("b的估计误差：",true_b-b)