import pandas as pd
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
'''
pandas：用于数据处理和读取CSV文件。
torch：PyTorch深度学习框架。
nn：PyTorch的神经网络模块。
data：PyTorch的数据加载工具。
'''

# 读取数据
data_trian = pd.read_csv('./data/kaggle_house_pred_train.csv')
data_val = pd.read_csv('./data/kaggle_house_pred_test.csv')
'''
读取训练集和测试集（这里测试集实际上是验证集，
因为Kaggle比赛中测试集没有标签，但这里我们用来做预测提交）。
'''
#print(data_trian.shape) # 训练集  (1460, 81)
#print(data_val.shape)  #  验证集  (1459, 80) //少了一个标签

'''
数据预处理：去除无关特征并合并数据集，以及筛选数值型特征
'''
# 去除id
all_features = pd.concat((data_trian.iloc[:, 1:-1], data_val.iloc[:, 1:]))
# 获取值为数字的索引
numeric = all_features.dtypes[all_features.dtypes != 'object'].index
'''
data_trian.iloc[:, 1:-1] 表示取训练集的所有行，
列从第 2 列（索引 1）到倒数第 2 列（排除最后一列标签SalePrice）；
data_val.iloc[:, 1:] 表示取测试集的所有行，列从第 2 列开始（排除第一列Id）。
用 pd.concat 将两者合并为 all_features，
目的是统一对训练集和测试集的特征进行预处理（避免分开处理导致的不一致）。
去除Id是因为它是样本的唯一标识，
与房价预测无关，属于无关特征，保留会干扰模型。
'''
'''
all_features.dtypes 是 pandas 中的 Series 对象（可以理解为带索引的一维数组）。
它的 索引（index） 是 all_features 中所有列的列名；
它的 值（values） 是对应列的数据类型（比如 int64、float64、object 等）。
然后[] 是 pandas 中 布尔索引（Boolean Indexing） 的用法，作用是「根据条件筛选数据」，筛选出来的是一个seris，他包含索引和值，同上
筛选之后再.index就会保留所有数据类型是数字的索引了
numeric
Out[3]: 
Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
'''

# 归一化
all_features[numeric] = all_features[numeric].apply(lambda x: ((x - x.mean()) / x.std()))
'''
选中之前筛选出来的数值类型的索引列all_features[numeric]，然后对每一列进行标准化：apply(lambda x: ((x - x.mean()) / x.std()))
x.mean()是该列的均值，x.std()是该列的标准差。
计算逻辑：（当前值 - 列均值）÷ 列标准差，最终让每列特征的均值为 0、标准差为 1。
问：.apply()的用法:他是一个批量处理函数，默认是对列进行处理，比如对allfeature这个dataframe进行函数应用
lambda的用法：就是省略的函数，接收 x 后，计算 “（x 的值 - x 的均值）÷ x 的标准差”，并返回这个结果
'''
# 将缺失值补全
all_features[numeric] = all_features[numeric].fillna(0)
'''
fillna(0)：将数值型特征列中的所有缺失值（NaN）填充为 0。填充为0不会影响归一化的结果。
然后.fillna(0),就是一个固定的填充手法
问：all_features[numric]是一个筛选出来的dataframe是吗，对
'''
# 将单一变量变成独热向量,这一步会明显在增加列数
print(all_features.shape)
all_features = pd.get_dummies(all_features, dummy_na=True)

'''
pd.get_dummies()是将object类型的特征，比如材质区域转化成独热编码，比如狗猫鼠，会转化成3列的向量
如果是狗，就是（1，0，0）
dummy_na=True就是将缺失值也当作一个类别。
'''
#############数据预处理到此结束！！！#############
###接下来是将预处理后的数据转换为 PyTorch 可识别的张量（Tensor），并提取训练标签：
# 将数据转为pytorch中的张量
feature_train = torch.tensor(all_features[:data_trian.shape[0]].values.astype(float), dtype=torch.float32)
'''
因为allfeatures是包含了训练集和验证集两个样本集的所有特征共2919行，现在要把两个样本集的数据分开，就从第一行到data_trian.shape[0]。
all_features[]默认对行处理
.values：将 DataFrame 转换为 NumPy 数组（pandas 和 PyTorch 之间的中间格式）。
.astype(float)：确保数组元素是浮点型（模型通常处理浮点数据）
torch.tensor(..., dtype=torch.float32)：将 NumPy 数组转换为 PyTorch 的Tensor，并指定数据类型为
float32（深度学习中常用的精度，平衡计算效率和数值精度）
'''
feature_test: torch.Tensor = torch.tensor(all_features[data_trian.shape[0]:].values.astype(float), dtype=torch.float32)
'''
从1460到最后的行中
问：这是什么鬼语法：': torch.Tensor'是类型注解，告诉开发者，这句话的作用就是feature_test变量的预期就是torch.tensor
也完全可以去掉注解
'''
# 提取标签
train_label = torch.tensor(data_trian['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)
'''
.reshape(-1, 1)：将一维数组转换为二维数组(1460, 1)（每行一个样本，一列标签），确保和模型输出的形状（(batch_size, 1)）匹配。
'''

# 定义模型

def get_net():
    return nn.Sequential(nn.Linear(feature_train.shape[1], 1))
'''
nn.Sequential是pytorch中的一个容器，可以按顺序堆叠神经网络
这个堆叠中实际只包含了一个全连接层对吗？
再线性全连接层中，输入特征的维度是feature_train.shape[1],就是训练特征的列数，
为什么是330？什么时候变成330的？将object特征变成独热编码的时候增加了列
定义了一个最简单的神经网络 ——单层线性回归模型（相当于 “线性回归” 的神经网络实现），输入是所有特征，输出是预测的房价。
'''

# 定义损失函数

loss = nn.MSELoss()
'''
均方误差损失函数，怎么用的？
答：看后面使用的例子loss(torch.log(clipped_pred), torch.log(label))，传入两个参数
第一个是预测值，第二个是真实标签，这里对他取了一个log,默认会返回所有样本的平均均方误差
问：为什么不使用交叉熵损失？
答：因为这是一个回归问题（预测连续值，如房价），
MSE 是回归任务中最常用的损失函数，能有效衡量预测值与真实值的差异。
'''

# 评估损失：对模型效果的评估，比直接用mse更加合理，用‘对数 RMSE’作为评估指标
def los_rmse(net, feature, label):
    clipped_pred = torch.clamp(net(feature), 1, float('inf'))
#clipped_pred是对模型的输出值进行截断处理，目的是确保预测值再合理的范围内，因为log的真数大于1 ，不然计算loss会出现问题
    rmse = torch.sqrt(loss(torch.log(clipped_pred), torch.log(label)))
    return rmse
'''
torch.sqrt(loss(...))：计算均方根误差（RMSE）。因为 loss 是 MSE（均方误差），
开平方后得到 RMSE，其单位与原目标变量（房价）一致
net(feature) 是模型对输入特征 feature 的原始预测值（比如预测的房价）；
torch.clamp(x, min, max) 是 PyTorch 的一个函数，作用是将张量 x 中的所有元素 “截断” 到 [min, max] 范围内：
若元素值 < min，则强制设为 min；
若元素值 > max，则强制设为 max；
若在范围内，则保持不变。
这里 min=1，max=float('inf')（正无穷），所以 clipped_pred 的含义是：
将模型的原始预测值限制在 “不小于 1” 的范围内（即预测值最小为 1，
更大的值保持不变）。
'''
def train(net: nn.Sequential, train_features, train_label,
          test_feature, test_label, epoch_num,
          learn_rate, weight_decay, batch_size):
    #test_feature和test_label可以再训练时，不传入，但是都写在这了，怎么知道它可以不传入呢？
    #weight_decy是权重衰减，比如L2正则化，防止模型过拟合

    #数据加载器，按批次读取训练数据，这里只加载了子训练集，不包含验证集，和传参有关
    train_iter = data.DataLoader(
        #作用是将特征和标签绑定成’样本-标签‘对，长啥样呢？？？？？
        data.TensorDataset(train_features, train_label),
        batch_size=batch_size, shuffle=True
    )

    #记录下训练/验证集的损失，用于观察数据的表现，损失记录列表（train_ls, test_ls
    train_ls, test_ls = [], [] ## 记录每轮训练后的损失，后续可画图看模型是否收敛
    ##画一个tensorboard看看

    # 优化器：Adam优化器，可以带权重衰减防止过拟合
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate, weight_decay=weight_decay)

    for epoch in range(epoch_num):
        # 1.训练阶段：遍历每一个批次epoch来更新模型的参数
        for X, y in train_iter:
            #梯度清零：避免上一次的梯度积累优化器（optimizer）封装了所有需要更新的参数，调用 optimizer.zero_grad() 本质是 “批量清零所有参数的梯度”，比手动逐个参数清零更简洁、更不易出错。
            optimizer.zero_grad()
            #前向传播：计算模型的预测值和损失
            l: torch.Tensor = loss(net(X), y)
            #反向传播：计算梯度（是不是说，反向传播就会计算一次导数，如果不对梯度进行清零就会导致梯度的累积？怎么累积的呢，找一个课看看）
            l.backward()
            #参数更新：用梯度来调整模型的权重。# 按梯度和学习率，更新模型的权重、偏置（让损失更小）
            optimizer.step()
        # 2.评估记录，每一轮记录一次：记录当前循环的额批次的训练和验证的损失？为什么会有验证集，验证集哪有损失，这是子训练集划分的
        train_ls.append(los_rmse(net, train_features, train_label))
        # 若有验证集，记录验证集损失（监控是否过拟合）
        if test_label is not None:
            test_ls.append(los_rmse(net, test_feature, test_label))
    # 返回的这是一个列表嘛？？怎么可视化？？
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """
    这个函数是把训练集拆成训练集和验证集
    k：总折数（比如 5 折交叉验证，k=5）；
    i：当前要拆分的折数（从 0 到 k-1，比如 5 折中i=0表示第 1 折，i=1表示第 2 折）；
    这个i就是作为验证集的那一折嘛？？
    X：完整的训练集特征（比如feature_train，形状(1460, 330)）；
    y：完整的训练集标签（比如train_label，形状(1460, 1)）。
    """
    #确保k折有效
    assert k > 1
    #计算每一折的样本数，双杠是整数除法，如果除不尽怎么办？？？？？？？？？？？？
    fold_size = X.shape[0] // k
    #初始化训练集和验证集变量
    X_train, Y_train = None, None

    for j in range(k):## j从0到k-1，遍历每一个可能的折
        # 步骤①：确定第j折的样本索引范围
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # slice(a, b)等价于索引[a:b]，表示从第a个样本到第b-1个样本（左闭右开）
        # 比如j=0时，idx=slice(0, 292) → 取0-291号样本（共292个）
        # j=1时，idx=slice(292, 584) → 取292-583号样本（共292个），以此类推

        # 步骤②：取出第j折的特征和标签
        X_part, Y_part = X[idx, :], y[idx]
        # X[idx, :]：取第j折的所有样本（idx范围）和所有特征（:表示全部列）
        # y[idx]：取第j折的所有标签

        # 步骤③：判断第j折是否是当前要作为验证集的第i折
        if j == i:
            # 如果是第i折，就把这部分作为验证集
            X_valid, Y_valid = X_part, Y_part

        elif X_train is None:
            # 如果不是第i折，且是第一次遇到非验证集的折（X_train还是None），就把这部分作为训练集的初始数据
            X_train, Y_train = X_part, Y_part

        else:
            # 如果不是第i折，且已有部分训练集数据，就把当前折拼接到训练集上
            # torch.cat([a, b], dim=0)：按行拼接（dim=0表示行维度），即增加样本数
            X_train = torch.cat((X_train, X_part), 0)
            Y_train = torch.cat((Y_train, Y_part), 0)

    return X_train, Y_train, X_valid, Y_valid
'''
返回 4 个变量：
X_train：子训练集特征（k-1 折样本）；
y_train：子训练集标签；
X_valid：子验证集特征（第 i 折样本）；
y_valid：子验证集标签。
'''

'''
k_fold 函数 —— 它是 K 折交叉验证的 “总调度中心”，
核心逻辑是循环调用 get_k_fold_data 拆分 k 次数据，
每次训练 1 折，最终返回 k 次训练的平均损失，确保模型评估更稳定。
'''
def k_fold(k, train_features, train_labels, learn_rate, epoch_num, weight_decay, batch_size):
    # 总损失
    train_l_sum, valid_l_sum = 0, 0# 记录K次训练的训练集和验证集总损失（最后求平均）

    train_l_sum_p = torch.zeros(epoch_num)  # 存储k折的“每轮训练损失总和”（100个元素）
    valid_l_sum_p= torch.zeros(epoch_num)  # 存储k折的“每轮验证损失总和”（100个元素）

    # 开始了，训练k次，除不尽怎么办
    for i in range(k):
        # 调用函数，对应每一个折进行拆分。
        #data 是一个元组，格式为 (X_train_fold, y_train_fold, X_valid_fold, y_valid_fold)；
        data = get_k_fold_data(k, i, train_features, train_labels)

        #每一折都重新创建新的模型，而不是复用之前的模型，防止前一折训练好的权重会污染后面的，作为初始权重
        net = get_net()
        #调用训练函数，*data是python中的解包操作，将data元组拆成4个参数，直接传给train函数
        #最后返回的是当前折的每一轮的损失的数组，包括验证损失和训练损失
        train_ls, valid_ls = train(net, *data, epoch_num=epoch_num, weight_decay=weight_decay, batch_size=batch_size,
                                   learn_rate=learn_rate)

        #train_ls[-1]：取 train_ls 列表的最后一个元素，即当前折 “最后一轮训练” 的损失（模型收敛后的最终损失）；
        #把 k 折的最终损失都累加起来，后续用于计算平均值 —— 平均值比单折损失更稳定，能避免 “某一折数据特殊导致的评估偏差”。
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        ###画图
        train_l_sum_p += torch.tensor([x.detach() for x in train_ls])  # 100个元素累加
        valid_l_sum_p += torch.tensor([x.detach() for x in valid_ls])  # 100个元素累加

        #print(train_ls)
        print(f'{i}折,训练log rsme{train_ls[-1]:f},验证lpg rsme{valid_ls[-1]:f}')

    return train_l_sum / k, valid_l_sum / k,train_l_sum_p / k,valid_l_sum_p / k
#返回值：train_l_sum/k（k 次训练损失的平均值）、valid_l_sum/k（k 次验证损失的平均值）

# 定义超参数

learn_rate = 5

k = 5

epoch_num = 100

weigth_decay = 0

bacth_size = 64

train_l, valid_l,train_l_per_epoch, valid_l_per_epoch  = k_fold(k, feature_train, train_label,learn_rate, epoch_num,  weigth_decay, bacth_size)

# tl,vl = k_fold(k,feature_train,train_label,learn_rate,epoch_num,weigth_decy,bacth_size)

# print(f"{k}折验证，平均训练log rsme {tl:f},平均验证log rsme {vl:f}")


def train_and_pred(train_featrue, train_label, test_feature: torch.Tensor, test_label, learn_rate,

                   epoch_num, weight_decay, batch_size):
    #初始化模型
    net: nn.Sequential = get_net()
    #调用train函数，不使用验证集，# 这里的`_`表示忽略验证损失列表（因为没传入验证集，返回的验证损失为空）
    train_ls, _ = train(net, train_featrue, train_label, test_feature, test_label, epoch_num, learn_rate, weight_decay,
                        batch_size)
    #打印最终的训练损失，确认模型收敛
    print(f'训练rsme {train_ls[-1]:f}')

    ####核心步骤2：预测测试集和整理结果
    #① 预测测试集房价（等价于之前的predict函数逻辑）
    pred = net(test_feature).detach().numpy()
    # net(test_features)：模型对测试集特征做前向传播，得到预测张量；
    # detach().numpy()：脱离计算图，转为NumPy数组（形状(1459, 1)）

    # ② 将预测结果存入test_data的'SalePrice'列
    data_val['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    # preds.reshape(1, -1)[0]：将(1459, 1)转为(1459,)的一维数组，符合Pandas Series的格式；
    # test_data原本包含测试集的'Id'列，新增'SalePrice'列后，就有了“ID+预测房价”的完整信息。

    # ③ 拼接ID和预测值，保存为CSV
    submission = pd.concat([data_val['Id'], data_val['SalePrice']], axis=1)
    # pd.concat(..., axis=1)：按列拼接（横向拼接），得到两列数据：'Id'和'SalePrice'；
    submission.to_csv("submission.csv", index=False)
    # 保存为CSV，index=False表示不保留行索引（比赛平台要求的格式）。
'''
train_features：全量训练集特征（feature_train，1460 样本）；
test_features：测试集特征（feature_test，1459 样本）；
train_labels：全量训练集标签（train_label，房价）；
test_data：原始测试集的 Pandas DataFrame（包含Id列，用于后续拼接结果）；
num_epochs/lr/weight_decay/batch_size：K 折交叉验证确定的最优超参数。
k折什么时候确定了最优超参数了？？
'''
train_and_pred(feature_train, train_label, feature_test, None, learn_rate, epoch_num, weigth_decay, bacth_size)


def plot_loss(train_ls, test_ls, epoch_num):
    epochs = list(range(1, epoch_num + 1))  # x轴轮次

    # 创建一个窗口，包含2行1列的子图（上下排列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  # 总尺寸宽8，高8

    # 第一张子图：训练损失（上半部分）
    ax1.plot(epochs, train_ls, label='train_loss', color='blue')
    ax1.set_xlabel('epoch')  # x轴标签
    ax1.set_ylabel('RMSE')  # y轴标签，损失
    ax1.set_title('train_loss')  # 子图标题
    ax1.legend()  # 显示标签
    ax1.grid(linestyle='--')  # 网格线

    # 第二张子图：验证损失（下半部分，仅当有数据时绘制）
    if len(test_ls) > 0:
        ax2.plot(epochs, test_ls, label='verify_loss', color='red')
        ax2.set_xlabel('epoch')#轮次
        ax2.set_ylabel('RMSE')
        ax2.set_title('verify_loss')
        ax2.legend()
        ax2.grid(linestyle='--')

    # 调整子图间距，避免标题/标签重叠
    plt.tight_layout()

    # 显示整个窗口（两张图同时显示）
    plt.show()

train_l_per_epoch = train_l_per_epoch.numpy()
valid_l_per_epoch = valid_l_per_epoch.numpy()
# 绘制损失曲线

plot_loss(train_l_per_epoch, valid_l_per_epoch, epoch_num=100)