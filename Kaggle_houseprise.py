# 完整代码

import pandas as pd

import torch

from torch import nn

from torch.utils import data

# 读取数据

data_trian = pd.read_csv('./data/kaggle_house_pred_train.csv')

data_val = pd.read_csv('./data/kaggle_house_pred_test.csv')

# print(data_trian.shape) 训练集  (1460, 81)

# print(data_val.shape)   验证集  (1459, 80) //少了一个标签


# 去除id

all_features = pd.concat((data_trian.iloc[:, 1:-1], data_val.iloc[:, 1:]))

# 获取值为数字的索引

numeric = all_features.dtypes[all_features.dtypes != 'object'].index

# 归一化

all_features[numeric] = all_features[numeric].apply(lambda x: ((x - x.mean()) / x.std()))

# 将缺失值补全

all_features[numeric] = all_features[numeric].fillna(0)

# 将单一变量变成独热向量

all_features = pd.get_dummies(all_features, dummy_na=True)

# 将数据转为pytorch中的张量

feature_train = torch.tensor(all_features[:data_trian.shape[0]].values.astype(float), dtype=torch.float32)

feature_test: torch.Tensor = torch.tensor(all_features[data_trian.shape[0]:].values.astype(float), dtype=torch.float32)

# 提取标签

train_label = torch.tensor(data_trian['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)


# 定义模型

def get_net():
    return nn.Sequential(nn.Linear(feature_train.shape[1], 1))


# 定义损失函数

loss = nn.MSELoss()


# 评估损失

def los_rmse(net, feature, label):
    clipped_pred = torch.clamp(net(feature), 1, float('inf'))

    rmse = torch.sqrt(loss(torch.log(clipped_pred), torch.log(label)))

    return rmse


def train(net: nn.Sequential, train_features, train_label,

          test_featere, test_label, epoch_num,

          learn_rate, weight_decy, batch_size):
    train_ls = []

    test_ls = []

    train_iter = data.DataLoader(data.TensorDataset(train_features, train_label), batch_size=batch_size, shuffle=True)

    # 优化器

    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate, weight_decay=weight_decy)

    for i in range(epoch_num):

        for X, y in train_iter:
            optimizer.zero_grad()

            l: torch.Tensor = loss(net(X), y)

            l.backward()

            optimizer.step()

        train_ls.append(los_rmse(net, train_features, train_label))

        if test_label is not None:
            test_ls.append(los_rmse(net, test_featere, test_label))

    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """

    这个函数是把训练集拆成训练集和验证集

    k是总折数，i是第几折。X是总样本，y是总标签

    """

    fold_size = X.shape[0] // k

    X_train, Y_train = None, None

    for j in range(k):

        idx = slice(j * fold_size, (j + 1) * fold_size)

        X_part, Y_part = X[idx, :], y[idx]

        if j == i:

            X_valid, Y_valid = X_part, Y_part

        elif X_train is None:

            X_train, Y_train = X_part, Y_part

        else:

            X_train = torch.cat((X_train, X_part), 0)

            Y_train = torch.cat((Y_train, Y_part), 0)

    return X_train, Y_train, X_valid, Y_valid


def k_fold(k, train_features, train_labels, learn_rate, epoch_num, weight_decy, batch_size):
    # 总损失

    train_l_sum, valid_l_sum = 0, 0

    # 开始了，训练k次

    for i in range(k):
        data = get_k_fold_data(k, i, train_features, train_labels)

        net = get_net()

        train_ls, valid_ls = train(net, *data, epoch_num=epoch_num, weight_decy=weight_decy, batch_size=batch_size,
                                   learn_rate=learn_rate)

        train_l_sum += train_ls[-1]

        valid_l_sum += valid_ls[-1]

        print(f'{i}折,训练log rsme{train_ls[-1]:f},验证lpg rsme{valid_ls[-1]:f}')

    return train_l_sum / k, valid_l_sum / k


# 定义超参数

learn_rate = 5

k = 5

epoch_num = 100

weigth_decy = 0

bacth_size = 64


# tl,vl = k_fold(k,feature_train,train_label,learn_rate,epoch_num,weigth_decy,bacth_size)

# print(f"{k}折验证，平均训练log rsme {tl:f},平均验证log rsme {vl:f}")


def train_and_pred(train_featrue, train_label, test_feature: torch.Tensor, test_label, learn_rate,

                   epoch_num, weight_decy, batch_size):
    net: nn.Sequential = get_net()

    train_ls, _ = train(net, train_featrue, train_label, test_feature, test_label, epoch_num, learn_rate, weight_decy,
                        batch_size)

    print(f'训练rsme {train_ls[-1]:f}')

    pred = net(test_feature).detach().numpy()

    data_val['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])

    submission = pd.concat([data_val['Id'], data_val['SalePrice']], axis=1)

    submission.to_csv("submission.csv", index=False)


train_and_pred(feature_train, train_label, feature_test, None, learn_rate, epoch_num, weigth_decy, bacth_size)

