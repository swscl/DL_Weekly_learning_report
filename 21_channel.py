import torch
from d2l import torch as d2l

# 多输入通道的互相关运算
def corr2d_muti_in(x,k):
    return sum (d2l.corr2d(x,k) for x ,k in zip(x,k))

def corr2d_muti_in_out(x,k):
    return torch.stack([corr2d_muti_in(x,k) for k in k],0)


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