from mxnet import nd
# 模型参数保存与读取
#1 通过ndarray读写
x = nd.ones(3)
y = nd.zeros(4)
filename = "data/test1.params"
nd.save(filename, [x, y])

a,b = nd.load(filename)
print(a,b)

mydict = {"x": x, "y": y}
filename = "data/test2.params"
nd.save(filename, mydict)
c = nd.load(filename)
print(c)

#2 通过gluon的save_parms和load_parms读写
from mxnet.gluon import nn

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(2))
    return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2,10))
print(net(x))
filename = "data/mlp.params"
net.save_params(filename)
#读取参数应用到模型上
import mxnet as mx
net2 = get_net()
net2.load_params(filename, mx.cpu())  # FIXME, gluon will support default ctx later
print(net2(x))