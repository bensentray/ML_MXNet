from mxnet import nd
from mxnet.gluon import nn
# 1 创建神经网络
#嵌套使用nn.Block和nn.Sequential
class rec_MLP(nn.Block):
    def __init__(self,**kwargs):
        super(rec_MLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256,activation='relu'))
            self.net.add(nn.Dense(128,activation='relu'))
            self.dense = nn.Dense(64)

    def forward(self,x):
        return nd.relu(self.dense(self.net(x)))

'''
问题：如果把RecMLP改成self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]，forward就用for loop来实现，会有什么问题吗？
看源码后发现原因为: [nn.Dense(256), nn.Dense(128), nn.Dense(64)] 的 type 是 list, 而不是 Block,
这样就不会被自动注册到 Block 类的 self._children 属性, 导致 initialize 时在 self._children 找不到神经元, 无法初始化参数.
当执行 self.xxx = yyy 时, __setattr__ 方法会检测 yyy 是否为 Block 类型, 如果是则添加到 self._children 列表中.
当执行 initialize() 时, 会从 self._children 中找神经元.
代码如下：
'''
class rec_MLP2(nn.Block):
    def __init__(self,**kwargs):
        super(rec_MLP2, self).__init__(**kwargs)
        with self.name_scope():
            self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]

    def forward(self,x):
        for layer in self.denses:
            x = nn.relu(layer(x))
        return x

rec_mlp = nn.Sequential()
rec_mlp.add(rec_MLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)

rec_mlp2 = rec_MLP2()
rec_mlp2.initialize()
print(rec_mlp2)
