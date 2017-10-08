from mxnet.gluon import nn
import sys
sys.path.append('..')
import utils
from mxnet import image
from mxnet import init
from mxnet import gluon
# 稍微简化过的alexnet 模型
'''
AlexNet包含8层变换，其中有五层卷积和两层全连接隐含层，以及一个输出层。
第一层中的卷积核大小是11×1111×11，接着第二层中的是5×55×5，之后都是3×33×3。
此外，第一，第二和第五个卷积层之后都跟了有重叠的大小为3×33×3，步距为2×22×2的池化操作。
紧接着卷积层，原版的AlexNet有每层大小为4096个节点的全连接层们。
这两个巨大的全连接层带来将近1GB的模型大小。
由于早期GPU显存的限制，最早的AlexNet包括了双数据流的设计，以让网络中一半的节点能存入一个GPU。
这两个数据流，也就是说两个GPU只在一部分层进行通信，这样达到限制GPU同步时的额外开销的效果。
'''
net = nn.Sequential()
with net.name_scope():
    net.add(
        # 第一阶段
        nn.Conv2D(channels=96, kernel_size=11,
                  strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第二阶段
        nn.Conv2D(channels=256, kernel_size=5,
                  padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第三阶段
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3,
                  padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第四阶段
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        # 第五阶段
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        # 第六阶段
        nn.Dense(10)   #imagenet有1000个分类，但是fashion_mnist只有10个分类
    )


#数据读取 并重新定义图像大小为ImageNet中的224x224大小
def transform(data, label):
    # resize from 28 x 28 to 224 x 224
    data = image.imresize(data, 224, 224)
    return utils.transform_mnist(data, label)
batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size, transform)


#配置GPU
ctx = utils.try_gpu()
#初始化网络
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.01}) 
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)