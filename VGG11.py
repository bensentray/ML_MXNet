import sys
sys.path.append('..')
import utils
from mxnet import image
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet import init
from mxnet.gluon import nn
#1读取数据-----------
def transform(data, label):
    # resize from 28 x 28 to 96 x 96
    data = image.imresize(data, 96, 96)
    return utils.transform_mnist(data, label)
batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size, transform)
#定义vgg_block
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                          padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
#定义vgg_stack函数
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
#2建立模型----------------------
ctx = utils.try_gpu()
num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))

net.initialize(ctx=ctx, init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})

#3训练和测试
for epoch in range(1):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
