#!/usr/bin/python
# -*- coding:utf-8 -*-
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
#from mxnet import
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


#实战Kaggle比赛——使用Gluon预测房价和K折交叉验证
if __name__ == "__main__":
    #一、读入数据
    # train_data = pd.read_csv("../data/kaggle_house_pred_train.csv")
    # test_data = pd.read_csv("../data/kaggle_house_pred_test.csv")
    train_data = pd.read_csv("D:/myproj/ML/MXNet/data/kaggle_house_pred_train.csv")
    test_data = pd.read_csv("D:/myproj/ML/MXNet/data/kaggle_house_pred_test.csv")
    all_X = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                       test_data.loc[:,'MSSubClass':'SaleCondition']))
    #train_data.head()  # 可以查看（默认）前5行数据信息
    #rain_data.tail()  # 可以查看后10行数据信息
    #rain_data.column  # 查看各个特征的具体名称
    #rain_data.describe()  # rain_data['SalePrice'].describe()能获得某一列的基本统计特征
    #print train_data.shape
    #print test_data.shape
    #二、预处理数据
    #1、使用pandas对数值特征做标准化处理xi=(xi−Exi)/std(xi)
    numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
    all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x:(x-x.mean())/(x.std()))
    #print all_X.dtypes
    #2、现在把离散数据点转换成数值标签
    all_X = pd.get_dummies(all_X,dummy_na=True)
    #3、把缺失数据用本特征的平均值估计
    all_X = all_X.fillna(all_X.mean())
    #4、将数据转换一下格式
    num_train = train_data.shape[0]
    #print num_train
    X_train = all_X[:num_train].as_matrix()
    print(X_train.shape)
    X_test = all_X[num_train:].as_matrix()
    y_train = train_data.SalePrice.as_matrix()
    #print len(y_train)
    # print y_train
    #三、导入NDArray格式数据
    #1、为了便于和gluon交互，我们需要导入NDArray格式数据
    X_train = nd.array(X_train)
    print( X_train.shape)
    y_train = nd.array(y_train)
    y_train.reshape((num_train,1))
    X_test = nd.array(X_test)
    #2、把损失函数定义为平方误差
    square_loss = gluon.loss.L2Loss()
    #3、定义比赛中测量结果用的函数
    def get_rmse_log(net,X_train,y_train):
        num_train = X_train.shape[0]
        clipped_preds = nd.clip(net(X_train), 1, float('inf'))
        return np.sqrt(2 * nd.sum(square_loss(
            nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)
    #四、定义模型
    #我们将模型的定义放在一个函数里供多次调用。这是一个基本的线性回归模型
    def get_net():
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(331, activation="relu"))
            #net.add(gluon.nn.Dropout(0.2))
            #net.add(gluon.nn.Dense(331, activation="relu"))
            net.add(gluon.nn.Dense(1))
        net.initialize()
        return net
    def train(net,X_train,y_train,X_test,y_test,epochs,verbose_epoch,
              learning_rate,weight_decay):
        train_loss = []
        if X_test is not None:
            test_loss = []
        batch_size = 100
        dataset_train = gluon.data.ArrayDataset(X_train,y_train)
        data_iter_train = gluon.data.DataLoader(dataset_train,batch_size,shuffle=True)
        #优化
        trainer = gluon.Trainer(net.collect_params(),'adam',
                                {'learning_rate':learning_rate,'wd':weight_decay})
        # trainer = gluon.Trainer(net.collect_params(), 'sgd',
        #                         {'learning_rate': learning_rate, 'wd': weight_decay})
        net.collect_params().initialize(force_reinit = True)
        for epoch in range(epochs):
            for data,label in data_iter_train:
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output,label)
                loss.backward()
                trainer.step(batch_size)

                cur_train_loss = get_rmse_log(net,X_train,y_train)
            if epoch > verbose_epoch:
                print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
            train_loss.append(cur_train_loss)
            if X_test is not None:
                cur_test_loss = get_rmse_log(net, X_test, y_test)
                test_loss.append(cur_test_loss)
        plt.plot(train_loss)
        plt.legend(['train'])
        if X_test is not None:
            plt.plot(test_loss)
            plt.legend(['train', 'test'])
        #plt.show()
        if X_test is not None:
            return cur_train_loss, cur_test_loss
        else:
            return cur_train_loss
    #五、K折交叉验证
    #在K折交叉验证中，我们把初始采样分割成K个子样本，一个单独的子样本
    # 被保留作为验证模型的数据，其他K−1个样本用来训练。
    def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                           learning_rate, weight_decay):
        assert k > 1
        fold_size = X_train.shape[0] // k
        train_loss_sum = 0.0
        test_loss_sum = 0.0
        for test_i in range(k):
            X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
            y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

            val_train_defined = False
            for i in range(k):
                if i != test_i:
                    X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                    y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                    if not val_train_defined:
                        X_val_train = X_cur_fold
                        y_val_train = y_cur_fold
                        val_train_defined = True
                    else:
                        X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                        y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
            net = get_net()
            train_loss, test_loss = train(
                net, X_val_train, y_val_train, X_val_test, y_val_test,
                epochs, verbose_epoch, learning_rate, weight_decay)
            train_loss_sum += train_loss
            print("Test loss: %f" % test_loss)
            test_loss_sum += test_loss
        return train_loss_sum / k, test_loss_sum / k


    k = 5
    epochs = 100
    verbose_epoch = 95
    learning_rate = 0.1
    weight_decay = 0.0

    train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                               y_train, learning_rate, weight_decay)
    print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
          (k, train_loss, test_loss))


    def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
              weight_decay):
        net = get_net()
        train(net, X_train, y_train, None, None, epochs, verbose_epoch,
              learning_rate, weight_decay)
        preds = net(X_test).asnumpy()
        test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
        submission.to_csv('submission_jalas.csv', index=False)

    learn(epochs, verbose_epoch, X_train,y_train,test_data,learning_rate, weight_decay)