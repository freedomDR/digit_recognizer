#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from mxnet import gluon
import mxnet as mx
import numpy as np

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:, 1:].values
labels = labeled_images.iloc[:, :1].values
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]

batch_size = 10
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(images.astype(np.float32)/255, labels.astype(np.float32)), batch_size=batch_size, shuffle=True)
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(10))
net.initialize(mx.init.Normal(sigma=1.), ctx=ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 10
for epoch in range(num_epochs):
    train_l_sum = 0
    train_acc_sum = 0
    for X, y in train_data:
        X = X.as_in_context(ctx[0])
        y = y.as_in_context(ctx[0])
        with mx.autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch_size)
        train_l_sum += mx.nd.sum(l).asscalar()
    print('epoch {0}, loss {1:.4f}'.format(epoch+1, train_l_sum/len(train_data)))

predict_images = pd.read_csv('test.csv')
test_data = mx.nd.array(predict_images.values.astype(np.float32)/255)
test_data = test_data.as_in_context(ctx(0))
preds = net(test_data).asnumpy()
print(preds.shape)
# ans = pd.DataFrame(preds, index=range(1,len(preds)+1), columns=['Label'])
# ans.to_csv('ans.csv', index_label='ImageId')
