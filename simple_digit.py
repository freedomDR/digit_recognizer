#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from mxnet import gluon
import mxnet as mx
import numpy as np

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:4000, 1:].values
labels = labeled_images.iloc[:4000, :1].values
test_images = labeled_images.iloc[4000:, 1:].values
test_labels = labeled_images.iloc[4000:, :1].values

predict_images = pd.read_csv('test.csv')
predict_data = mx.nd.array(predict_images.values.astype(np.float32)/255.)
ctx = mx.gpu()

batch_size = 10
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(images.astype(np.float32)/255., labels.astype(np.float32)), batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_images.astype(np.float32)/255., test_labels.astype(np.float32)), batch_size=batch_size, shuffle=False)
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(10))
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype(np.float32)).mean().asscalar()
def evaluate_accuracy(data_iter, net):
    # acc = 0
    acc = mx.metric.Accuracy()
    for X, y in data_iter:
        X = X.as_in_context(ctx).reshape((-1, 784))
        y = y.as_in_context(ctx).reshape(-1)
        output = net(X)
        predictions = mx.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=y)
        # acc += accuracy(net(X), y)
    # return acc/len(data_iter)
    return acc.get()[1]
print(evaluate_accuracy(test_data, net))
num_epochs = 10
for epoch in range(num_epochs):
    train_l_sum = 0
    train_acc = 0
    for X, y in train_data:
        X = X.as_in_context(ctx).reshape((-1, 784))
        y = y.as_in_context(ctx).reshape(-1)
        with mx.autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch_size)
        train_l_sum += mx.nd.mean(l).asscalar()
    train_acc = evaluate_accuracy(train_data, net)
    test_acc = evaluate_accuracy(test_data, net)
    print('epoch {0}, loss {1:.4f}, train acc {2:.3f}, test acc {3:.3f}'.format(epoch+1, train_l_sum/len(train_data), train_acc, test_acc))

predict_data = predict_data.as_in_context(ctx)
preds = net(predict_data).argmax(axis=1).astype(np.int).asnumpy()
print(preds.shape)
ans = pd.DataFrame(preds, index=range(1,len(preds)+1), columns=['Label'])
ans.to_csv('ans.csv', index_label='ImageId')
