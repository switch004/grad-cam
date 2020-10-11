import collections

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import numpy as np
from lib import utils

G =51#書き換える

class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 40, 5, stride=1),
            conv2 = L.Convolution2D(40, 50, 5, pad=0),
            l1 = L.Linear(None, 500),
            l2 = L.Linear(500, 500),
            l3 = L.Linear(500, G)
        )


        serializers.load_npz('C:\\Users\\Iitsuka\\Desktop\\model.npz', self)

        #self.size = 70 
        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, F.relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2', [self.conv2, F.relu]),
            ('pool2', [_max_pooling_2d]),
            ('l1', [self.l1, F.relu]),
            ('l2', [self.l2, F.relu]),
            ('l3', [self.l3]),
            ('prob', [_softmax]),
        ])

    def __call__(self, x, layers=['prob']):
        h = chainer.Variable(x)
        activations = {'input': h}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2, stride=2)

def _softmax(x):
    max_abs = np.max([np.abs(np.max(x.data)), np.abs(np.min(x.data))])
    _x = x
    _x.data /= max_abs
    h = F.softmax(_x)
    return h
