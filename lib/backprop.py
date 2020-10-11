import copy

import chainer
import chainer.functions as F

from lib.functions import GuidedReLU


class BaseBackprop(object):

    def __init__(self, model):
        self.model = model
        #self.size = model.size
        self.xp = model.xp

    def backward(self, x, label, layer):
        with chainer.using_config('train', False):
            acts = self.model(self.xp.asarray(x), layers=[layer, 'prob'])

        acts['prob'].grad = self.xp.zeros_like(acts['prob'].data)
        if label == -1:
            acts['prob'].grad[:, acts['prob'].data.argmax()] = 1
        else:
            acts['prob'].grad[:, label] = 1

        self.model.cleargrads()
        acts['prob'].backward(retain_grad=True)
        #print(str(acts))

        return acts


class GradCAM(BaseBackprop):

    def __init__(self, model):
        super(GradCAM, self).__init__(model)

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        #print(str(acts))
        weights = self.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.xp.tensordot(weights[0], acts[layer].data[0], axes=(0, 0))
        gcam = self.xp.maximum(gcam, 0)

        return chainer.cuda.to_cpu(gcam)


class GuidedBackprop(BaseBackprop):

    def __init__(self, model):
        super(GuidedBackprop, self).__init__(copy.deepcopy(model))
        _replace_relu(self.model)

    def generate(self, x, label, layer):
        acts = self.backward(x, label, layer)
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        gbp = gbp.transpose(1, 2, 0)

        return gbp


def _replace_relu(chain):
    for key, funcs in chain.functions.items():
        for i in range(len(funcs)):
            if hasattr(funcs[i], 'functions'):
                _replace_relu(funcs[i])
            elif funcs[i] is F.relu:
                funcs[i] = GuidedReLU()
