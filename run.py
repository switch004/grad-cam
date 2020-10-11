import argparse

import chainer
import cv2
import numpy as np
import os 

from lib import backprop
import models


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='images/dog_cat.png')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--arch', '-a',
               choices=['alex', 'vgg', 'resnet', 'CNN'],
               default='CNN')
p.add_argument('--label', '-y', type=int, default=-1)
p.add_argument('--layer', '-l', default='conv2')
args = p.parse_args()


if __name__ == '__main__':
    #必要なら画像の高さ，幅を変更
    h = 75
    w = 65
    if args.arch == 'alex':
        model = models.Alex()
    elif args.arch == 'vgg':
        model = models.VGG16Layers()
    elif args.arch == 'resnet':
        model = models.ResNet152Layers()
    elif args.arch == 'CNN':
        model = models.CNN()

    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    grad_cam = backprop.GradCAM(model)
    guided_backprop = backprop.GuidedBackprop(model)

    src = imread(args.input, 1)
    src = cv2.resize(src, (w, h))
    if args.arch == 'vgg':
        x = src.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
    else:
        x = src.astype(np.float32)
    x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]
    #x = x.transpose(0, 1, 2)[np.newaxis, :, :, :]

    print(x.shape)

    gcam = grad_cam.generate(x, args.label, args.layer)
    gcam = np.uint8(gcam * 255 / gcam.max())
    gcam = cv2.resize(gcam, (w, h))
    gbp = guided_backprop.generate(x, args.label, args.layer)

    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()
    imwrite('ggcam.png', ggcam)

    gbp -= gbp.min()
    gbp = 255 * gbp / gbp.max()
    imwrite('gbp.png', gbp)

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(src) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    imwrite('gcam.png', gcam)
    
