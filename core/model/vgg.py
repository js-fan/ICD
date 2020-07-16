from .layers import *

def vgg16_deeplab(x, name=None, lr_mult=1, reuse=None):
    name = '' if name is None else name

    x = ConvRelu(x, 64, (3, 3), pad=(1, 1), name=name+'conv1_1', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 64, (3, 3), pad=(1, 1), name=name+'conv1_2', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=name+'pool1')

    x = ConvRelu(x, 128, (3, 3), pad=(1, 1), name=name+'conv2_1', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 128, (3, 3), pad=(1, 1), name=name+'conv2_2', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=name+'pool2')

    x = ConvRelu(x, 256, (3, 3), pad=(1, 1), name=name+'conv3_1', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 256, (3, 3), pad=(1, 1), name=name+'conv3_2', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 256, (3, 3), pad=(1, 1), name=name+'conv3_3', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=name+'pool3')

    x = ConvRelu(x, 512, (3, 3), pad=(1, 1), name=name+'conv4_1', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 512, (3, 3), pad=(1, 1), name=name+'conv4_2', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 512, (3, 3), pad=(1, 1), name=name+'conv4_3', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=name+'pool4')

    x = ConvRelu(x, 512, (3, 3), dilate=(2, 2), pad=(2, 2), name=name+'conv5_1', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 512, (3, 3), dilate=(2, 2), pad=(2, 2), name=name+'conv5_2', lr_mult=lr_mult, reuse=reuse)
    x = ConvRelu(x, 512, (3, 3), dilate=(2, 2), pad=(2, 2), name=name+'conv5_3', lr_mult=lr_mult, reuse=reuse)
    x = Pool(x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=name+'pool5')
    x = Pool(x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=name+'pool5a', pool_type='avg')
    return x

def vgg16_largefov(x, num_cls, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name

    x = vgg16_deeplab(x, name, lr_mult=1, reuse=reuse)

    x = ConvRelu(x, 1024, (3, 3), dilate=(12, 12), pad=(12, 12), name=name+'fc6', reuse=reuse)
    x = Drop(x, 0.5, name=name+'drop6')

    x = ConvRelu(x, 1024, (1, 1), name=name+'fc7', reuse=reuse)
    x = Drop(x, 0.5, name=name+'drop7')

    x = Conv(x, num_cls, (1, 1), name=name+'fc8', lr_mult=lr_mult, reuse=reuse)
    return x

def vgg16_aspp(x, num_cls, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name

    x_backbone = vgg16_deeplab(x, name, lr_mult=1, reuse=reuse)
    
    x_aspp = []
    for d in (6, 12, 18, 24):
        x = ConvRelu(x_backbone, 1024, (3, 3), dilate=(d, d), pad=(d, d), name=name+'fc6_aspp%d'%d, reuse=reuse)
        x = Drop(x, 0.5)

        x = ConvRelu(x, 1024, (1, 1), name=name+'fc7_aspp%d'%d, reuse=reuse)
        x = Drop(x, 0.5)

        x = Conv(x, num_cls, (1, 1), name=name+'fc8_aspp%d'%d, lr_mult=lr_mult, reuse=reuse)
        x_aspp.append(x) 

    x = sum(x_aspp)
    return x

def vgg16_cam(x, num_cls, name=None, lr_mult=10, reuse=None):
    name = '' if name is None else name

    x = vgg16_deeplab(x, name, lr_mult=1, reuse=reuse)

    x = ConvRelu(x, 1024, (3, 3), pad=(1, 1), name=name+'fc6', reuse=reuse)
    x = Drop(x, 0.5, name=name+'drop6')

    x = ConvRelu(x, 1024, (1, 1), name=name+'fc7', reuse=reuse)
    x = Drop(x, 0.5, name=name+'drop7')

    x = Conv(x, num_cls, (1, 1), name=name+'fc8', lr_mult=lr_mult, reuse=reuse)
    return x

